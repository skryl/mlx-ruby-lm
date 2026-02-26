require_relative "activations"
require_relative "cache"
require_relative "rope_utils"
require_relative "switch_layers"

module MlxLm
  module Models
    module Step3p5
      def self.clamped_swiglu(x, gate, limit)
        mx = MLX::Core
        clipped_gate = mx.minimum(MLX::NN.silu(gate), limit)
        clipped_x = mx.clip(x, -limit, limit)
        clipped_gate * clipped_x
      end

      class ModelArgs < BaseModelArgs
        field :model_type, default: "step3p5"
        field :hidden_size
        field :num_hidden_layers
        field :vocab_size
        field :num_attention_heads
        field :num_attention_groups
        field :head_dim
        field :intermediate_size
        field :rms_norm_eps, default: 1e-5
        field :rope_theta, default: 10_000.0
        field :rope_scaling, default: nil
        field :max_position_embeddings, default: 262_144
        field :sliding_window, default: 512
        field :layer_types, default: nil
        field :yarn_only_types, default: nil
        field :partial_rotary_factors, default: nil
        field :attention_other_setting, default: nil
        field :use_head_wise_attn_gate, default: true
        field :moe_num_experts, default: 288
        field :moe_top_k, default: 8
        field :moe_intermediate_size, default: 1280
        field :share_expert_dim, default: 1280
        field :moe_layers_enum, default: nil
        field :moe_router_scaling_factor, default: 3.0
        field :norm_expert_weight, default: true
        field :swiglu_limits, default: nil
        field :swiglu_limits_shared, default: nil
        field :tie_word_embeddings, default: false
      end

      class ZeroCenteredRMSNorm < MLX::NN::Module
        def initialize(dims, eps: 1e-5)
          super()
          self.weight = MLX::Core.ones([dims])
          @eps = eps
        end

        def call(x)
          mx = MLX::Core
          mean_sq = mx.mean(x * x, -1, keepdims: true)
          (x * mx.rsqrt(mean_sq + @eps)) * weight
        end
      end

      class Step3p5MLP < MLX::NN::Module
        def initialize(args, intermediate_size:, swiglu_limit: 0)
          super()
          @hidden_size = args.hidden_size
          @intermediate_size = intermediate_size

          self.gate_proj = MLX::NN::Linear.new(@hidden_size, @intermediate_size, bias: false)
          self.up_proj = MLX::NN::Linear.new(@hidden_size, @intermediate_size, bias: false)
          self.down_proj = MLX::NN::Linear.new(@intermediate_size, @hidden_size, bias: false)

          @limit = swiglu_limit && swiglu_limit > 0 ? swiglu_limit : nil
        end

        def call(x)
          if @limit
            return down_proj.call(
              Step3p5.clamped_swiglu(up_proj.call(x), gate_proj.call(x), @limit)
            )
          end

          down_proj.call(Activations.swiglu(gate_proj.call(x), up_proj.call(x)))
        end
      end

      class Step3p5MoEGate < MLX::NN::Module
        def initialize(args)
          super()
          @top_k = args.moe_top_k
          @n_routed_experts = args.moe_num_experts
          @routed_scaling_factor = args.moe_router_scaling_factor
          @norm_topk_prob = args.norm_expert_weight

          self.gate = MLX::NN::Linear.new(args.hidden_size, @n_routed_experts, bias: false)
          self.router_bias = MLX::Core.zeros([@n_routed_experts])
        end

        def call(x)
          _moe_gate_select(gate.call(x))
        end

        private

        def _moe_gate_select(gates)
          mx = MLX::Core
          scores = mx.sigmoid(gates.astype(mx.float32))
          corrected_scores = scores + router_bias

          k = [[@top_k.to_i, 1].max, @n_routed_experts].min
          topk_indices = mx.argpartition(corrected_scores * -1.0, k - 1, -1)
          take_ids = mx.array((0...k).to_a, dtype: mx.int32)
          topk_indices = mx.take(topk_indices, take_ids, -1)
          topk_weights = mx.take_along_axis(scores, topk_indices, -1)

          if @norm_topk_prob
            topk_weights = topk_weights / (mx.expand_dims(mx.sum(topk_weights, -1), -1) + 1e-20)
          end

          [topk_indices, topk_weights * @routed_scaling_factor]
        end
      end

      class Step3p5MoE < MLX::NN::Module
        def initialize(args, layer_idx)
          super()
          swiglu_limit = _limit_at(args.swiglu_limits, layer_idx)
          swiglu_limit_shared = _limit_at(args.swiglu_limits_shared, layer_idx)

          self.gate = Step3p5MoEGate.new(args)
          self.switch_mlp = SwitchLayers::SwitchGLU.new(
            args.hidden_size,
            args.moe_intermediate_size,
            args.moe_num_experts
          )
          self.share_expert = Step3p5MLP.new(
            args,
            intermediate_size: args.share_expert_dim,
            swiglu_limit: swiglu_limit_shared
          )

          @swiglu_limit = swiglu_limit
        end

        def call(x)
          mx = MLX::Core
          topk_indices, topk_weights = gate.call(x)

          routed_output = switch_mlp.call(x, topk_indices)
          routed_output = mx.sum(routed_output * mx.expand_dims(topk_weights, -1), -2).astype(routed_output.dtype)
          routed_output + share_expert.call(x)
        end

        private

        def _limit_at(values, idx)
          arr = Array(values)
          return 0 unless idx < arr.length

          arr[idx] || 0
        end
      end

      class Step3p5Attention < MLX::NN::Module
        attr_reader :is_sliding

        def initialize(args, layer_idx)
          super()
          dim = args.hidden_size
          layer_types = Array(args.layer_types)

          @is_sliding = if layer_types.empty?
            layer_idx.even?
          else
            layer_types[layer_idx] == "sliding_attention"
          end

          if @is_sliding && args.attention_other_setting
            settings = args.attention_other_setting
            @num_heads = _cfg_value(settings, "num_attention_heads", args.num_attention_heads)
            @num_kv_heads = _cfg_value(settings, "num_attention_groups", args.num_attention_groups)
          else
            @num_heads = args.num_attention_heads
            @num_kv_heads = args.num_attention_groups
          end

          @head_dim = args.head_dim
          @scale = @head_dim**(-0.5)

          self.q_proj = MLX::NN::Linear.new(dim, @num_heads * @head_dim, bias: false)
          self.k_proj = MLX::NN::Linear.new(dim, @num_kv_heads * @head_dim, bias: false)
          self.v_proj = MLX::NN::Linear.new(dim, @num_kv_heads * @head_dim, bias: false)
          self.o_proj = MLX::NN::Linear.new(@num_heads * @head_dim, dim, bias: false)

          self.q_norm = ZeroCenteredRMSNorm.new(@head_dim, eps: args.rms_norm_eps)
          self.k_norm = ZeroCenteredRMSNorm.new(@head_dim, eps: args.rms_norm_eps)

          @use_head_wise_attn_gate = args.use_head_wise_attn_gate
          self.g_proj = MLX::NN::Linear.new(dim, @num_heads, bias: false) if @use_head_wise_attn_gate

          rope_theta = args.rope_theta
          if rope_theta.is_a?(Array)
            rope_theta = rope_theta[layer_idx] || rope_theta[0]
          end

          partial_rotary_factor = _partial_rotary_factor(args.partial_rotary_factors, layer_idx)
          rope_dims = (@head_dim * partial_rotary_factor).to_i
          rope_dims = 1 if rope_dims < 1

          yarn_only_types = Array(args.yarn_only_types)
          layer_type = layer_types.empty? ? "full_attention" : layer_types[layer_idx]
          rope_scaling = if !yarn_only_types.empty? && !yarn_only_types.include?(layer_type)
            nil
          else
            args.rope_scaling
          end

          self.rope = MlxLm::Models.initialize_rope(
            rope_dims,
            rope_theta,
            false,
            rope_scaling,
            max_position_embeddings: args.max_position_embeddings
          )
        end

        def call(x, mask: nil, cache: nil)
          mx = MLX::Core
          b, l, _ = x.shape

          queries = q_proj.call(x)
          keys = k_proj.call(x)
          values = v_proj.call(x)

          queries = q_norm.call(queries.reshape([b, l, @num_heads, @head_dim])).transpose([0, 2, 1, 3])
          keys = k_norm.call(keys.reshape([b, l, @num_kv_heads, @head_dim])).transpose([0, 2, 1, 3])
          values = values.reshape([b, l, @num_kv_heads, @head_dim]).transpose([0, 2, 1, 3])

          if cache
            queries = rope.call(queries, offset: cache.offset)
            keys = rope.call(keys, offset: cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
          else
            queries = rope.call(queries)
            keys = rope.call(keys)
          end

          output = mx.scaled_dot_product_attention(queries, keys, values, @scale, mask)
          output = output.transpose([0, 2, 1, 3])

          if @use_head_wise_attn_gate
            output = output * mx.expand_dims(mx.sigmoid(g_proj.call(x)), -1)
          end

          o_proj.call(output.reshape([b, l, @num_heads * @head_dim]))
        end

        private

        def _partial_rotary_factor(factors, idx)
          arr = Array(factors)
          return 1.0 unless idx < arr.length

          arr[idx] || 1.0
        end

        def _cfg_value(hash, key, default = nil)
          return hash[key] if hash.key?(key)

          hash.fetch(key.to_sym, default)
        end
      end

      class Step3p5DecoderLayer < MLX::NN::Module
        attr_reader :is_sliding

        def initialize(args, layer_idx)
          super()
          self.self_attn = Step3p5Attention.new(args, layer_idx)
          @is_sliding = self_attn.is_sliding

          moe_layers_idx = _build_moe_layers_idx(args)
          is_moe_layer = moe_layers_idx[layer_idx]

          if is_moe_layer
            self.mlp = Step3p5MoE.new(args, layer_idx)
          else
            swiglu_limit = _limit_at(args.swiglu_limits_shared, layer_idx)
            self.mlp = Step3p5MLP.new(
              args,
              intermediate_size: args.intermediate_size,
              swiglu_limit: swiglu_limit
            )
          end

          self.input_layernorm = ZeroCenteredRMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
          self.post_attention_layernorm = ZeroCenteredRMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
        end

        def call(x, mask: nil, cache: nil)
          r = self_attn.call(input_layernorm.call(x), mask: mask, cache: cache)
          h = x + r
          h + mlp.call(post_attention_layernorm.call(h))
        end

        private

        def _build_moe_layers_idx(args)
          mapping = {}
          if args.moe_layers_enum
            args.moe_layers_enum.split(",").each do |idx|
              stripped = idx.strip
              next if stripped.empty?

              mapping[stripped.to_i] = true
            end
          else
            (1...args.num_hidden_layers).each { |idx| mapping[idx] = true }
          end
          mapping
        end

        def _limit_at(values, idx)
          arr = Array(values)
          return 0 unless idx < arr.length

          arr[idx] || 0
        end
      end

      class Step3p5Model < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          @num_layers = args.num_hidden_layers

          self.embed_tokens = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
          self.layers = Array.new(args.num_hidden_layers) { |layer_idx| Step3p5DecoderLayer.new(args, layer_idx) }
          self.norm = ZeroCenteredRMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)

          @swa_idx = layers.index(&:is_sliding)
          @full_idx = layers.index { |layer| !layer.is_sliding }
        end

        def call(inputs, cache: nil)
          h = embed_tokens.call(inputs)
          layer_cache = cache || [nil] * @num_layers

          full_mask = @full_idx.nil? ? nil : _create_attention_mask(h, layer_cache[@full_idx])
          swa_mask = if @swa_idx.nil?
            nil
          else
            _create_attention_mask(h, layer_cache[@swa_idx], window_size: @args.sliding_window)
          end

          layers.each_with_index do |layer, i|
            mask = layer.is_sliding ? swa_mask : full_mask
            h = layer.call(h, mask: mask, cache: layer_cache[i])
          end

          norm.call(h)
        end

        private

        def _create_attention_mask(h, cache = nil, window_size: nil)
          n = h.shape[1]
          if cache && cache.respond_to?(:make_mask)
            return cache.make_mask(n, window_size: window_size)
          end

          if window_size
            offset = 0
            if cache
              offset = cache.offset
              if cache.instance_variable_defined?(:@max_size)
                max_size = cache.instance_variable_get(:@max_size)
                offset = [max_size - 1, offset].min if max_size && max_size > 0
              end
            end
            return _create_causal_mask(n, offset: offset, window_size: window_size) if offset + n > window_size
          end
          return nil if n == 1

          "causal"
        end

        def _create_causal_mask(n, offset: 0, window_size: nil)
          mx = MLX::Core
          rinds = mx.arange(0, offset + n, 1, mx.int32).reshape([1, offset + n])
          linds = mx.arange(offset, offset + n, 1, mx.int32).reshape([n, 1])

          mask = mx.greater_equal(linds, rinds)
          if window_size
            mask = mx.logical_and(mask, mx.less(linds, mx.add(rinds, window_size)))
          end
          mask
        end
      end

      class Model < MLX::NN::Module
        attr_reader :args

        def initialize(args)
          super()
          @args = args
          self.model_type = args.model_type
          self.model = Step3p5Model.new(args)
          self.lm_head = MLX::NN::Linear.new(args.hidden_size, args.vocab_size, bias: false)
        end

        def call(inputs, cache: nil)
          lm_head.call(model.call(inputs, cache: cache))
        end

        def layers
          model.layers
        end

        def make_cache
          Array.new(layers.length) { MlxLm::KVCache.new }
        end

        def sanitize(weights)
          remappings = [
            [".moe.gate_proj.", ".mlp.switch_mlp.gate_proj."],
            [".moe.up_proj.", ".mlp.switch_mlp.up_proj."],
            [".moe.down_proj.", ".mlp.switch_mlp.down_proj."],
            [".moe.gate.", ".mlp.gate.gate."],
            [".moe.router_bias", ".mlp.gate.router_bias"],
            [".share_expert.", ".mlp.share_expert."],
          ]

          is_vanilla = weights.any? do |key, _|
            remappings.any? { |src, dst| key.include?(src) && !key.include?(dst) }
          end

          sanitized = {}
          weights.each do |key, value|
            next if key.include?(".mtp")

            if (match = key.match(/model\.layers\.(\d+)\./)) && match[1].to_i >= args.num_hidden_layers
              next
            end

            mapped_key = key
            remappings.each do |src, dst|
              if mapped_key.include?(src) && !mapped_key.include?(dst)
                mapped_key = mapped_key.gsub(src, dst)
                break
              end
            end

            mapped_value = value
            if is_vanilla && mapped_key.end_with?(".weight") && mapped_key.include?("norm")
              mapped_value = mapped_value + 1
            end

            sanitized[mapped_key] = mapped_value
          end

          sanitized
        end

        def cast_predicate
          ->(key) { !key.include?("router_bias") }
        end

        def quant_predicate
          lambda do |path, _|
            return {group_size: 64, bits: 8} if path.include?("mlp.gate.gate")

            true
          end
        end
      end

      Models.register("step3p5", Model, ModelArgs)
    end
  end
end
