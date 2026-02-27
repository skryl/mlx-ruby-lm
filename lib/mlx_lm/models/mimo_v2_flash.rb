require_relative "activations"
require_relative "cache"
require_relative "rope_utils"
require_relative "switch_layers"

module MlxLm
  module Models
    module MimoV2Flash
      class ModelArgs < BaseModelArgs
        field :model_type, default: "mimo_v2_flash"
        field :num_experts_per_tok, default: 1
        field :hybrid_layer_pattern, default: nil
        field :moe_layer_freq, default: nil
        field :add_swa_attention_sink_bias, default: false
        field :add_full_attention_sink_bias, default: false
        field :sliding_window_size, default: 4096
        field :vocab_size
        field :hidden_size
        field :intermediate_size
        field :moe_intermediate_size
        field :num_hidden_layers
        field :num_attention_heads
        field :num_key_value_heads, default: nil
        field :n_shared_experts, default: nil
        field :n_routed_experts, default: nil
        field :routed_scaling_factor, default: nil
        field :topk_method, default: "noaux_tc"
        field :scoring_func, default: "sigmoid"
        field :norm_topk_prob, default: false
        field :n_group, default: 1
        field :topk_group, default: 1
        field :max_position_embeddings, default: 32768
        field :layernorm_epsilon, default: 1e-6
        field :rope_theta, default: 10_000.0
        field :swa_rope_theta, default: nil
        field :swa_num_attention_heads, default: nil
        field :swa_num_key_value_heads, default: nil
        field :head_dim, default: nil
        field :v_head_dim, default: nil
        field :swa_head_dim, default: nil
        field :swa_v_head_dim, default: nil
        field :partial_rotary_factor, default: 1.0

        def initialize(**kwargs)
          super
          @num_key_value_heads ||= @num_attention_heads
          @swa_num_attention_heads ||= @num_attention_heads
          @swa_num_key_value_heads ||= @num_key_value_heads

          @head_dim ||= @hidden_size / @num_attention_heads
          @v_head_dim ||= @head_dim
          @swa_head_dim ||= @head_dim
          @swa_v_head_dim ||= @swa_head_dim
          @swa_rope_theta ||= @rope_theta

          @n_routed_experts ||= 1
          @routed_scaling_factor = 1.0 if @routed_scaling_factor.nil?
          @hybrid_layer_pattern ||= Array.new(@num_hidden_layers, 0)
          @moe_layer_freq ||= Array.new(@num_hidden_layers, 0)
          @topk_group ||= @n_group
        end
      end

      class Attention < MLX::NN::Module
        def initialize(args, is_sliding_window)
          super()

          dim = args.hidden_size
          @is_sliding_window = is_sliding_window
          if @is_sliding_window
            @n_heads = args.swa_num_attention_heads
            @n_kv_heads = args.swa_num_key_value_heads
            @has_sinks = args.add_swa_attention_sink_bias
            @head_dim = args.swa_head_dim
            @v_head_dim = args.swa_v_head_dim
            rope_theta = args.swa_rope_theta
          else
            @n_heads = args.num_attention_heads
            @n_kv_heads = args.num_key_value_heads
            @has_sinks = args.add_full_attention_sink_bias
            @head_dim = args.head_dim
            @v_head_dim = args.v_head_dim
            rope_theta = args.rope_theta
          end

          @scale = @head_dim**(-0.5)

          self.q_proj = MLX::NN::Linear.new(dim, @n_heads * @head_dim, bias: false)
          self.k_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: false)
          self.v_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @v_head_dim, bias: false)
          self.o_proj = MLX::NN::Linear.new(@n_heads * @v_head_dim, dim, bias: false)
          self.attention_sink_bias = if @has_sinks
            MLX::Core.ones([@n_heads])
          else
            nil
          end

          rotary_dim = [(@head_dim * args.partial_rotary_factor.to_f).to_i, 1].max
          self.rope = MLX::NN::RoPE.new(
            rotary_dim,
            traditional: false,
            base: rope_theta
          )
        end

        def call(x, mask: nil, cache: nil)
          b, l, _d = x.shape

          queries = q_proj.call(x)
          keys = k_proj.call(x)
          values = v_proj.call(x)

          queries = queries.reshape([b, l, @n_heads, @head_dim]).transpose([0, 2, 1, 3])
          keys = keys.reshape([b, l, @n_kv_heads, @head_dim]).transpose([0, 2, 1, 3])
          values = values.reshape([b, l, @n_kv_heads, @v_head_dim]).transpose([0, 2, 1, 3])

          if cache
            queries = rope.call(queries, offset: cache.offset)
            keys = rope.call(keys, offset: cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
          else
            queries = rope.call(queries)
            keys = rope.call(keys)
          end

          output = _scaled_dot_product_attention(queries, keys, values, mask)
          output = output.transpose([0, 2, 1, 3]).reshape([b, l, @n_heads * @v_head_dim])
          o_proj.call(output)
        end

        private

        def _scaled_dot_product_attention(queries, keys, values, mask)
          mx = MLX::Core

          if attention_sink_bias
            begin
              return mx.scaled_dot_product_attention(
                queries,
                keys,
                values,
                @scale,
                mask,
                sinks: attention_sink_bias
              )
            rescue StandardError
              # Fallback when sinks are unsupported by the local MLX runtime.
            end
          end

          mx.scaled_dot_product_attention(queries, keys, values, @scale, mask)
        end
      end

      class MLP < MLX::NN::Module
        def initialize(config, hidden_size: nil, intermediate_size: nil)
          super()
          @hidden_size = hidden_size || config.hidden_size
          @intermediate_size = intermediate_size || config.intermediate_size

          self.gate_proj = MLX::NN::Linear.new(@hidden_size, @intermediate_size, bias: false)
          self.up_proj = MLX::NN::Linear.new(@hidden_size, @intermediate_size, bias: false)
          self.down_proj = MLX::NN::Linear.new(@intermediate_size, @hidden_size, bias: false)
        end

        def call(x)
          down_proj.call(Activations.swiglu(gate_proj.call(x), up_proj.call(x)))
        end
      end

      module_function

      def group_expert_select(
        gates,
        e_score_correction_bias,
        top_k,
        n_group,
        topk_group,
        routed_scaling_factor,
        norm_topk_prob
      )
        mx = MLX::Core

        scores = mx.sigmoid(gates.astype(mx.float32))
        orig_scores = scores
        scores = scores + e_score_correction_bias

        if n_group.to_i > 1
          experts_per_group = scores.shape[-1] / n_group
          scores = mx.unflatten(scores, -1, [n_group, experts_per_group])
          group_scores = mx.topk(scores, 2, -1)
          group_scores = mx.expand_dims(mx.sum(group_scores, -1), -1)

          drop_count = n_group - topk_group.to_i
          if drop_count > 0
            group_idx = mx.argpartition(group_scores, drop_count - 1, -2)
            take_ids = mx.array((0...drop_count).to_a, dtype: mx.int32)
            group_idx = mx.take(group_idx, take_ids, -2)
            scores = mx.put_along_axis(
              scores,
              mx.stop_gradient(group_idx),
              mx.array(0.0),
              -2
            )
          end

          scores = mx.flatten(scores, -2, -1)
        end

        k = [top_k.to_i, scores.shape[-1]].min
        inds = mx.argpartition(scores * -1.0, k - 1, -1)
        take_ids = mx.array((0...k).to_a, dtype: mx.int32)
        inds = mx.take(inds, take_ids, -1)

        selected_scores = mx.take_along_axis(orig_scores, inds, -1)
        if k > 1 && norm_topk_prob
          denominator = mx.expand_dims(mx.sum(selected_scores, -1), -1)
          selected_scores = selected_scores / (denominator + 1e-20)
        end

        selected_scores = selected_scores * routed_scaling_factor.to_f
        [inds, selected_scores]
      end

      class MoEGate < MLX::NN::Module
        def initialize(config)
          super()
          @top_k = config.num_experts_per_tok
          @norm_topk_prob = config.norm_topk_prob
          @n_routed_experts = config.n_routed_experts
          @routed_scaling_factor = config.routed_scaling_factor || 1.0
          @n_group = config.n_group
          @topk_group = config.topk_group

          raise ArgumentError, "Unsupported topk method: #{config.topk_method}" unless config.topk_method == "noaux_tc"

          mx = MLX::Core
          self.weight = mx.zeros([@n_routed_experts, config.hidden_size])
          self.e_score_correction_bias = mx.zeros([@n_routed_experts])
        end

        def call(x)
          mx = MLX::Core
          gates = mx.matmul(x, mx.transpose(weight))
          MimoV2Flash.group_expert_select(
            gates,
            e_score_correction_bias,
            @top_k,
            @n_group,
            @topk_group,
            @routed_scaling_factor,
            @norm_topk_prob
          )
        end
      end

      class MoE < MLX::NN::Module
        def initialize(config)
          super()
          @config = config

          self.switch_mlp = SwitchLayers::SwitchGLU.new(
            config.hidden_size,
            config.moe_intermediate_size,
            config.n_routed_experts
          )

          self.gate = MoEGate.new(config)
          if config.n_shared_experts
            shared_intermediate = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = MLP.new(config, intermediate_size: shared_intermediate)
          end
        end

        def call(x)
          mx = MLX::Core
          inds, scores = gate.call(x)
          y = switch_mlp.call(x, inds)
          y = mx.sum(y * mx.expand_dims(scores, -1), -2).astype(y.dtype)
          y = y + shared_experts.call(x) if @config.n_shared_experts
          y
        end
      end

      class DecoderLayer < MLX::NN::Module
        attr_reader :is_sliding_window

        def initialize(config, is_moe, is_sliding_window)
          super()
          @is_sliding_window = is_sliding_window

          self.self_attn = Attention.new(config, is_sliding_window)
          self.mlp = is_moe ? MoE.new(config) : MLP.new(config)
          self.input_layernorm = MLX::NN::RMSNorm.new(
            config.hidden_size,
            eps: config.layernorm_epsilon
          )
          self.post_attention_layernorm = MLX::NN::RMSNorm.new(
            config.hidden_size,
            eps: config.layernorm_epsilon
          )
        end

        def call(x, mask: nil, cache: nil)
          r = self_attn.call(input_layernorm.call(x), mask: mask, cache: cache)
          h = x + r
          r = mlp.call(post_attention_layernorm.call(h))
          h + r
        end
      end

      class LanguageModel < MLX::NN::Module
        def initialize(config)
          super()
          @hybrid_layer_pattern = config.hybrid_layer_pattern
          @sliding_window_size = config.sliding_window_size

          self.embed_tokens = MLX::NN::Embedding.new(config.vocab_size, config.hidden_size)
          self.layers = Array.new(config.num_hidden_layers) do |idx|
            DecoderLayer.new(
              config,
              config.moe_layer_freq[idx] == 1,
              config.hybrid_layer_pattern[idx] == 1
            )
          end
          self.norm = MLX::NN::RMSNorm.new(config.hidden_size, eps: config.layernorm_epsilon)
          self.swa_idx = @hybrid_layer_pattern.index(1) || 0
          self.ga_idx = @hybrid_layer_pattern.index(0) || 0
        end

        def call(x, cache: nil)
          h = embed_tokens.call(x)
          layer_cache = cache || [nil] * layers.length

          full_mask = _create_attention_mask(h, layer_cache[ga_idx])
          swa_mask = _create_attention_mask(
            h,
            layer_cache[swa_idx],
            window_size: @sliding_window_size
          )

          layers.each_with_index do |layer, i|
            mask = layer.is_sliding_window ? swa_mask : full_mask
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
              offset = cache.offset if cache.respond_to?(:offset)
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
        def initialize(config)
          super()
          @args = config
          self.model_type = config.model_type
          self.model = LanguageModel.new(config)
          self.lm_head = MLX::NN::Linear.new(config.hidden_size, config.vocab_size, bias: false)
        end

        def call(inputs, cache: nil)
          out = model.call(inputs, cache: cache)
          lm_head.call(out)
        end

        def sanitize(weights)
          mx = MLX::Core
          new_weights = {}

          weights.each do |k, v|
            if k.include?("weight_scale_inv")
              wk = k.sub("_scale_inv", "")
              if weights.key?(wk)
                new_weights[wk] = _dequant(weights[wk], v)
              end
            elsif !new_weights.key?(k)
              new_weights[k] = v
            end
          end

          result = new_weights
          @args.num_hidden_layers.times do |layer_idx|
            prefix = "model.layers.#{layer_idx}"
            %w[gate_proj down_proj up_proj].each do |proj|
              %w[weight scales biases].each do |param|
                first_key = "#{prefix}.mlp.experts.0.#{proj}.#{param}"
                next unless result.key?(first_key)

                expert_keys = (0...@args.n_routed_experts).map do |expert_idx|
                  "#{prefix}.mlp.experts.#{expert_idx}.#{proj}.#{param}"
                end
                next unless expert_keys.all? { |key| result.key?(key) }

                stacked = expert_keys.map { |key| result.delete(key) }
                result["#{prefix}.mlp.switch_mlp.#{proj}.#{param}"] = mx.stack(stacked)
              end
            end
          end

          result.reject { |k, _| k.start_with?("model.mtp") }
        end

        def layers
          model.layers
        end

        def cast_predicate
          lambda { |k| !k.include?("e_score_correction_bias") }
        end

        def make_cache
          layers.map do |layer|
            if layer.is_sliding_window
              MlxLm::RotatingKVCache.new(max_size: @args.sliding_window_size)
            else
              MlxLm::KVCache.new
            end
          end
        end

        private

        def _dequant(weight, scale_inv)
          mx = MLX::Core
          dtype = mx.bfloat16
          block_size = 128

          dequantized = mx.from_fp8(weight, dtype: dtype)
          m, n = dequantized.shape
          pad_bottom = block_size * scale_inv.shape[0] - m
          pad_side = block_size * scale_inv.shape[1] - n

          dequantized = mx.pad(dequantized, [[0, pad_bottom], [0, pad_side]])
          dequantized = dequantized.reshape([
            (m + pad_bottom) / block_size,
            block_size,
            (n + pad_side) / block_size,
            block_size,
          ])

          scaled = dequantized * scale_inv.reshape([scale_inv.shape[0], 1, scale_inv.shape[1], 1])
          scaled = scaled.reshape([m + pad_bottom, n + pad_side])
          scaled = mx.split(scaled, [m], 0)[0]
          scaled = mx.split(scaled, [n], 1)[0]
          scaled.astype(dtype)
        rescue StandardError
          weight
        end
      end

      Models.register("mimo_v2_flash", Model, ModelArgs)
    end
  end
end
