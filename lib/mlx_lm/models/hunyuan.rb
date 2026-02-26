require_relative "activations"
require_relative "switch_layers"

module MlxLm
  module Models
    module Hunyuan
      module_function

      def int_or_list(value, idx)
        return value[idx] if value.is_a?(Array)

        value
      end

      class ModelArgs < BaseModelArgs
        field :model_type, default: "hunyuan"
        field :vocab_size
        field :hidden_size
        field :num_hidden_layers
        field :intermediate_size
        field :num_attention_heads
        field :num_key_value_heads, default: nil
        field :attention_bias
        field :moe_topk
        field :num_experts
        field :num_shared_expert
        field :use_mixed_mlp_moe
        field :use_qk_norm
        field :rms_norm_eps
        field :rope_theta
        field :use_cla
        field :cla_share_factor, default: 2
        field :moe_intermediate_size, default: nil
        field :rope_scaling, default: nil
        field :tie_word_embeddings, default: false

        def initialize(**kwargs)
          super
          @num_key_value_heads ||= @num_attention_heads
          _validate_rope_scaling!
        end

        private

        def _validate_rope_scaling!
          return if @rope_scaling.nil?

          required_keys = %w[factor type]
          return if required_keys.all? { |key| _rope_scaling_has_key?(key) }

          raise ArgumentError, "rope_scaling must contain keys #{required_keys}"
        end

        def _rope_scaling_has_key?(key)
          @rope_scaling.key?(key) || @rope_scaling.key?(key.to_sym)
        end
      end

      class DynamicNTKAlphaRoPE < MLX::NN::Module
        def initialize(dims, base: 10_000.0, scaling_alpha: 1.0)
          super()
          mx = MLX::Core

          @dims = dims
          adjusted_base = base * (scaling_alpha**(dims.to_f / (dims - 2)))
          self._freqs = mx.power(
            adjusted_base,
            mx.divide(mx.arange(0, dims, 2, mx.float32), dims.to_f)
          )
        end

        def call(x, offset: 0)
          MLX::Core.rope(x, @dims, false, nil, 1.0, offset, _freqs)
        end
      end

      class Attention < MLX::NN::Module
        def initialize(kv_proj, args)
          super()
          dim = args.hidden_size

          @kv_proj = kv_proj
          @n_heads = args.num_attention_heads
          @n_kv_heads = args.num_key_value_heads
          @head_dim = dim / @n_heads
          @scale = @head_dim**(-0.5)
          @use_qk_norm = args.use_qk_norm

          self.q_proj = MLX::NN::Linear.new(dim, @n_heads * @head_dim, bias: args.attention_bias)
          if kv_proj
            self.k_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: args.attention_bias)
            self.v_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: args.attention_bias)
          end
          self.o_proj = MLX::NN::Linear.new(@n_heads * @head_dim, dim, bias: args.attention_bias)

          if @use_qk_norm
            self.query_layernorm = MLX::NN::RMSNorm.new(@head_dim, eps: args.rms_norm_eps)
            self.key_layernorm = MLX::NN::RMSNorm.new(@head_dim, eps: args.rms_norm_eps)
          end

          scaling_alpha = _config_value(args.rope_scaling, "alpha", 1.0)
          self.rope = DynamicNTKAlphaRoPE.new(
            @head_dim,
            base: args.rope_theta,
            scaling_alpha: scaling_alpha
          )
        end

        def call(x, mask: nil, cache: nil, kv_states: nil)
          mx = MLX::Core
          b, l, _d = x.shape

          queries = q_proj.call(x)
          if kv_states
            keys, values = kv_states
          else
            raise ArgumentError, "kv_states required when kv_proj is disabled" unless @kv_proj

            keys = k_proj.call(x)
            values = v_proj.call(x)
            kv_states = [keys, values]
          end

          queries = queries.reshape([b, l, @n_heads, @head_dim]).transpose([0, 2, 1, 3])
          keys = keys.reshape([b, l, @n_kv_heads, @head_dim]).transpose([0, 2, 1, 3])
          values = values.reshape([b, l, @n_kv_heads, @head_dim]).transpose([0, 2, 1, 3])

          offset = cache ? cache.offset : 0
          queries = rope.call(queries, offset: offset)
          keys = rope.call(keys, offset: offset)

          if @use_qk_norm
            queries = query_layernorm.call(queries)
            keys = key_layernorm.call(keys)
          end

          keys, values = cache.update_and_fetch(keys, values) if cache

          output = mx.scaled_dot_product_attention(queries, keys, values, @scale, mask)
          output = output.transpose([0, 2, 1, 3]).reshape([b, l, @n_heads * @head_dim])
          [o_proj.call(output), kv_states]
        end

        private

        def _config_value(config, key, default = nil)
          return default if config.nil?
          return config[key] if config.key?(key)

          config.fetch(key.to_sym, default)
        end
      end

      class MLP < MLX::NN::Module
        def initialize(dim, hidden_dim)
          super()
          self.gate_proj = MLX::NN::Linear.new(dim, hidden_dim, bias: false)
          self.down_proj = MLX::NN::Linear.new(hidden_dim, dim, bias: false)
          self.up_proj = MLX::NN::Linear.new(dim, hidden_dim, bias: false)
        end

        def call(x)
          down_proj.call(Activations.swiglu(gate_proj.call(x), up_proj.call(x)))
        end
      end

      class Gate < MLX::NN::Module
        def initialize(dim, num_experts)
          super()
          self.wg = MLX::NN::Linear.new(dim, num_experts, bias: false)
        end

        def call(x)
          wg.call(x)
        end
      end

      class MoeBlock < MLX::NN::Module
        def initialize(args, layer_idx: 0)
          super()
          dim = args.hidden_size
          intermediate_size = args.intermediate_size

          @use_shared_mlp = args.use_mixed_mlp_moe
          if @use_shared_mlp
            num_shared = Hunyuan.int_or_list(args.num_shared_expert, layer_idx).to_i
            self.shared_mlp = MLP.new(dim, (intermediate_size * num_shared).to_i)
          end

          @num_experts = args.num_experts
          @top_k = Hunyuan.int_or_list(args.moe_topk, layer_idx).to_i
          self.gate = Gate.new(dim, @num_experts)

          expert_intermediate_size = args.moe_intermediate_size.nil? ?
            intermediate_size :
            Hunyuan.int_or_list(args.moe_intermediate_size, layer_idx)

          self.switch_mlp = SwitchLayers::SwitchGLU.new(
            dim,
            expert_intermediate_size,
            @num_experts
          )
        end

        def call(x)
          mx = MLX::Core

          gates = gate.call(x)
          gates = mx.softmax(gates.astype(mx.float32), -1).astype(gates.dtype)

          k = [[@top_k, 1].max, @num_experts].min
          inds = mx.stop_gradient(mx.argpartition(gates * -1.0, k - 1, -1))
          take_ids = mx.array((0...k).to_a, dtype: mx.int32)
          inds = mx.take(inds, take_ids, -1)
          scores = mx.take_along_axis(gates, inds, -1)

          y = switch_mlp.call(x, inds)
          y = mx.sum(y * mx.expand_dims(scores.astype(mx.float32), -1), -2).astype(y.dtype)

          y = y + shared_mlp.call(x) if @use_shared_mlp
          y
        end
      end

      class DecoderLayer < MLX::NN::Module
        def initialize(args, kv_proj:, layer_idx:)
          super()
          self.self_attn = Attention.new(kv_proj, args)
          if args.num_experts.to_i == 1
            self.mlp = MLP.new(args.hidden_size, args.intermediate_size)
          else
            self.mlp = MoeBlock.new(args, layer_idx: layer_idx)
          end
          self.input_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
          self.post_attention_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
        end

        def call(x, mask: nil, cache: nil, shared_kv_states: nil)
          r, shared_kv_states = self_attn.call(
            input_layernorm.call(x),
            mask: mask,
            cache: cache,
            kv_states: shared_kv_states
          )
          h = x + r
          r = mlp.call(post_attention_layernorm.call(h))
          [h + r, shared_kv_states]
        end
      end

      class HunYuanModel < MLX::NN::Module
        def initialize(args)
          super()
          @args = args

          self.embed_tokens = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
          self.layers = Array.new(args.num_hidden_layers) do |i|
            kv_proj = (!args.use_cla) || (i % args.cla_share_factor).zero?
            DecoderLayer.new(args, kv_proj: kv_proj, layer_idx: i)
          end
          self.norm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
        end

        def call(inputs, cache: nil)
          h = embed_tokens.call(inputs)
          layer_cache = cache || [nil] * layers.length
          mask = _create_attention_mask(h, layer_cache[0])

          shared_kv_states = nil
          layers.each_with_index do |layer, i|
            if (!@args.use_cla) || (i % @args.cla_share_factor).zero?
              shared_kv_states = nil
            end
            h, shared_kv_states = layer.call(
              h,
              mask: mask,
              cache: layer_cache[i],
              shared_kv_states: shared_kv_states
            )
          end

          norm.call(h)
        end

        private

        def _create_attention_mask(hidden, cache)
          return cache.make_mask(hidden.shape[1]) if cache && cache.respond_to?(:make_mask)
          return nil if hidden.shape[1] == 1

          "causal"
        end
      end

      class Model < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.model_type = args.model_type
          self.model = HunYuanModel.new(args)
          unless args.tie_word_embeddings
            self.lm_head = MLX::NN::Linear.new(args.hidden_size, args.vocab_size, bias: false)
          end
        end

        def call(inputs, cache: nil)
          out = model.call(inputs, cache: cache)
          if @args.tie_word_embeddings
            model.embed_tokens.as_linear(out)
          else
            lm_head.call(out)
          end
        end

        def sanitize(weights)
          mx = MLX::Core
          result = weights.dup

          if result.key?("model.layers.0.mlp.gate_and_up_proj.weight")
            new_weights = {}
            d = @args.hidden_size
            n_kv_heads = @args.num_key_value_heads
            n_kv_groups = @args.num_attention_heads / n_kv_heads
            head_dim = d / @args.num_attention_heads

            result.each do |key, value|
              if key.include?("qkv_proj")
                reshaped = value.reshape([n_kv_heads, n_kv_groups + 2, head_dim, -1])
                qkv_splits = mx.split(reshaped, [n_kv_groups, n_kv_groups + 1], 1)
                %w[q_proj k_proj v_proj].each_with_index do |proj, idx|
                  new_weights[key.sub("qkv_proj", proj)] = mx.flatten(qkv_splits[idx], 0, 2)
                end
              elsif key.include?("gate_and_up_proj")
                split_idx = value.shape[0] / 2
                up_proj, gate_proj = mx.split(value, [split_idx], 0)
                new_weights[key.sub("gate_and_up_proj", "up_proj")] = up_proj
                new_weights[key.sub("gate_and_up_proj", "gate_proj")] = gate_proj
              else
                new_weights[key] = value
              end
            end

            result = new_weights
          end

          if result.key?("model.layers.0.mlp.experts.0.up_proj.weight")
            @args.num_hidden_layers.times do |layer_idx|
              prefix = "model.layers.#{layer_idx}"
              %w[up_proj down_proj gate_proj].each do |projection|
                %w[weight scales biases].each do |param|
                  first_key = "#{prefix}.mlp.experts.0.#{projection}.#{param}"
                  next unless result.key?(first_key)

                  expert_keys = (0...@args.num_experts).map do |expert_idx|
                    "#{prefix}.mlp.experts.#{expert_idx}.#{projection}.#{param}"
                  end
                  next unless expert_keys.all? { |k| result.key?(k) }

                  stacked = expert_keys.map { |k| result.delete(k) }
                  result["#{prefix}.mlp.switch_mlp.#{projection}.#{param}"] = mx.stack(stacked)
                end
              end
            end
          end

          result.delete("lm_head.weight") if @args.tie_word_embeddings
          result
        end

        def layers
          model.layers
        end
      end

      Models.register("hunyuan", Model, ModelArgs)
    end
  end
end
