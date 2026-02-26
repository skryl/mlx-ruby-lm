require_relative "activations"

module MlxLm
  module Models
    module HunyuanV1Dense
      class ModelArgs < BaseModelArgs
        field :model_type, default: "hunyuan_v1_dense"
        field :vocab_size, default: 151_936
        field :hidden_size, default: 4096
        field :num_hidden_layers, default: 40
        field :intermediate_size, default: 12_288
        field :num_attention_heads, default: 32
        field :num_key_value_heads, default: 8
        field :rms_norm_eps, default: 1e-6
        field :rope_theta, default: 10_000.0
        field :max_position_embeddings, default: 32_768
        field :attention_bias, default: false
        field :use_qk_norm, default: true
        field :rope_scaling, default: nil
        field :tie_word_embeddings, default: false
        field :head_dim, default: nil

        def initialize(**kwargs)
          super
          @num_key_value_heads ||= @num_attention_heads
          @head_dim ||= @hidden_size / @num_attention_heads
          _validate_rope_scaling!
        end

        private

        def _validate_rope_scaling!
          return if @rope_scaling.nil?

          required_keys = %w[alpha factor type]
          missing = required_keys.reject { |key| _config_has_key?(key) }
          return if missing.empty?

          raise ArgumentError, "rope_scaling must contain keys #{required_keys}"
        end

        def _config_has_key?(key)
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
        def initialize(args)
          super()
          dim = args.hidden_size

          @n_heads = args.num_attention_heads
          @n_kv_heads = args.num_key_value_heads
          @head_dim = args.head_dim
          @scale = @head_dim**(-0.5)
          @use_qk_norm = args.use_qk_norm

          self.q_proj = MLX::NN::Linear.new(dim, @n_heads * @head_dim, bias: args.attention_bias)
          self.k_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: args.attention_bias)
          self.v_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: args.attention_bias)
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

        def call(x, mask: nil, cache: nil)
          mx = MLX::Core
          b, l, _d = x.shape

          queries = q_proj.call(x)
          keys = k_proj.call(x)
          values = v_proj.call(x)

          queries = queries.reshape([b, l, @n_heads, @head_dim]).transpose([0, 2, 1, 3])
          keys = keys.reshape([b, l, @n_kv_heads, @head_dim]).transpose([0, 2, 1, 3])
          values = values.reshape([b, l, @n_kv_heads, @head_dim]).transpose([0, 2, 1, 3])

          if cache
            queries = rope.call(queries, offset: cache.offset)
            keys = rope.call(keys, offset: cache.offset)
          else
            queries = rope.call(queries)
            keys = rope.call(keys)
          end

          if @use_qk_norm
            queries = query_layernorm.call(queries)
            keys = key_layernorm.call(keys)
          end

          if cache
            keys, values = cache.update_and_fetch(keys, values)
          end

          output = mx.scaled_dot_product_attention(queries, keys, values, @scale, mask)
          output = output.transpose([0, 2, 1, 3]).reshape([b, l, @n_heads * @head_dim])
          o_proj.call(output)
        end

        private

        def _config_value(config, key, default = nil)
          return default if config.nil?
          return config[key] if config.key?(key)

          config.fetch(key.to_sym, default)
        end
      end

      class MLP < MLX::NN::Module
        def initialize(args)
          super()
          dim = args.hidden_size
          hidden_dim = args.intermediate_size

          self.gate_proj = MLX::NN::Linear.new(dim, hidden_dim, bias: false)
          self.down_proj = MLX::NN::Linear.new(hidden_dim, dim, bias: false)
          self.up_proj = MLX::NN::Linear.new(dim, hidden_dim, bias: false)
        end

        def call(x)
          down_proj.call(Activations.swiglu(gate_proj.call(x), up_proj.call(x)))
        end
      end

      class TransformerBlock < MLX::NN::Module
        def initialize(args)
          super()
          self.self_attn = Attention.new(args)
          self.mlp = MLP.new(args)
          self.input_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
          self.post_attention_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
        end

        def call(x, mask: nil, cache: nil)
          r = self_attn.call(input_layernorm.call(x), mask: mask, cache: cache)
          h = x + r
          r = mlp.call(post_attention_layernorm.call(h))
          h + r
        end
      end

      class HunyuanV1DenseModel < MLX::NN::Module
        def initialize(args)
          super()
          self.embed_tokens = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
          self.layers = Array.new(args.num_hidden_layers) { TransformerBlock.new(args) }
          self.norm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
        end

        def call(inputs, cache: nil)
          h = embed_tokens.call(inputs)
          layer_cache = cache || [nil] * layers.length
          mask = _create_attention_mask(h, layer_cache[0])

          layers.each_with_index do |layer, layer_idx|
            h = layer.call(h, mask: mask, cache: layer_cache[layer_idx])
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
          self.model = HunyuanV1DenseModel.new(args)
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
          result = weights.dup
          result.delete("lm_head.weight") if @args.tie_word_embeddings
          result
        end

        def layers
          model.layers
        end
      end

      Models.register("hunyuan_v1_dense", Model, ModelArgs)
    end
  end
end
