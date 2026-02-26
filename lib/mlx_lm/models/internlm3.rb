module MlxLm
  module Models
    module InternLM3
      class ModelArgs < BaseModelArgs
        field :model_type, default: "internlm3"
        field :hidden_size, default: 4096
        field :num_hidden_layers, default: 32
        field :intermediate_size, default: 11008
        field :num_attention_heads, default: 32
        field :rms_norm_eps, default: 1e-6
        field :vocab_size, default: 103168
        field :bias, default: false
        field :qkv_bias, default: false
        field :max_position_embeddings, default: 32768
        field :num_key_value_heads, default: nil
        field :rope_theta, default: 10_000.0
        field :rope_traditional, default: false
        field :rope_scaling, default: nil
        field :tie_word_embeddings, default: false

        def initialize(**kwargs)
          super
          @num_key_value_heads ||= @num_attention_heads

          return if @rope_scaling.nil?

          required_keys = %w[factor rope_type]
          missing = required_keys.reject { |k| _config_has_key?(k) }
          unless missing.empty?
            raise ArgumentError, "rope_scaling must contain keys #{required_keys}"
          end

          rope_type = _config_value("rope_type")
          unless %w[linear dynamic].include?(rope_type)
            raise ArgumentError, "rope_scaling 'rope_type' only supports 'linear' or 'dynamic'"
          end
        end

        private

        def _config_has_key?(key)
          return false unless @rope_scaling.respond_to?(:key?)

          @rope_scaling.key?(key) || @rope_scaling.key?(key.to_sym)
        end

        def _config_value(key, default = nil)
          return default unless _config_has_key?(key)

          if @rope_scaling.key?(key)
            @rope_scaling[key]
          else
            @rope_scaling[key.to_sym]
          end
        end
      end

      class DynamicNTKScalingRoPE < MLX::NN::Module
        def initialize(
          dims,
          max_position_embeddings: 2048,
          traditional: false,
          base: 10_000.0,
          scale: 1.0
        )
          super()
          @max_position_embeddings = max_position_embeddings
          @original_base = base
          @dims = dims
          @traditional = traditional
          @scale = scale
        end

        def call(x, offset: 0)
          seq_len = x.shape[-2] + offset
          if seq_len > @max_position_embeddings
            scaled_ctx = (@scale * seq_len.to_f / @max_position_embeddings) - (@scale - 1.0)
            base = @original_base * (scaled_ctx**(@dims.to_f / (@dims - 2)))
          else
            base = @original_base
          end

          MLX::Core.rope(x, @dims, @traditional, base, @scale, offset)
        end
      end

      class Attention < MLX::NN::Module
        def initialize(args)
          super()
          dim = args.hidden_size
          qkv_bias = args.qkv_bias
          @n_heads = args.num_attention_heads
          @n_kv_heads = args.num_key_value_heads
          @head_dim = args.hidden_size / @n_heads
          @scale = @head_dim**(-0.5)

          self.q_proj = MLX::NN::Linear.new(dim, @n_heads * @head_dim, bias: qkv_bias)
          self.k_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: qkv_bias)
          self.v_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: qkv_bias)
          self.o_proj = MLX::NN::Linear.new(@n_heads * @head_dim, dim, bias: qkv_bias)

          rope_scale = if args.rope_scaling && _config_value(args.rope_scaling, "rope_type") == "linear"
            1.0 / _config_value(args.rope_scaling, "factor").to_f
          else
            2.0
          end

          self.rope = DynamicNTKScalingRoPE.new(
            @head_dim,
            max_position_embeddings: args.max_position_embeddings,
            traditional: args.rope_traditional,
            base: args.rope_theta,
            scale: rope_scale
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
            keys, values = cache.update_and_fetch(keys, values)
          else
            queries = rope.call(queries)
            keys = rope.call(keys)
          end

          output = mx.scaled_dot_product_attention(queries, keys, values, @scale, mask)
          output = output.transpose([0, 2, 1, 3]).reshape([b, l, @n_heads * @head_dim])
          o_proj.call(output)
        end

        private

        def _config_value(config, key, default = nil)
          return default if config.nil? || !config.respond_to?(:key?)
          return config[key] if config.key?(key)

          config.fetch(key.to_sym, default)
        end
      end

      class MLP < MLX::NN::Module
        def initialize(dim, hidden_dim, bias)
          super()
          self.gate_proj = MLX::NN::Linear.new(dim, hidden_dim, bias: bias)
          self.down_proj = MLX::NN::Linear.new(hidden_dim, dim, bias: bias)
          self.up_proj = MLX::NN::Linear.new(dim, hidden_dim, bias: bias)
        end

        def call(x)
          down_proj.call(Activations.swiglu(gate_proj.call(x), up_proj.call(x)))
        end
      end

      class TransformerBlock < MLX::NN::Module
        def initialize(args)
          super()
          self.self_attn = Attention.new(args)
          self.mlp = MLP.new(args.hidden_size, args.intermediate_size, args.bias)
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

      class InternLM3Model < MLX::NN::Module
        def initialize(args)
          super()
          self.embed_tokens = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
          self.layers = Array.new(args.num_hidden_layers) { TransformerBlock.new(args) }
          self.norm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
        end

        def call(inputs, cache: nil)
          h = embed_tokens.call(inputs)
          layer_cache = cache || [nil] * layers.length

          mask = nil
          mask = "causal" if h.shape[1] > 1

          layers.each_with_index do |layer, i|
            h = layer.call(h, mask: mask, cache: layer_cache[i])
          end

          norm.call(h)
        end
      end

      class Model < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.model = InternLM3Model.new(args)
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
          weights.reject { |k, _| k.include?("attention.rope.inv_freq") }
        end

        def layers
          model.layers
        end
      end

      Models.register("internlm3", Model, ModelArgs)
    end
  end
end
