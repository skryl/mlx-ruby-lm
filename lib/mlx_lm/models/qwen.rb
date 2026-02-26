module MlxLm
  module Models
    module Qwen
      class ModelArgs < BaseModelArgs
        field :model_type, default: "qwen"
        field :hidden_size, default: 2048
        field :num_attention_heads, default: 16
        field :num_hidden_layers, default: 24
        field :kv_channels, default: 128
        field :max_position_embeddings, default: 8192
        field :layer_norm_epsilon, default: 1e-6
        field :intermediate_size, default: 11008
        field :no_bias, default: true
        field :vocab_size, default: 151936
        field :num_key_value_heads, default: nil

        def initialize(**kwargs)
          super
          @num_key_value_heads ||= @num_attention_heads
        end
      end

      class Attention < MLX::NN::Module
        def initialize(args)
          super()

          hidden_size = args.hidden_size
          @num_attention_heads = args.num_attention_heads
          hidden_size_per_attention_head = hidden_size / @num_attention_heads

          self.rotary_emb = MLX::NN::RoPE.new(
            hidden_size_per_attention_head,
            traditional: false
          )

          @proj_size = args.kv_channels * @num_attention_heads

          self.c_attn = MLX::NN::Linear.new(hidden_size, @proj_size * 3, bias: true)
          self.c_proj = MLX::NN::Linear.new(hidden_size, @proj_size, bias: !args.no_bias)

          @head_dim = args.kv_channels
          @scale = hidden_size_per_attention_head**(-0.5)
        end

        def call(x, mask: nil, cache: nil)
          mx = MLX::Core

          qkv = c_attn.call(x)
          q, k, v = mx.split(qkv, [@proj_size, 2 * @proj_size], -1)

          b, l, _ = q.shape

          queries = q.reshape([b, l, @num_attention_heads, @head_dim]).transpose([0, 2, 1, 3])
          keys = k.reshape([b, l, @num_attention_heads, @head_dim]).transpose([0, 2, 1, 3])
          values = v.reshape([b, l, @num_attention_heads, @head_dim]).transpose([0, 2, 1, 3])

          if cache
            queries = rotary_emb.call(queries, offset: cache.offset)
            keys = rotary_emb.call(keys, offset: cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
          else
            queries = rotary_emb.call(queries)
            keys = rotary_emb.call(keys)
          end

          output = mx.scaled_dot_product_attention(queries, keys, values, @scale, mask)
          output = output.transpose([0, 2, 1, 3]).reshape([b, l, @proj_size])

          c_proj.call(output)
        end
      end

      class MLP < MLX::NN::Module
        def initialize(args)
          super()

          self.w1 = MLX::NN::Linear.new(
            args.hidden_size,
            args.intermediate_size / 2,
            bias: !args.no_bias
          )
          self.w2 = MLX::NN::Linear.new(
            args.hidden_size,
            args.intermediate_size / 2,
            bias: !args.no_bias
          )
          self.c_proj = MLX::NN::Linear.new(
            args.intermediate_size / 2,
            args.hidden_size,
            bias: !args.no_bias
          )
        end

        def call(x)
          a1 = w1.call(x)
          a2 = w2.call(x)
          c_proj.call(Activations.swiglu(a2, a1))
        end
      end

      class TransformerBlock < MLX::NN::Module
        def initialize(args)
          super()

          self.ln_1 = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.layer_norm_epsilon)
          self.attn = Attention.new(args)
          self.ln_2 = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.layer_norm_epsilon)
          self.mlp = MLP.new(args)
        end

        def call(x, mask: nil, cache: nil)
          residual = x
          x = ln_1.call(x)
          x = attn.call(x, mask: mask, cache: cache)
          residual = x + residual
          x = ln_2.call(residual)
          x = mlp.call(x)
          x + residual
        end
      end

      class QwenModel < MLX::NN::Module
        def initialize(args)
          super()
          self.wte = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
          self.h = Array.new(args.num_hidden_layers) { TransformerBlock.new(args) }
          self.ln_f = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.layer_norm_epsilon)
        end

        def call(inputs, cache: nil)
          x = wte.call(inputs)
          layer_cache = cache || [nil] * h.length

          mask = nil
          mask = "causal" if x.shape[1] > 1

          h.each_with_index do |layer, i|
            x = layer.call(x, mask: mask, cache: layer_cache[i])
          end

          ln_f.call(x)
        end
      end

      class Model < MLX::NN::Module
        def initialize(config)
          super()
          @args = config
          self.model_type = config.model_type
          self.transformer = QwenModel.new(config)
          self.lm_head = MLX::NN::Linear.new(
            config.hidden_size,
            config.vocab_size,
            bias: !config.no_bias
          )
        end

        def call(x, cache: nil)
          y = transformer.call(x, cache: cache)
          lm_head.call(y)
        end

        def sanitize(weights)
          weights.reject { |k, _| k.include?("rotary_emb.inv_freq") }
        end

        def layers
          transformer.h
        end
      end

      Models.register("qwen", Model, ModelArgs)
    end
  end
end
