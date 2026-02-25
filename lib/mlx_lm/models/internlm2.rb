module MlxLm
  module Models
    module InternLM2
      class ModelArgs < BaseModelArgs
        field :model_type, default: "internlm2"
        field :hidden_size, default: 4096
        field :num_hidden_layers, default: 32
        field :num_attention_heads, default: 32
        field :num_key_value_heads, default: nil
        field :intermediate_size, default: 11008
        field :vocab_size, default: 103168
        field :rms_norm_eps, default: 1e-6
        field :rope_theta, default: 10000.0
        field :rope_traditional, default: false
        field :rope_scaling, default: nil
        field :bias, default: true
        field :tie_word_embeddings, default: false
        field :max_position_embeddings, default: 32768

        def initialize(**kwargs)
          super
          @num_key_value_heads ||= @num_attention_heads
        end
      end

      class Attention < MLX::NN::Module
        def initialize(args)
          super()
          dim = args.hidden_size
          @n_heads = args.num_attention_heads
          @n_kv_heads = args.num_key_value_heads
          @head_dim = dim / @n_heads
          @scale = @head_dim**(-0.5)

          # Combined QKV projection
          total_qkv = (@n_heads + 2 * @n_kv_heads) * @head_dim
          self.wqkv = MLX::NN::Linear.new(dim, total_qkv, bias: args.bias)
          self.wo = MLX::NN::Linear.new(@n_heads * @head_dim, dim, bias: args.bias)

          self.rope = MLX::NN::RoPE.new(
            @head_dim,
            traditional: args.rope_traditional,
            base: args.rope_theta
          )
        end

        def call(x, mask: nil, cache: nil)
          mx = MLX::Core
          b, l, _d = x.shape

          qkv = wqkv.call(x)
          q_size = @n_heads * @head_dim
          kv_size = @n_kv_heads * @head_dim
          queries, keys, values = mx.split(qkv, [q_size, q_size + kv_size], 2)

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
          wo.call(output)
        end
      end

      class MLP < MLX::NN::Module
        def initialize(dim, hidden_dim)
          super()
          self.w1 = MLX::NN::Linear.new(dim, hidden_dim, bias: false)
          self.w2 = MLX::NN::Linear.new(hidden_dim, dim, bias: false)
          self.w3 = MLX::NN::Linear.new(dim, hidden_dim, bias: false)
        end

        def call(x)
          w2.call(MLX::NN.silu(w1.call(x)) * w3.call(x))
        end
      end

      class TransformerBlock < MLX::NN::Module
        def initialize(args)
          super()
          self.attention = Attention.new(args)
          self.feed_forward = MLP.new(args.hidden_size, args.intermediate_size)
          self.attention_norm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
          self.ffn_norm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
        end

        def call(x, mask: nil, cache: nil)
          r = attention.call(attention_norm.call(x), mask: mask, cache: cache)
          h = x + r
          r = feed_forward.call(ffn_norm.call(h))
          h + r
        end
      end

      class InternLM2Model < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.tok_embeddings = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
          self.layers = Array.new(args.num_hidden_layers) { TransformerBlock.new(args) }
          self.norm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
        end

        def call(inputs, cache: nil)
          h = tok_embeddings.call(inputs)
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
          self.model = InternLM2Model.new(args)
          unless args.tie_word_embeddings
            self.output = MLX::NN::Linear.new(args.hidden_size, args.vocab_size, bias: false)
          end
        end

        def call(inputs, cache: nil)
          out = model.call(inputs, cache: cache)
          if @args.tie_word_embeddings
            model.tok_embeddings.as_linear(out)
          else
            output.call(out)
          end
        end

        def sanitize(weights)
          weights.reject { |k, _| k.include?("attention.rope.inv_freq") }
        end

        def layers
          model.layers
        end
      end

      Models.register("internlm2", Model, ModelArgs)
    end
  end
end
