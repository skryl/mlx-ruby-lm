module MlxLm
  module Models
    module Cohere
      class ModelArgs < BaseModelArgs
        field :model_type, default: "cohere"
        field :hidden_size, default: 8192
        field :num_hidden_layers, default: 40
        field :num_attention_heads, default: 64
        field :num_key_value_heads, default: 64
        field :intermediate_size, default: 22528
        field :vocab_size, default: 256000
        field :rope_theta, default: 8000000.0
        field :layer_norm_eps, default: 1e-5
        field :logit_scale, default: 0.0625
        field :attention_bias, default: false
        field :layer_norm_bias, default: false
        field :use_qk_norm, default: false

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

          bias = args.attention_bias
          self.q_proj = MLX::NN::Linear.new(dim, @n_heads * @head_dim, bias: bias)
          self.k_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: bias)
          self.v_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: bias)
          self.o_proj = MLX::NN::Linear.new(@n_heads * @head_dim, dim, bias: bias)

          self.rope = MLX::NN::RoPE.new(@head_dim, traditional: true, base: args.rope_theta)
        end

        def call(x, mask: nil, cache: nil)
          mx = MLX::Core
          b, l, _d = x.shape

          queries = q_proj.call(x).reshape([b, l, @n_heads, @head_dim]).transpose([0, 2, 1, 3])
          keys = k_proj.call(x).reshape([b, l, @n_kv_heads, @head_dim]).transpose([0, 2, 1, 3])
          values = v_proj.call(x).reshape([b, l, @n_kv_heads, @head_dim]).transpose([0, 2, 1, 3])

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
      end

      class MLP < MLX::NN::Module
        def initialize(dim, hidden_dim)
          super()
          self.gate_proj = MLX::NN::Linear.new(dim, hidden_dim, bias: false)
          self.down_proj = MLX::NN::Linear.new(hidden_dim, dim, bias: false)
          self.up_proj = MLX::NN::Linear.new(dim, hidden_dim, bias: false)
        end

        def call(x)
          down_proj.call(MLX::NN.silu(gate_proj.call(x)) * up_proj.call(x))
        end
      end

      class TransformerBlock < MLX::NN::Module
        def initialize(args)
          super()
          self.self_attn = Attention.new(args)
          self.mlp = MLP.new(args.hidden_size, args.intermediate_size)
          self.input_layernorm = MLX::NN::LayerNorm.new(
            args.hidden_size, eps: args.layer_norm_eps
          )
        end

        def call(x, mask: nil, cache: nil)
          # Cohere uses parallel residuals: attn + mlp + x
          h = input_layernorm.call(x)
          attn_h = self_attn.call(h, mask: mask, cache: cache)
          ff_h = mlp.call(h)
          attn_h + ff_h + x
        end
      end

      class CohereModel < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.embed_tokens = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
          self.layers = Array.new(args.num_hidden_layers) { TransformerBlock.new(args) }
          self.norm = MLX::NN::LayerNorm.new(
            args.hidden_size, eps: args.layer_norm_eps
          )
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
          self.model = CohereModel.new(args)
        end

        def call(inputs, cache: nil)
          out = model.call(inputs, cache: cache)
          # Tied embeddings + logit scaling
          out = model.embed_tokens.as_linear(out)
          out * @args.logit_scale
        end

        def sanitize(weights)
          weights.reject { |k, _| k.include?("self_attn.rotary_emb.inv_freq") }
        end

        def layers
          model.layers
        end
      end

      Models.register("cohere", Model, ModelArgs)
    end
  end
end
