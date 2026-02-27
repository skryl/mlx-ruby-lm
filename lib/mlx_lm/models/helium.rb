module MlxLm
  module Models
    module Helium
      class ModelArgs < BaseModelArgs
        field :hidden_size, default: 256
        field :num_hidden_layers, default: 24
        field :intermediate_size, default: 1024
        field :num_attention_heads, default: 4
        field :num_key_value_heads, default: nil
        field :rms_norm_eps, default: 1e-5
        field :vocab_size, default: 32_000
        field :attention_bias, default: false
        field :head_dim, default: nil
        field :max_position_embeddings, default: 2048
        field :mlp_bias, default: false
        field :model_type, default: "helium"
        field :rope_theta, default: 10_000.0
        field :tie_word_embeddings, default: false

        def initialize(**kwargs)
          super
          @num_key_value_heads ||= @num_attention_heads
          @head_dim ||= @hidden_size / @num_attention_heads
        end
      end

      class Attention < MLX::NN::Module
        def initialize(args)
          super()

          dim = args.hidden_size
          @n_heads = args.num_attention_heads
          @n_kv_heads = args.num_key_value_heads
          @head_dim = args.hidden_size / @n_heads
          @scale = @head_dim**(-0.5)

          self.q_proj = MLX::NN::Linear.new(dim, @n_heads * @head_dim, bias: args.attention_bias)
          self.k_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: args.attention_bias)
          self.v_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: args.attention_bias)
          self.o_proj = MLX::NN::Linear.new(@n_heads * @head_dim, dim, bias: false)
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
        def initialize(args)
          super()
          self.gate_proj = MLX::NN::Linear.new(
            args.hidden_size,
            args.intermediate_size,
            bias: args.mlp_bias
          )
          self.up_proj = MLX::NN::Linear.new(
            args.hidden_size,
            args.intermediate_size,
            bias: args.mlp_bias
          )
          self.down_proj = MLX::NN::Linear.new(
            args.intermediate_size,
            args.hidden_size,
            bias: args.mlp_bias
          )
        end

        def call(x)
          down_proj.call(Activations.swiglu(gate_proj.call(x), up_proj.call(x)))
        end
      end

      class DecoderLayer < MLX::NN::Module
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

      class HeliumModel < MLX::NN::Module
        def initialize(args)
          super()
          self.embed_tokens = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
          self.layers = Array.new(args.num_hidden_layers) { DecoderLayer.new(args) }
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
          @model_type = args.model_type
          self.model = HeliumModel.new(args)
          self.lm_head = MLX::NN::Linear.new(args.hidden_size, args.vocab_size, bias: false)
        end

        def call(inputs, cache: nil)
          out = model.call(inputs, cache: cache)
          if @args.tie_word_embeddings
            model.embed_tokens.as_linear(out)
          else
            lm_head.call(out)
          end
        end

        def layers
          model.layers
        end
      end

      Models.register("helium", Model, ModelArgs)
    end
  end
end
