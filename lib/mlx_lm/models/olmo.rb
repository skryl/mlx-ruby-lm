module MlxLm
  module Models
    module OLMo
      class ModelArgs < BaseModelArgs
        field :model_type, default: "olmo"
        field :d_model, default: nil
        field :n_layers, default: nil
        field :mlp_hidden_size, default: nil
        field :n_heads, default: nil
        field :vocab_size, default: 50304
        field :embedding_size, default: nil
        field :rope_theta, default: 10000.0
        field :rope_traditional, default: false
        field :mlp_ratio, default: 4
        field :weight_tying, default: false

        # Compatibility aliases used in some generic tests/config builders.
        field :hidden_size, default: nil
        field :num_hidden_layers, default: nil
        field :intermediate_size, default: nil
        field :num_attention_heads, default: nil
        field :tie_word_embeddings, default: nil

        def initialize(**kwargs)
          super
          @d_model = @hidden_size if @hidden_size
          @n_layers = @num_hidden_layers if @num_hidden_layers
          @n_heads = @num_attention_heads if @num_attention_heads
          @mlp_hidden_size = @intermediate_size if @intermediate_size
          @weight_tying = @tie_word_embeddings unless @tie_word_embeddings.nil?

          @d_model ||= 4096
          @n_layers ||= 32
          @n_heads ||= 32
          @embedding_size ||= @vocab_size
          @mlp_hidden_size ||= @mlp_ratio * @d_model
        end
      end

      class TransformerBlock < MLX::NN::Module
        def initialize(args)
          super()
          dim = args.d_model
          @n_heads = args.n_heads
          @head_dim = dim / @n_heads
          @scale = @head_dim**(-0.5)
          @ff_hidden_size = args.mlp_hidden_size

          self.ff_proj = MLX::NN::Linear.new(dim, @ff_hidden_size, bias: false)
          self.ff_out = MLX::NN::Linear.new(@ff_hidden_size / 2, dim, bias: false)

          self.att_norm = MLX::NN::LayerNorm.new(dim, affine: false)
          self.ff_norm = MLX::NN::LayerNorm.new(dim, affine: false)

          self.att_proj = MLX::NN::Linear.new(dim, 3 * dim, bias: false)
          self.attn_out = MLX::NN::Linear.new(dim, dim, bias: false)

          self.rope = MLX::NN::RoPE.new(
            @head_dim,
            traditional: args.rope_traditional,
            base: args.rope_theta
          )
        end

        def attend(x, mask: nil, cache: nil)
          mx = MLX::Core
          b, l, d = x.shape

          qkv = att_proj.call(x)
          queries, keys, values = mx.split(qkv, [d, 2 * d], 2)

          queries = queries.reshape([b, l, @n_heads, @head_dim]).transpose([0, 2, 1, 3])
          keys = keys.reshape([b, l, @n_heads, @head_dim]).transpose([0, 2, 1, 3])
          values = values.reshape([b, l, @n_heads, @head_dim]).transpose([0, 2, 1, 3])

          if cache
            queries = rope.call(queries, offset: cache.offset)
            keys = rope.call(keys, offset: cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
          else
            queries = rope.call(queries)
            keys = rope.call(keys)
          end

          output = mx.scaled_dot_product_attention(queries, keys, values, @scale, mask)
          output = output.transpose([0, 2, 1, 3]).reshape([b, l, d])
          attn_out.call(output)
        end

        def call(x, mask: nil, cache: nil)
          mx = MLX::Core

          r = attend(att_norm.call(x), mask: mask, cache: cache)
          h = x + r

          ff_hidden = ff_proj.call(ff_norm.call(h))
          x1, x2 = mx.split(ff_hidden, [@ff_hidden_size / 2], 2)
          h + ff_out.call(Activations.swiglu(x2, x1))
        end
      end

      class Transformer < MLX::NN::Module
        def initialize(args)
          super()
          @weight_tying = args.weight_tying

          self.wte = MLX::NN::Embedding.new(args.embedding_size, args.d_model)
          self.blocks = Array.new(args.n_layers) { TransformerBlock.new(args) }
          self.ff_out = MLX::NN::Linear.new(args.d_model, args.embedding_size, bias: false) unless @weight_tying
          self.norm = MLX::NN::LayerNorm.new(args.d_model, affine: false)
        end

        def call(inputs, cache: nil)
          h = wte.call(inputs)
          layer_cache = cache || [nil] * blocks.length

          mask = nil
          mask = "causal" if h.shape[1] > 1

          blocks.each_with_index do |block, i|
            h = block.call(h, mask: mask, cache: layer_cache[i])
          end

          h = norm.call(h)

          if @weight_tying
            wte.as_linear(h)
          else
            ff_out.call(h)
          end
        end
      end

      class OlmoModel < MLX::NN::Module
        def initialize(args)
          super()
          self.transformer = Transformer.new(args)
        end

        def call(inputs, cache: nil)
          transformer.call(inputs, cache: cache)
        end
      end

      class Model < MLX::NN::Module
        def initialize(args)
          super()
          self.model_type = args.model_type
          self.model = OlmoModel.new(args)
          self.args = args
        end

        def call(inputs, cache: nil)
          model.call(inputs, cache: cache)
        end

        def layers
          model.transformer.blocks
        end
      end

      Models.register("olmo", Model, ModelArgs)
    end
  end
end
