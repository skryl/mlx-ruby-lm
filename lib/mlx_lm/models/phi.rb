module MlxLm
  module Models
    module Phi
      class ModelArgs < BaseModelArgs
        field :model_type, default: "phi"
        field :max_position_embeddings, default: 2048
        field :vocab_size, default: 51_200
        field :hidden_size, default: 2560
        field :num_attention_heads, default: 32
        field :num_hidden_layers, default: 32
        field :num_key_value_heads, default: nil
        field :partial_rotary_factor, default: 0.4
        field :intermediate_size, default: 10_240
        field :layer_norm_eps, default: 1e-5
        field :rope_theta, default: 10_000.0

        def initialize(**kwargs)
          super
          @num_key_value_heads ||= @num_attention_heads
        end
      end

      class PhiAttention < MLX::NN::Module
        def initialize(args)
          super()
          @hidden_size = args.hidden_size
          @num_heads = args.num_attention_heads
          @head_dim = @hidden_size / @num_heads
          @num_key_value_heads = args.num_key_value_heads
          @scale = @head_dim**(-0.5)

          if (@head_dim * @num_heads) != @hidden_size
            raise ArgumentError,
              "hidden_size must be divisible by num_heads (hidden_size=#{@hidden_size}, num_heads=#{@num_heads})"
          end

          self.q_proj = MLX::NN::Linear.new(@hidden_size, @num_heads * @head_dim, bias: true)
          self.k_proj = MLX::NN::Linear.new(@hidden_size, @num_key_value_heads * @head_dim, bias: true)
          self.v_proj = MLX::NN::Linear.new(@hidden_size, @num_key_value_heads * @head_dim, bias: true)
          self.dense = MLX::NN::Linear.new(@num_heads * @head_dim, @hidden_size, bias: true)

          self.rope = MLX::NN::RoPE.new(
            (args.partial_rotary_factor * @head_dim).to_i,
            traditional: false,
            base: args.rope_theta
          )
        end

        def call(x, mask: nil, cache: nil)
          mx = MLX::Core
          queries = q_proj.call(x)
          keys = k_proj.call(x)
          values = v_proj.call(x)

          b, l, _d = queries.shape
          queries = queries.reshape([b, l, @num_heads, @head_dim]).transpose([0, 2, 1, 3])
          keys = keys.reshape([b, l, @num_key_value_heads, @head_dim]).transpose([0, 2, 1, 3])
          values = values.reshape([b, l, @num_key_value_heads, @head_dim]).transpose([0, 2, 1, 3])

          if cache
            queries = rope.call(queries, offset: cache.offset)
            keys = rope.call(keys, offset: cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
          else
            queries = rope.call(queries)
            keys = rope.call(keys)
          end

          output = mx.scaled_dot_product_attention(
            queries.astype(mx.float32),
            keys,
            values,
            @scale,
            mask
          ).astype(values.dtype)

          output = output.transpose([0, 2, 1, 3]).reshape([b, l, @num_heads * @head_dim])
          dense.call(output)
        end
      end

      class PhiMLP < MLX::NN::Module
        def initialize(args)
          super()
          self.fc1 = MLX::NN::Linear.new(args.hidden_size, args.intermediate_size, bias: true)
          self.fc2 = MLX::NN::Linear.new(args.intermediate_size, args.hidden_size, bias: true)
        end

        def call(x)
          fc2.call(MLX::NN.gelu_approx(fc1.call(x)))
        end
      end

      class PhiDecoderLayer < MLX::NN::Module
        def initialize(args)
          super()
          self.self_attn = PhiAttention.new(args)
          self.input_layernorm = MLX::NN::LayerNorm.new(args.hidden_size, eps: args.layer_norm_eps)
          self.mlp = PhiMLP.new(args)
        end

        def call(x, mask: nil, cache: nil)
          h = input_layernorm.call(x)
          attn_h = self_attn.call(h, mask: mask, cache: cache)
          ff_h = mlp.call(h)
          attn_h + ff_h + x
        end
      end

      class PhiModel < MLX::NN::Module
        def initialize(args)
          super()
          self.embed_tokens = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
          self.layers = Array.new(args.num_hidden_layers) { PhiDecoderLayer.new(args) }
          self.final_layernorm = MLX::NN::LayerNorm.new(args.hidden_size, eps: args.layer_norm_eps)
        end

        def call(inputs, cache: nil)
          h = embed_tokens.call(inputs)
          layer_cache = cache || [nil] * layers.length

          mask = nil
          mask = "causal" if h.shape[1] > 1

          layers.each_with_index do |layer, i|
            h = layer.call(h, mask: mask, cache: layer_cache[i])
          end

          final_layernorm.call(h)
        end
      end

      class Model < MLX::NN::Module
        def initialize(args)
          super()
          self.model = PhiModel.new(args)
          self.lm_head = MLX::NN::Linear.new(args.hidden_size, args.vocab_size, bias: true)
        end

        def call(inputs, cache: nil)
          lm_head.call(model.call(inputs, cache: cache))
        end

        def sanitize(weights)
          weights.reject { |k, _| k.include?("rotary_emb.inv_freq") }
        end

        def layers
          model.layers
        end
      end

      Models.register("phi", Model, ModelArgs)
    end
  end
end
