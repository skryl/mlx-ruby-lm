module MlxLm
  module Models
    module MiniCPM
      class ModelArgs < BaseModelArgs
        field :model_type, default: "minicpm"
        field :hidden_size
        field :dim_model_base
        field :num_hidden_layers
        field :intermediate_size
        field :num_attention_heads
        field :rms_norm_eps
        field :vocab_size
        field :num_key_value_heads
        field :scale_depth
        field :scale_emb
        field :max_position_embeddings, default: nil
        field :rope_theta, default: 1_000_000.0
        field :rope_traditional, default: false
        field :rope_scaling, default: nil
        field :tie_word_embeddings, default: false

        def initialize(**kwargs)
          super
          @num_key_value_heads ||= @num_attention_heads
        end
      end

      class MLP < MLX::NN::Module
        def initialize(args)
          super()
          self.gate_proj = MLX::NN::Linear.new(args.hidden_size, args.intermediate_size, bias: false)
          self.up_proj = MLX::NN::Linear.new(args.hidden_size, args.intermediate_size, bias: false)
          self.down_proj = MLX::NN::Linear.new(args.intermediate_size, args.hidden_size, bias: false)
        end

        def call(x)
          down_proj.call(Activations.swiglu(gate_proj.call(x), up_proj.call(x)))
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

          self.q_proj = MLX::NN::Linear.new(dim, @n_heads * @head_dim, bias: false)
          self.k_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: false)
          self.v_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: false)
          self.o_proj = MLX::NN::Linear.new(@n_heads * @head_dim, dim, bias: false)

          self.rope = MlxLm::Models.initialize_rope(
            @head_dim,
            args.rope_theta,
            args.rope_traditional,
            args.rope_scaling,
            max_position_embeddings: args.max_position_embeddings
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
      end

      class DecoderLayer < MLX::NN::Module
        def initialize(args)
          super()
          self.self_attn = Attention.new(args)
          self.mlp = MLP.new(args)
          self.input_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
          self.post_attention_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
          @residual_scale = args.scale_depth / Math.sqrt(args.num_hidden_layers)
        end

        def call(x, mask: nil, cache: nil)
          r = self_attn.call(input_layernorm.call(x), mask: mask, cache: cache)
          h = x + r * @residual_scale
          r = mlp.call(post_attention_layernorm.call(h))
          h + r * @residual_scale
        end
      end

      class MiniCPMModel < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.embed_tokens = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
          self.layers = Array.new(args.num_hidden_layers) { DecoderLayer.new(args) }
          self.norm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
        end

        def call(inputs, cache: nil)
          h = embed_tokens.call(inputs) * @args.scale_emb
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
          self.model_type = args.model_type
          self.model = MiniCPMModel.new(args)
          self.lm_head = MLX::NN::Linear.new(args.hidden_size, args.vocab_size, bias: false) unless args.tie_word_embeddings
        end

        def call(inputs, cache: nil)
          mx = MLX::Core
          out = model.call(inputs, cache: cache)

          if @args.tie_word_embeddings
            mx.matmul(out, model.embed_tokens.weight.T)
          else
            lm_head.call(out / (@args.hidden_size.to_f / @args.dim_model_base))
          end
        end

        def sanitize(weights)
          unless weights.key?("lm_head.weight")
            weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
          end
          weights
        end

        def layers
          model.layers
        end
      end

      Models.register("minicpm", Model, ModelArgs)
    end
  end
end
