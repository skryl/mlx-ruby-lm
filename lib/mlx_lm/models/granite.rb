module MlxLm
  module Models
    module Granite
      class ModelArgs < BaseModelArgs
        field :model_type, default: "granite"
        field :hidden_size
        field :num_hidden_layers
        field :intermediate_size
        field :num_attention_heads
        field :rms_norm_eps
        field :vocab_size
        field :logits_scaling
        field :attention_multiplier
        field :embedding_multiplier
        field :residual_multiplier
        field :max_position_embeddings
        field :num_key_value_heads
        field :attention_bias
        field :mlp_bias
        field :rope_theta
        field :rope_scaling, default: nil
        field :tie_word_embeddings, default: true

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
          @scale = args.attention_multiplier

          bias = args.attention_bias
          self.q_proj = MLX::NN::Linear.new(dim, @n_heads * @head_dim, bias: bias)
          self.k_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: bias)
          self.v_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: bias)
          self.o_proj = MLX::NN::Linear.new(@n_heads * @head_dim, dim, bias: bias)

          self.rope = MlxLm::Models.initialize_rope(
            @head_dim,
            args.rope_theta,
            false,
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

      class MLP < MLX::NN::Module
        def initialize(args)
          super()

          dim = args.hidden_size
          hidden_dim = args.intermediate_size
          bias = args.mlp_bias

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
          self.mlp = MLP.new(args)
          self.input_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
          self.post_attention_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
          @residual_multiplier = args.residual_multiplier
        end

        def call(x, mask: nil, cache: nil)
          r = self_attn.call(input_layernorm.call(x), mask: mask, cache: cache)
          h = x + r * @residual_multiplier
          r = mlp.call(post_attention_layernorm.call(h))
          h + r * @residual_multiplier
        end
      end

      class GraniteModel < MLX::NN::Module
        def initialize(args)
          super()
          self.embed_tokens = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
          self.layers = Array.new(args.num_hidden_layers) { TransformerBlock.new(args) }
          self.norm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
          @embedding_multiplier = args.embedding_multiplier
        end

        def call(inputs, cache: nil)
          h = embed_tokens.call(inputs) * @embedding_multiplier
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
          self.model = GraniteModel.new(args)
          self.lm_head = MLX::NN::Linear.new(args.hidden_size, args.vocab_size, bias: false) unless args.tie_word_embeddings
          @logits_scaling = args.logits_scaling
        end

        def call(inputs, cache: nil)
          out = model.call(inputs, cache: cache)
          out = if @args.tie_word_embeddings
            model.embed_tokens.as_linear(out)
          else
            lm_head.call(out)
          end
          out / @logits_scaling
        end

        def layers
          model.layers
        end
      end

      Models.register("granite", Model, ModelArgs)
    end
  end
end
