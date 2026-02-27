module MlxLm
  module Models
    module Exaone
      class ModelArgs < BaseModelArgs
        field :model_type
        field :hidden_size
        field :num_layers
        field :intermediate_size
        field :num_attention_heads
        field :vocab_size
        field :rope_theta
        field :layer_norm_epsilon
        field :num_key_value_heads
        field :head_dim, default: nil
        field :max_position_embeddings, default: nil
        field :rope_traditional, default: false
        field :rope_scaling, default: nil
        field :tie_word_embeddings, default: true
        field :attention_bias, default: false
        field :mlp_bias, default: false

        def initialize(**kwargs)
          super
          @head_dim ||= @hidden_size / @num_attention_heads
        end
      end

      class AttentionModule < MLX::NN::Module
        def initialize(args)
          super()
          dim = args.hidden_size
          @n_heads = args.num_attention_heads
          @n_kv_heads = args.num_key_value_heads
          @head_dim = args.head_dim
          @scale = @head_dim**(-0.5)

          bias = args.attention_bias
          self.q_proj = MLX::NN::Linear.new(dim, @n_heads * @head_dim, bias: bias)
          self.k_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: bias)
          self.v_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: bias)
          self.out_proj = MLX::NN::Linear.new(@n_heads * @head_dim, dim, bias: bias)

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
          b, l, d = x.shape

          q = q_proj.call(x).reshape([b, l, @n_heads, @head_dim]).transpose([0, 2, 1, 3])
          k = k_proj.call(x).reshape([b, l, @n_kv_heads, @head_dim]).transpose([0, 2, 1, 3])
          v = v_proj.call(x).reshape([b, l, @n_kv_heads, @head_dim]).transpose([0, 2, 1, 3])

          if cache
            q = rope.call(q, offset: cache.offset)
            k = rope.call(k, offset: cache.offset)
            k, v = cache.update_and_fetch(k, v)
          else
            q = rope.call(q)
            k = rope.call(k)
          end

          out = mx.scaled_dot_product_attention(q, k, v, @scale, mask)
          out = out.transpose([0, 2, 1, 3]).reshape([b, l, d])
          out_proj.call(out)
        end
      end

      class Attention < MLX::NN::Module
        def initialize(args)
          super()
          self.attention = AttentionModule.new(args)
        end
      end

      class MLP < MLX::NN::Module
        def initialize(args)
          super()
          dim = args.hidden_size
          hidden_dim = args.intermediate_size
          bias = args.mlp_bias
          self.c_fc_0 = MLX::NN::Linear.new(dim, hidden_dim, bias: bias)
          self.c_fc_1 = MLX::NN::Linear.new(dim, hidden_dim, bias: bias)
          self.c_proj = MLX::NN::Linear.new(hidden_dim, dim, bias: bias)
        end

        def call(x)
          c_proj.call(MlxLm::Models::Activations.swiglu(c_fc_0.call(x), c_fc_1.call(x)))
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
          h = x + attn.attention.call(ln_1.call(x), mask: mask, cache: cache)
          h + mlp.call(ln_2.call(h))
        end
      end

      class ExaoneModel < MLX::NN::Module
        def initialize(args)
          super()
          self.wte = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
          self.h = Array.new(args.num_layers) { TransformerBlock.new(args) }
          self.ln_f = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.layer_norm_epsilon)
        end

        def call(inputs, cache: nil)
          hidden = wte.call(inputs)
          layer_cache = cache || [nil] * h.length

          mask = nil
          mask = "causal" if hidden.shape[1] > 1

          h.each_with_index do |layer, i|
            hidden = layer.call(hidden, mask: mask, cache: layer_cache[i])
          end

          ln_f.call(hidden)
        end
      end

      class Model < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.transformer = ExaoneModel.new(args)
          unless args.tie_word_embeddings
            self.lm_head = MLX::NN::Linear.new(args.hidden_size, args.vocab_size, bias: false)
          end
        end

        def call(inputs, cache: nil)
          out = transformer.call(inputs, cache: cache)
          if @args.tie_word_embeddings
            transformer.wte.as_linear(out)
          else
            lm_head.call(out)
          end
        end

        def sanitize(weights)
          result = weights.reject { |k, _| k.include?("rotary_emb.inv_freq") }
          result.delete("lm_head.weight") if @args.tie_word_embeddings
          result
        end

        def layers
          transformer.h
        end
      end

      Models.register("exaone", Model, ModelArgs)
    end
  end
end
