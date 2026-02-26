module MlxLm
  module Models
    module Telechat3
      class ModelArgs < BaseModelArgs
        field :model_type, default: "telechat3"
        field :hidden_size, default: 4096
        field :intermediate_size, default: 14336
        field :max_position_embeddings, default: 32768
        field :num_attention_heads, default: 32
        field :num_hidden_layers, default: 32
        field :num_key_value_heads, default: nil
        field :rms_norm_eps, default: 1e-6
        field :vocab_size, default: 151936
        field :rope_theta, default: 10_000.0
        field :mlp_bias, default: false
        field :attention_bias, default: false
        field :head_dim, default: nil
        field :rope_scaling, default: nil
        field :tie_word_embeddings, default: false

        def initialize(**kwargs)
          super
          @num_key_value_heads ||= @num_attention_heads
          @head_dim ||= @hidden_size / @num_attention_heads
        end
      end

      class Telechat3Attention < MLX::NN::Module
        def initialize(args)
          super()
          dim = args.hidden_size
          @num_attention_heads = args.num_attention_heads
          @num_key_value_heads = args.num_key_value_heads
          @head_dim = args.head_dim
          @scale = @head_dim**(-0.5)

          self.q_proj = MLX::NN::Linear.new(
            dim,
            args.num_attention_heads * @head_dim,
            bias: args.attention_bias
          )
          self.k_proj = MLX::NN::Linear.new(
            dim,
            args.num_key_value_heads * @head_dim,
            bias: args.attention_bias
          )
          self.v_proj = MLX::NN::Linear.new(
            dim,
            args.num_key_value_heads * @head_dim,
            bias: args.attention_bias
          )
          self.o_proj = MLX::NN::Linear.new(
            args.num_attention_heads * @head_dim,
            dim,
            bias: args.attention_bias
          )

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

          queries = queries.reshape([b, l, @num_attention_heads, @head_dim]).transpose([0, 2, 1, 3])
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

          output = mx.scaled_dot_product_attention(queries, keys, values, @scale, mask)
          output = output.transpose([0, 2, 1, 3]).reshape([b, l, @num_attention_heads * @head_dim])
          o_proj.call(output)
        end
      end

      class Telechat3MLP < MLX::NN::Module
        def initialize(args)
          super()
          self.gate_proj = MLX::NN::Linear.new(
            args.hidden_size,
            args.intermediate_size,
            bias: args.mlp_bias
          )
          self.down_proj = MLX::NN::Linear.new(
            args.intermediate_size,
            args.hidden_size,
            bias: args.mlp_bias
          )
          self.up_proj = MLX::NN::Linear.new(
            args.hidden_size,
            args.intermediate_size,
            bias: args.mlp_bias
          )
        end

        def call(x)
          down_proj.call(Activations.swiglu(gate_proj.call(x), up_proj.call(x)))
        end
      end

      class Telechat3DecoderLayer < MLX::NN::Module
        def initialize(args)
          super()
          self.self_attn = Telechat3Attention.new(args)
          self.mlp = Telechat3MLP.new(args)
          self.input_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
          self.post_attention_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
        end

        def call(x, mask: nil, cache: nil)
          r = self_attn.call(input_layernorm.call(x), mask: mask, cache: cache)
          h = x + r
          h + mlp.call(post_attention_layernorm.call(h))
        end
      end

      class Telechat3Model < MLX::NN::Module
        def initialize(args)
          super()
          self.embed_tokens = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
          self.layers = Array.new(args.num_hidden_layers) { Telechat3DecoderLayer.new(args) }
          self.norm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
        end

        def call(inputs, cache: nil, input_embeddings: nil)
          h = input_embeddings || embed_tokens.call(inputs)
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
          self.model = Telechat3Model.new(args)
          unless args.tie_word_embeddings
            self.lm_head = MLX::NN::Linear.new(args.hidden_size, args.vocab_size, bias: false)
          end
        end

        def call(inputs, cache: nil, input_embeddings: nil)
          out = model.call(inputs, cache: cache, input_embeddings: input_embeddings)
          if @args.tie_word_embeddings
            model.embed_tokens.as_linear(out)
          else
            lm_head.call(out)
          end
        end

        def sanitize(weights)
          result = weights.reject { |k, _| k.include?("self_attn.rotary_emb.inv_freq") }
          result.delete("lm_head.weight") if @args.tie_word_embeddings
          result
        end

        def layers
          model.layers
        end
      end

      Models.register("telechat3", Model, ModelArgs)
    end
  end
end
