module MlxLm
  module Models
    module Apertus
      class ModelArgs < BaseModelArgs
        field :model_type
        field :hidden_size
        field :num_hidden_layers
        field :intermediate_size
        field :mlp_bias
        field :num_attention_heads
        field :attention_bias
        field :rms_norm_eps
        field :vocab_size
        field :num_key_value_heads
        field :max_position_embeddings
        field :rope_theta
        field :post_norm
        field :qk_norm
        field :tie_word_embeddings
        field :rope_traditional, default: false
        field :rope_scaling, default: nil

        def initialize(**kwargs)
          super
          @num_key_value_heads ||= @num_attention_heads
        end
      end

      class ApertusMLP < MLX::NN::Module
        def initialize(args)
          super()
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
          self.act_fn = Activations::XieLU.new
        end

        def call(x)
          down_proj.call(act_fn.call(up_proj.call(x)))
        end
      end

      class ApertusAttention < MLX::NN::Module
        def initialize(args)
          super()
          dim = args.hidden_size
          @num_attention_heads = args.num_attention_heads
          @num_key_value_heads = args.num_key_value_heads
          @head_dim = dim / @num_attention_heads
          @scale = @head_dim**(-0.5)

          self.q_proj = MLX::NN::Linear.new(dim, @num_attention_heads * @head_dim, bias: false)
          self.k_proj = MLX::NN::Linear.new(dim, @num_key_value_heads * @head_dim, bias: false)
          self.v_proj = MLX::NN::Linear.new(dim, @num_key_value_heads * @head_dim, bias: false)
          self.o_proj = MLX::NN::Linear.new(@num_attention_heads * @head_dim, dim, bias: false)

          self.q_norm = MLX::NN::RMSNorm.new(@head_dim, eps: args.rms_norm_eps)
          self.k_norm = MLX::NN::RMSNorm.new(@head_dim, eps: args.rms_norm_eps)
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

          queries = q_norm.call(queries.reshape([b, l, @num_attention_heads, @head_dim])).transpose([0, 2, 1, 3])
          keys = k_norm.call(keys.reshape([b, l, @num_key_value_heads, @head_dim])).transpose([0, 2, 1, 3])
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

      class ApertusDecoderLayer < MLX::NN::Module
        def initialize(args)
          super()
          self.self_attn = ApertusAttention.new(args)
          self.mlp = ApertusMLP.new(args)
          self.attention_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
          self.feedforward_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
        end

        def call(x, mask: nil, cache: nil)
          h = x + self_attn.call(attention_layernorm.call(x), mask: mask, cache: cache)
          h + mlp.call(feedforward_layernorm.call(h))
        end
      end

      class ApertusModel < MLX::NN::Module
        def initialize(args)
          super()
          self.embed_tokens = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
          self.layers = Array.new(args.num_hidden_layers) { ApertusDecoderLayer.new(args) }
          self.norm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
        end

        def call(inputs, cache: nil)
          h = embed_tokens.call(inputs)
          layer_cache = cache || [nil] * layers.length
          mask = _create_attention_mask(h, layer_cache[0])

          layers.each_with_index do |layer, i|
            h = layer.call(h, mask: mask, cache: layer_cache[i])
          end

          norm.call(h)
        end

        private

        def _create_attention_mask(h, cache)
          return cache.make_mask(h.shape[1]) if cache && cache.respond_to?(:make_mask)
          return nil if h.shape[1] == 1

          "causal"
        end
      end

      class Model < MLX::NN::Module
        def initialize(args)
          super()
          self.args = args
          self.model_type = args.model_type
          self.model = ApertusModel.new(args)
          self.lm_head = MLX::NN::Linear.new(args.hidden_size, args.vocab_size, bias: false)
        end

        def call(inputs, cache: nil)
          out = model.call(inputs, cache: cache)
          lm_head.call(out)
        end

        def sanitize(weights)
          mx = MLX::Core
          weights.each do |k, v|
            if k.end_with?("alpha_p") || k.end_with?("alpha_n")
              weights[k] = mx.squeeze(v)
            end
          end
          weights
        end

        def layers
          model.layers
        end
      end

      Models.register("apertus", Model, ModelArgs)
    end
  end
end
