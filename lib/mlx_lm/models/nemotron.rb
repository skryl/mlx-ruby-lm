module MlxLm
  module Models
    module Nemotron
      class ModelArgs < BaseModelArgs
        field :model_type, default: "nemotron"
        field :hidden_size
        field :hidden_act
        field :num_hidden_layers
        field :intermediate_size
        field :num_attention_heads
        field :norm_eps
        field :vocab_size
        field :num_key_value_heads
        field :head_dim, default: nil
        field :max_position_embeddings, default: nil
        field :attention_bias, default: false
        field :mlp_bias, default: false
        field :partial_rotary_factor, default: 0.5
        field :rope_theta, default: 10_000.0
        field :rope_traditional, default: false
        field :rope_scaling, default: nil
        field :tie_word_embeddings, default: false

        def initialize(**kwargs)
          super
          @head_dim ||= @hidden_size / @num_attention_heads
          validate_rope_scaling!
        end

        private

        def rope_scaling_value(key)
          return nil unless @rope_scaling

          @rope_scaling[key] || @rope_scaling[key.to_s]
        end

        def validate_rope_scaling!
          return unless @rope_scaling

          raise ArgumentError, "rope_scaling must contain 'factor'" if rope_scaling_value(:factor).nil?

          rope_type = rope_scaling_value(:type) || rope_scaling_value(:rope_type)
          if rope_type.nil?
            raise ArgumentError, "rope_scaling must contain either 'type' or 'rope_type'"
          end
          return if rope_type == "linear"

          raise ArgumentError, "rope_scaling 'type' currently only supports 'linear'"
        end
      end

      class NemotronLayerNorm1P < MLX::NN::LayerNorm
        def call(x)
          w = state.key?("weight") ? weight + 1.0 : nil
          b = state.key?("bias") ? bias : nil
          MLX::Core.layer_norm(x, w, b, @eps)
        end
      end

      class Attention < MLX::NN::Module
        def initialize(args)
          super()

          dim = args.hidden_size
          @n_heads = args.num_attention_heads
          @n_kv_heads = args.num_key_value_heads
          @head_dim = args.head_dim
          @partial_rotary_factor = args.partial_rotary_factor
          @scale = @head_dim**(-0.5)

          bias = args.attention_bias
          self.q_proj = MLX::NN::Linear.new(dim, @n_heads * @head_dim, bias: bias)
          self.k_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: bias)
          self.v_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: bias)
          self.o_proj = MLX::NN::Linear.new(@n_heads * @head_dim, dim, bias: bias)

          rope_scale = 1.0
          if args.rope_scaling
            rope_type = args.rope_scaling[:type] || args.rope_scaling["type"] ||
              args.rope_scaling[:rope_type] || args.rope_scaling["rope_type"]
            if rope_type == "linear"
              factor = args.rope_scaling[:factor] || args.rope_scaling["factor"]
              rope_scale = 1.0 / factor.to_f
            end
          end

          self.rope = MLX::NN::RoPE.new(
            (@partial_rotary_factor * @head_dim).to_i,
            base: args.rope_theta,
            scale: rope_scale
          )
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

          dim = args.hidden_size
          hidden_dim = args.intermediate_size
          bias = args.mlp_bias
          self.down_proj = MLX::NN::Linear.new(hidden_dim, dim, bias: bias)
          self.up_proj = MLX::NN::Linear.new(dim, hidden_dim, bias: bias)
        end

        def call(x)
          down_proj.call(MLX::NN.relu2(up_proj.call(x)))
        end
      end

      class TransformerBlock < MLX::NN::Module
        def initialize(args)
          super()
          self.self_attn = Attention.new(args)
          self.mlp = MLP.new(args)
          self.input_layernorm = NemotronLayerNorm1P.new(args.hidden_size, eps: args.norm_eps)
          self.post_attention_layernorm = NemotronLayerNorm1P.new(args.hidden_size, eps: args.norm_eps)
        end

        def call(x, mask: nil, cache: nil)
          r = self_attn.call(input_layernorm.call(x), mask: mask, cache: cache)
          h = x + r
          r = mlp.call(post_attention_layernorm.call(h))
          h + r
        end
      end

      class NemotronModel < MLX::NN::Module
        def initialize(args)
          super()
          self.embed_tokens = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
          self.layers = Array.new(args.num_hidden_layers) { TransformerBlock.new(args) }
          self.norm = NemotronLayerNorm1P.new(args.hidden_size, eps: args.norm_eps)
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
          self.model_type = args.model_type
          self.model = NemotronModel.new(args)
          unless args.tie_word_embeddings
            self.lm_head = MLX::NN::Linear.new(args.hidden_size, args.vocab_size, bias: false)
          end
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

      Models.register("nemotron", Model, ModelArgs)
    end
  end
end
