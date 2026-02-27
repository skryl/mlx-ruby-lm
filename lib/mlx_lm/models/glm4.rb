module MlxLm
  module Models
    module GLM4
      class ModelArgs < BaseModelArgs
        field :model_type, default: "glm4"
        field :hidden_size, default: 4096
        field :num_hidden_layers, default: 40
        field :intermediate_size, default: 13696
        field :num_attention_heads, default: 32
        field :attention_bias, default: false
        field :head_dim, default: nil
        field :rms_norm_eps, default: 1e-5
        field :vocab_size, default: 151552
        field :num_key_value_heads, default: nil
        field :partial_rotary_factor, default: 0.5
        field :rope_theta, default: 10_000.0
        field :rope_traditional, default: true
        field :max_position_embeddings, default: 32768

        def initialize(**kwargs)
          super
          @num_key_value_heads ||= @num_attention_heads
          @head_dim ||= @hidden_size / @num_attention_heads
        end
      end

      class GLM4MLP < MLX::NN::Module
        def initialize(args)
          super()
          self.gate_up_proj = MLX::NN::Linear.new(
            args.hidden_size,
            2 * args.intermediate_size,
            bias: false
          )
          self.down_proj = MLX::NN::Linear.new(
            args.intermediate_size,
            args.hidden_size,
            bias: false
          )
        end

        def call(x)
          mx = MLX::Core
          x = gate_up_proj.call(x)
          split_dim = x.shape[-1] / 2
          gate, up_states = mx.split(x, [split_dim], -1)
          down_proj.call(Activations.swiglu(gate, up_states))
        end
      end

      class GLM4Attention < MLX::NN::Module
        def initialize(args)
          super()
          dim = args.hidden_size
          @head_dim = args.head_dim
          @n_heads = args.num_attention_heads
          @n_kv_heads = args.num_key_value_heads
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
            args.hidden_size,
            bias: false
          )

          self.rope = MLX::NN::RoPE.new(
            (args.partial_rotary_factor * @head_dim).to_i,
            base: args.rope_theta,
            traditional: args.rope_traditional
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

      class GLM4DecoderLayer < MLX::NN::Module
        def initialize(args)
          super()
          self.self_attn = GLM4Attention.new(args)
          self.mlp = GLM4MLP.new(args)
          self.input_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
          self.post_attention_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
          self.post_self_attn_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
          self.post_mlp_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
        end

        def call(x, mask: nil, cache: nil)
          x = x + post_self_attn_layernorm.call(
            self_attn.call(input_layernorm.call(x), mask: mask, cache: cache)
          )
          residual = x
          post_mlp_layernorm.call(mlp.call(post_attention_layernorm.call(x))) + residual
        end
      end

      class GLM4Model < MLX::NN::Module
        def initialize(args)
          super()
          self.embed_tokens = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
          self.layers = Array.new(args.num_hidden_layers) { GLM4DecoderLayer.new(args) }
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
          self.model_type = args.model_type
          self.model = GLM4Model.new(args)
          self.lm_head = MLX::NN::Linear.new(args.hidden_size, args.vocab_size, bias: false)
        end

        def call(inputs, cache: nil)
          out = model.call(inputs, cache: cache)
          lm_head.call(out)
        end

        def sanitize(weights)
          weights.reject { |k, _| k.include?("self_attn.rotary_emb.inv_freq") }
        end

        def layers
          model.layers
        end
      end

      Models.register("glm4", Model, ModelArgs)
    end
  end
end
