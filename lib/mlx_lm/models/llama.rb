module MlxLm
  module Models
    module Llama
      class ModelArgs < BaseModelArgs
        field :model_type, default: "llama"
        field :hidden_size, default: 4096
        field :num_hidden_layers, default: 32
        field :num_attention_heads, default: 32
        field :num_key_value_heads, default: nil
        field :intermediate_size, default: 11008
        field :vocab_size, default: 32000
        field :rms_norm_eps, default: 1e-6
        field :rope_theta, default: 10000.0
        field :rope_traditional, default: false
        field :rope_scaling, default: nil
        field :tie_word_embeddings, default: true
        field :attention_bias, default: false
        field :mlp_bias, default: false
        field :head_dim, default: nil
        field :max_position_embeddings, default: 2048

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
          @head_dim = args.head_dim
          @scale = @head_dim**(-0.5)

          bias = args.attention_bias
          self.q_proj = MLX::NN::Linear.new(dim, @n_heads * @head_dim, bias: bias)
          self.k_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: bias)
          self.v_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: bias)
          self.o_proj = MLX::NN::Linear.new(@n_heads * @head_dim, dim, bias: bias)

          self.rope = MLX::NN::RoPE.new(
            @head_dim,
            traditional: args.rope_traditional,
            base: args.rope_theta
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

          output = mx.scaled_dot_product_attention(
            queries, keys, values, @scale, mask
          )

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
          mx = MLX::Core
          down_proj.call(MLX::NN.silu(gate_proj.call(x)) * up_proj.call(x))
        end
      end

      class TransformerBlock < MLX::NN::Module
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

      class LlamaModel < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.embed_tokens = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
          self.layers = Array.new(args.num_hidden_layers) { TransformerBlock.new(args) }
          self.norm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
        end

        def call(inputs, cache: nil)
          mx = MLX::Core
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
          mx = MLX::Core
          n = h.shape[1]
          return nil if n == 1
          "causal"
        end
      end

      class Model < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.model = LlamaModel.new(args)
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

        def sanitize(weights)
          result = weights.reject { |k, _| k.include?("self_attn.rotary_emb.inv_freq") }
          result.delete("lm_head.weight") if @args.tie_word_embeddings
          result
        end

        def layers
          model.layers
        end
      end

      # Register in model registry
      Models.register("llama", Model, ModelArgs)
    end
  end
end
