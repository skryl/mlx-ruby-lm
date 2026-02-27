module MlxLm
  module Models
    module GPTBigCode
      class ModelArgs < BaseModelArgs
        field :model_type, default: "gpt_bigcode"
        field :n_embd
        field :n_layer
        field :n_inner
        field :n_head
        field :n_positions
        field :layer_norm_epsilon
        field :vocab_size
        field :num_key_value_heads, default: nil
        field :multi_query, default: true
        field :attention_bias, default: true
        field :mlp_bias, default: true
        field :tie_word_embeddings, default: true

        def initialize(**kwargs)
          super
          @num_key_value_heads ||= @multi_query ? 1 : @n_head
        end
      end

      class Attention < MLX::NN::Module
        def initialize(args)
          super()

          @dim = args.n_embd
          @n_heads = args.n_head
          @n_kv_heads = args.multi_query ? 1 : args.n_head
          @head_dim = @dim / @n_heads
          @kv_dim = @n_kv_heads * @head_dim
          @scale = @head_dim**(-0.5)

          bias = args.attention_bias
          self.c_attn = MLX::NN::Linear.new(@dim, @dim + 2 * @kv_dim, bias: bias)
          self.c_proj = MLX::NN::Linear.new(@dim, @dim, bias: bias)
        end

        def call(x, mask: nil, cache: nil)
          mx = MLX::Core
          b, l, _d = x.shape

          qkv = c_attn.call(x)
          queries, keys, values = mx.split(qkv, [@dim, @dim + @kv_dim], -1)

          queries = queries.reshape([b, l, @n_heads, @head_dim]).transpose([0, 2, 1, 3])
          keys = keys.reshape([b, l, @n_kv_heads, @head_dim]).transpose([0, 2, 1, 3])
          values = values.reshape([b, l, @n_kv_heads, @head_dim]).transpose([0, 2, 1, 3])

          if cache
            keys, values = cache.update_and_fetch(keys, values)
          end

          output = mx.scaled_dot_product_attention(queries, keys, values, @scale, mask)
          output = output.transpose([0, 2, 1, 3]).reshape([b, l, @dim])
          c_proj.call(output)
        end
      end

      class MLP < MLX::NN::Module
        def initialize(args)
          super()

          dim = args.n_embd
          hidden_dim = args.n_inner
          bias = args.mlp_bias
          self.c_fc = MLX::NN::Linear.new(dim, hidden_dim, bias: bias)
          self.c_proj = MLX::NN::Linear.new(hidden_dim, dim, bias: bias)
        end

        def call(x)
          c_proj.call(MLX::NN.gelu(c_fc.call(x)))
        end
      end

      class TransformerBlock < MLX::NN::Module
        def initialize(args)
          super()
          self.attn = Attention.new(args)
          self.mlp = MLP.new(args)
          self.ln_1 = MLX::NN::LayerNorm.new(args.n_embd, eps: args.layer_norm_epsilon)
          self.ln_2 = MLX::NN::LayerNorm.new(args.n_embd, eps: args.layer_norm_epsilon)
        end

        def call(x, mask: nil, cache: nil)
          r = attn.call(ln_1.call(x), mask: mask, cache: cache)
          h = x + r
          r = mlp.call(ln_2.call(h))
          h + r
        end
      end

      class GPTBigCodeModel < MLX::NN::Module
        def initialize(args)
          super()
          self.wte = MLX::NN::Embedding.new(args.vocab_size, args.n_embd)
          self.wpe = MLX::NN::Embedding.new(args.n_positions, args.n_embd)
          self.h = Array.new(args.n_layer) { TransformerBlock.new(args) }
          self.ln_f = MLX::NN::LayerNorm.new(args.n_embd, eps: args.layer_norm_epsilon)
        end

        def call(inputs, cache: nil)
          mx = MLX::Core
          _b, l = inputs.shape

          hidden_states = wte.call(inputs)
          layer_cache = cache || [nil] * h.length
          offset = layer_cache[0] ? layer_cache[0].offset : 0
          position_ids = mx.arange(offset, offset + l, 1, mx.int32)

          mask = nil
          mask = "causal" if hidden_states.shape[1] > 1

          hidden_states = hidden_states + wpe.call(position_ids)

          h.each_with_index do |layer, i|
            hidden_states = layer.call(hidden_states, mask: mask, cache: layer_cache[i])
          end

          ln_f.call(hidden_states)
        end
      end

      class Model < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.model_type = args.model_type
          self.transformer = GPTBigCodeModel.new(args)
          unless args.tie_word_embeddings
            self.lm_head = MLX::NN::Linear.new(args.n_embd, args.vocab_size, bias: false)
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

        def layers
          transformer.h
        end
      end

      Models.register("gpt_bigcode", Model, ModelArgs)
    end
  end
end
