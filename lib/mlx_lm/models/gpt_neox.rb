module MlxLm
  module Models
    module GPTNeoX
      class ModelArgs < BaseModelArgs
        field :model_type, default: "gpt_neox"
        field :hidden_size, default: 2560
        field :num_hidden_layers, default: 32
        field :num_attention_heads, default: 32
        field :num_key_value_heads, default: nil
        field :vocab_size, default: 50432
        field :layer_norm_eps, default: 1e-5
        field :rotary_emb_base, default: 10000
        field :rotary_pct, default: 0.25
        field :use_parallel_residual, default: true
        field :max_position_embeddings, default: 2048
        field :intermediate_size, default: nil

        def initialize(**kwargs)
          super
          @num_key_value_heads ||= @num_attention_heads
          @intermediate_size ||= 4 * @hidden_size
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

          # Partial rotary: only apply RoPE to rotary_pct fraction of head_dim
          rope_dim = (args.rotary_pct * @head_dim).to_i

          # Combined QKV projection
          total_qkv = (@n_heads + 2 * @n_kv_heads) * @head_dim
          self.query_key_value = MLX::NN::Linear.new(dim, total_qkv, bias: true)
          self.dense = MLX::NN::Linear.new(dim, dim, bias: true)

          self.rope = MLX::NN::RoPE.new(
            rope_dim,
            traditional: false,
            base: args.rotary_emb_base
          )
        end

        def call(x, mask: nil, cache: nil)
          mx = MLX::Core
          b, l, _d = x.shape

          qkv = query_key_value.call(x)

          # Split into Q, K, V
          q_size = @n_heads * @head_dim
          kv_size = @n_kv_heads * @head_dim
          queries, keys, values = mx.split(qkv, [q_size, q_size + kv_size], 2)

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
          dense.call(output)
        end
      end

      class MLP < MLX::NN::Module
        def initialize(dim, hidden_dim)
          super()
          self.dense_h_to_4h = MLX::NN::Linear.new(dim, hidden_dim, bias: true)
          self.dense_4h_to_h = MLX::NN::Linear.new(hidden_dim, dim, bias: true)
        end

        def call(x)
          dense_4h_to_h.call(MLX::NN.gelu_approx(dense_h_to_4h.call(x)))
        end
      end

      class TransformerBlock < MLX::NN::Module
        def initialize(args)
          super()
          self.self_attn = Attention.new(args)
          self.mlp = MLP.new(args.hidden_size, args.intermediate_size)
          self.input_layernorm = MLX::NN::LayerNorm.new(args.hidden_size, eps: args.layer_norm_eps)
          @use_parallel_residual = args.use_parallel_residual
          unless @use_parallel_residual
            self.post_attention_layernorm = MLX::NN::LayerNorm.new(args.hidden_size, eps: args.layer_norm_eps)
          end
        end

        def call(x, mask: nil, cache: nil)
          h = input_layernorm.call(x)
          r = self_attn.call(h, mask: mask, cache: cache)

          if @use_parallel_residual
            x + r + mlp.call(h)
          else
            h = x + r
            r = mlp.call(post_attention_layernorm.call(h))
            h + r
          end
        end
      end

      class GPTNeoXModel < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.embed_in = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
          self.h = Array.new(args.num_hidden_layers) { TransformerBlock.new(args) }
          self.final_layer_norm = MLX::NN::LayerNorm.new(args.hidden_size, eps: args.layer_norm_eps)
        end

        def call(inputs, cache: nil)
          hidden = embed_in.call(inputs)
          layer_cache = cache || [nil] * h.length

          mask = nil
          mask = "causal" if hidden.shape[1] > 1

          h.each_with_index do |layer, i|
            hidden = layer.call(hidden, mask: mask, cache: layer_cache[i])
          end

          final_layer_norm.call(hidden)
        end
      end

      class Model < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.model = GPTNeoXModel.new(args)
          self.embed_out = MLX::NN::Linear.new(args.hidden_size, args.vocab_size, bias: false)
        end

        def call(inputs, cache: nil)
          out = model.call(inputs, cache: cache)
          embed_out.call(out)
        end

        def sanitize(weights)
          result = {}
          weights.each do |k, v|
            next if k.include?(".attention.bias") || k.include?(".attention.masked_bias")
            next if k.include?(".attention.rotary_emb.inv_freq")

            # Remap weight keys
            key = k.dup
            key.gsub!(".gpt_neox.layers.", ".h.")
            key.gsub!(".gpt_neox.", ".")
            key = "model.#{key}" unless key.start_with?("model.")
            result[key] = v
          end
          result
        end

        def layers
          model.h
        end
      end

      Models.register("gpt_neox", Model, ModelArgs)
    end
  end
end
