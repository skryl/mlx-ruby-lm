module MlxLm
  module Models
    module Lille130m
      class ModelArgs < BaseModelArgs
        field :model_type, default: "lille-130m"
        field :block_size
        field :layer_norm_eps
        field :n_embd
        field :n_head
        field :n_kv_heads
        field :n_layer
        field :rope_theta
        field :vocab_size
        field :tie_word_embeddings, default: true
      end

      class Lille130mAttention < MLX::NN::Module
        def initialize(args)
          super()
          @n_head = args.n_head
          @n_kv_heads = args.n_kv_heads
          @head_dim = args.n_embd / @n_head
          @scale = @head_dim**(-0.5)

          self.qkv_proj = MLX::NN::Linear.new(
            args.n_embd,
            (@n_head + (2 * @n_kv_heads)) * @head_dim,
            bias: false
          )
          self.out_proj = MLX::NN::Linear.new(@n_head * @head_dim, args.n_embd, bias: false)
          self.norm = MLX::NN::RMSNorm.new(args.n_embd, eps: args.layer_norm_eps)
          self.rope = MLX::NN::RoPE.new(@head_dim, traditional: true, base: args.rope_theta)
        end

        def call(x, mask: nil, cache: nil)
          mx = MLX::Core
          b, l, _d = x.shape

          qkv = qkv_proj.call(norm.call(x))
          q_size = @n_head * @head_dim
          kv_size = @n_kv_heads * @head_dim
          queries, keys, values = mx.split(qkv, [q_size, q_size + kv_size], 2)

          queries = queries.reshape([b, l, @n_head, @head_dim]).transpose([0, 2, 1, 3])
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
          output = output.transpose([0, 2, 1, 3]).reshape([b, l, @n_head * @head_dim])
          out_proj.call(output)
        end
      end

      class Lille130mMLP < MLX::NN::Module
        def initialize(args)
          super()
          hidden_dim = 256 * ((8 * args.n_embd / 3) / 256.0).round
          hidden_dim = 256 if hidden_dim.zero?

          self.norm = MLX::NN::RMSNorm.new(args.n_embd, eps: args.layer_norm_eps)
          self.gate_proj = MLX::NN::Linear.new(args.n_embd, hidden_dim, bias: false)
          self.up_proj = MLX::NN::Linear.new(args.n_embd, hidden_dim, bias: false)
          self.down_proj = MLX::NN::Linear.new(hidden_dim, args.n_embd, bias: false)
        end

        def call(x)
          h = norm.call(x)
          down_proj.call(Activations.swiglu(gate_proj.call(h), up_proj.call(h)))
        end
      end

      class Lille130Block < MLX::NN::Module
        def initialize(args)
          super()
          self.attention = Lille130mAttention.new(args)
          self.feed_forward = Lille130mMLP.new(args)
        end

        def call(x, mask: nil, cache: nil)
          h = x + attention.call(x, mask: mask, cache: cache)
          h + feed_forward.call(h)
        end
      end

      class Lille130 < MLX::NN::Module
        def initialize(args)
          super()
          self.tok_embeddings = MLX::NN::Embedding.new(args.vocab_size, args.n_embd)
          self.layers = Array.new(args.n_layer) { Lille130Block.new(args) }
          self.norm = MLX::NN::RMSNorm.new(args.n_embd, eps: args.layer_norm_eps)
        end

        def call(inputs, cache: nil)
          h = tok_embeddings.call(inputs)
          layer_cache = cache || [nil] * layers.length
          mask = _create_attention_mask(h, layer_cache[0])

          layers.each_with_index do |layer, i|
            h = layer.call(h, mask: mask, cache: layer_cache[i])
          end

          tok_embeddings.as_linear(norm.call(h))
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
          self.transformer = Lille130.new(args)
        end

        def call(inputs, cache: nil)
          transformer.call(inputs, cache: cache)
        end

        def layers
          transformer.layers
        end

        def sanitize(weights)
          weights.reject { |k, _| k.include?("rotary_emb") }
        end
      end

      Models.register("lille-130m", Model, ModelArgs)
    end
  end
end
