module MlxLm
  module Models
    module Nanochat
      module_function

      def rms_norm(x, eps: 1e-5)
        mx = MLX::Core
        variance = mx.mean(mx.square(x), -1, true)
        mx.multiply(x, mx.rsqrt(mx.add(variance, eps)))
      end

      def softcap(logits, cap: 15.0)
        mx = MLX::Core
        mx.multiply(cap, mx.tanh(mx.divide(logits, cap)))
      end

      class ModelArgs < BaseModelArgs
        field :model_type, default: "nanochat"
        field :hidden_size, default: 1280
        field :num_hidden_layers, default: 20
        field :num_attention_heads, default: 10
        field :num_key_value_heads, default: 10
        field :vocab_size, default: 65_536
        field :max_position_embeddings, default: 2048
        field :intermediate_size, default: 5120
        field :rope_theta, default: 10_000.0
      end

      class Attention < MLX::NN::Module
        def initialize(args)
          super()

          @hidden_size = args.hidden_size
          @num_heads = args.num_attention_heads
          @num_kv_heads = args.num_key_value_heads
          @head_dim = @hidden_size / @num_heads
          @scale = @head_dim**(-0.5)
          @rope_theta = args.rope_theta

          self.c_q = MLX::NN::Linear.new(@hidden_size, @num_heads * @head_dim, bias: false)
          self.c_k = MLX::NN::Linear.new(@hidden_size, @num_kv_heads * @head_dim, bias: false)
          self.c_v = MLX::NN::Linear.new(@hidden_size, @num_kv_heads * @head_dim, bias: false)
          self.c_proj = MLX::NN::Linear.new(@hidden_size, @hidden_size, bias: false)

          mx = MLX::Core
          exponent = mx.multiply(
            mx.arange(0, @head_dim, 2, mx.float32),
            Math.log(@rope_theta) / @head_dim.to_f
          )
          self._rope_freqs = mx.multiply(-1.0, mx.exp(exponent))
        end

        def call(x, mask: nil, cache: nil)
          mx = MLX::Core
          b, l, _d = x.shape

          queries = c_q.call(x)
          keys = c_k.call(x)
          values = c_v.call(x)

          queries = queries.reshape([b, l, @num_heads, @head_dim]).transpose([0, 2, 1, 3])
          keys = keys.reshape([b, l, @num_kv_heads, @head_dim]).transpose([0, 2, 1, 3])
          values = values.reshape([b, l, @num_kv_heads, @head_dim]).transpose([0, 2, 1, 3])

          offset = cache ? cache.offset : 0
          queries = _apply_rotary_emb(queries, offset: offset)
          keys = _apply_rotary_emb(keys, offset: offset)

          queries = Nanochat.rms_norm(queries)
          keys = Nanochat.rms_norm(keys)

          if cache
            keys, values = cache.update_and_fetch(keys, values)
          end

          output = mx.scaled_dot_product_attention(queries, keys, values, @scale, mask)
          output = output.transpose([0, 2, 1, 3]).reshape([b, l, @hidden_size])
          c_proj.call(output)
        end

        private

        def _apply_rotary_emb(x, offset:)
          MLX::Core.rope(x, @head_dim, false, nil, 1.0, offset, _rope_freqs)
        end
      end

      class MLP < MLX::NN::Module
        def initialize(args)
          super()
          self.c_fc = MLX::NN::Linear.new(args.hidden_size, args.intermediate_size, bias: false)
          self.c_proj = MLX::NN::Linear.new(args.intermediate_size, args.hidden_size, bias: false)
        end

        def call(x)
          c_proj.call(MLX::NN.relu2(c_fc.call(x)))
        end
      end

      class TransformerBlock < MLX::NN::Module
        def initialize(args)
          super()
          self.attn = Attention.new(args)
          self.mlp = MLP.new(args)
        end

        def call(x, mask: nil, cache: nil)
          h = x + attn.call(Nanochat.rms_norm(x), mask: mask, cache: cache)
          h + mlp.call(Nanochat.rms_norm(h))
        end
      end

      class NanoChatModel < MLX::NN::Module
        def initialize(args)
          super()
          self.wte = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
          self.h = Array.new(args.num_hidden_layers) { TransformerBlock.new(args) }
        end

        def call(inputs, cache: nil)
          hidden = wte.call(inputs)
          hidden = Nanochat.rms_norm(hidden)

          layer_cache = cache || [nil] * h.length
          mask = _create_attention_mask(hidden, layer_cache[0])

          h.each_with_index do |layer, i|
            hidden = layer.call(hidden, mask: mask, cache: layer_cache[i])
          end

          Nanochat.rms_norm(hidden)
        end

        private

        def _create_attention_mask(hidden, cache)
          return cache.make_mask(hidden.shape[1]) if cache && cache.respond_to?(:make_mask)
          return nil if hidden.shape[1] == 1

          "causal"
        end
      end

      class Model < MLX::NN::Module
        def initialize(args)
          super()
          self.args = args
          self.model_type = args.model_type
          self.transformer = NanoChatModel.new(args)
          self.lm_head = MLX::NN::Linear.new(args.hidden_size, args.vocab_size, bias: false)
        end

        def call(inputs, cache: nil)
          out = transformer.call(inputs, cache: cache)
          logits = lm_head.call(out)
          Nanochat.softcap(logits)
        end

        def layers
          transformer.h
        end
      end

      Models.register("nanochat", Model, ModelArgs)
    end
  end
end
