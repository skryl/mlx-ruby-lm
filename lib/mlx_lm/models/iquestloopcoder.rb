require_relative "cache"
require_relative "rope_utils"

module MlxLm
  module Models
    module Iquestloopcoder
      class ModelArgs < BaseModelArgs
        field :model_type, default: "iquestloopcoder"
        field :hidden_size
        field :num_hidden_layers
        field :intermediate_size
        field :num_attention_heads
        field :rms_norm_eps
        field :vocab_size
        field :head_dim
        field :num_key_value_heads
        field :max_position_embeddings, default: 131_072
        field :attention_bias, default: false
        field :mlp_bias, default: false
        field :rope_theta, default: 500_000.0
        field :rope_scaling, default: nil
        field :tie_word_embeddings, default: false
        field :loop_num, default: 2
        field :loop_window_size, default: 64

        def initialize(**kwargs)
          super
          @num_key_value_heads ||= @num_attention_heads
          @head_dim ||= @hidden_size / @num_attention_heads
        end
      end

      class LoopGateProjection < MLX::NN::Module
        def initialize(num_heads, head_dim)
          super()
          @num_heads = num_heads
          @head_dim = head_dim

          mx = MLX::Core
          self.weight = mx.zeros([num_heads, head_dim])
          self.bias = mx.zeros([num_heads])
        end

        def call(query)
          mx = MLX::Core
          projection = weight.reshape([@num_heads, @head_dim, 1])
          gate_logits = mx.matmul(query, projection)
          gate_logits = gate_logits + bias.reshape([1, @num_heads, 1, 1])
          mx.sigmoid(gate_logits)
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

          self.q_proj = MLX::NN::Linear.new(dim, @n_heads * @head_dim, bias: args.attention_bias)
          self.k_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: args.attention_bias)
          self.v_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: args.attention_bias)
          self.o_proj = MLX::NN::Linear.new(@n_heads * @head_dim, dim, bias: args.attention_bias)

          self.rope = MlxLm::Models.initialize_rope(
            @head_dim,
            args.rope_theta,
            false,
            args.rope_scaling,
            max_position_embeddings: args.max_position_embeddings
          )
        end

        def get_qkv(x, offset: 0)
          b, l, _d = x.shape

          queries = q_proj.call(x).reshape([b, l, @n_heads, @head_dim]).transpose([0, 2, 1, 3])
          keys = k_proj.call(x).reshape([b, l, @n_kv_heads, @head_dim]).transpose([0, 2, 1, 3])
          values = v_proj.call(x).reshape([b, l, @n_kv_heads, @head_dim]).transpose([0, 2, 1, 3])

          queries = rope.call(queries, offset: offset)
          keys = rope.call(keys, offset: offset)

          [queries, keys, values]
        end

        def attention(queries, keys, values, mask: nil, cache: nil)
          _cache = cache
          MLX::Core.scaled_dot_product_attention(queries, keys, values, @scale, mask)
        end
      end

      class MLP < MLX::NN::Module
        def initialize(args)
          super()
          dim = args.hidden_size
          hidden_dim = args.intermediate_size
          self.gate_proj = MLX::NN::Linear.new(dim, hidden_dim, bias: args.mlp_bias)
          self.down_proj = MLX::NN::Linear.new(hidden_dim, dim, bias: args.mlp_bias)
          self.up_proj = MLX::NN::Linear.new(dim, hidden_dim, bias: args.mlp_bias)
        end

        def call(x)
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
      end

      class IQuestLoopCoderModel < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          unless args.loop_num == 2
            raise ArgumentError, "Only loop_num=2 is supported, got #{args.loop_num}"
          end

          self.vocab_size = args.vocab_size
          self.embed_tokens = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
          self.layers = Array.new(args.num_hidden_layers) { TransformerBlock.new(args) }
          self.norm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
          self.gate_projections = Array.new(args.num_hidden_layers) do
            LoopGateProjection.new(args.num_attention_heads, args.head_dim)
          end
          self.loop_num = args.loop_num
          self.loop_window_size = args.loop_window_size
        end

        def call(inputs, cache: nil)
          b, l = inputs.shape[0], inputs.shape[1]

          h = embed_tokens.call(inputs)
          layer_count = layers.length
          layer_cache = cache || [nil] * (2 * layer_count)

          mask = _create_attention_mask(h, layer_cache[0])
          window_mask = _create_attention_mask(h, layer_cache[layer_count], window_size: loop_window_size)

          loop1_kv = []
          layers.each_with_index do |layer, idx|
            c = layer_cache[idx]
            h_norm = layer.input_layernorm.call(h)
            offset = c ? c.offset : 0
            q1, k1, v1 = layer.self_attn.get_qkv(h_norm, offset: offset)

            if c
              k1, v1 = c.update_and_fetch(k1, v1)
            end
            loop1_kv << [k1, v1]

            out = layer.self_attn.attention(q1, k1, v1, mask: mask, cache: c)
            r = layer.self_attn.o_proj.call(out.transpose([0, 2, 1, 3]).reshape([b, l, @args.hidden_size]))
            h = h + r
            r = layer.mlp.call(layer.post_attention_layernorm.call(h))
            h = h + r
          end

          layers.each_with_index do |layer, idx|
            gate_proj = gate_projections[idx]
            c = layer_cache[layer_count + idx]
            k1, v1 = loop1_kv[idx]

            h_norm = layer.input_layernorm.call(h)
            offset = c ? c.offset : 0
            q2, k2, v2 = layer.self_attn.get_qkv(h_norm, offset: offset)

            gate = gate_proj.call(q2)
            attn_global = layer.self_attn.attention(q2, k1, v1, mask: mask, cache: c)

            if c
              k2, v2 = c.update_and_fetch(k2, v2)
            end

            attn_local = layer.self_attn.attention(q2, k2, v2, mask: window_mask, cache: c)
            mixed = _mix_attention(gate, attn_global, attn_local)

            r = layer.self_attn.o_proj.call(mixed.transpose([0, 2, 1, 3]).reshape([b, l, @args.hidden_size]))
            h = h + r
            r = layer.mlp.call(layer.post_attention_layernorm.call(h))
            h = h + r
          end

          norm.call(h)
        end

        private

        def _create_attention_mask(h, cache = nil, window_size: nil)
          n = h.shape[1]
          return cache.make_mask(n) if cache && cache.respond_to?(:make_mask)
          return nil if n == 1
          return _create_causal_mask(n, window_size: window_size) if window_size && n > window_size

          "causal"
        end

        def _create_causal_mask(n, offset: 0, window_size: nil)
          mx = MLX::Core
          rinds = mx.arange(0, offset + n, 1, mx.int32).reshape([1, offset + n])
          linds = mx.arange(offset, offset + n, 1, mx.int32).reshape([n, 1])

          mask = mx.greater_equal(linds, rinds)
          if window_size
            mask = mx.logical_and(mask, mx.less(linds, mx.add(rinds, window_size)))
          end
          mask
        end

        def _mix_attention(gate, attn_global, attn_local)
          gate = gate.astype(attn_global.dtype)
          (gate * attn_global) + ((1.0 - gate) * attn_local)
        end
      end

      class Model < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.model_type = args.model_type
          self.model = IQuestLoopCoderModel.new(args)
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

        def make_cache
          Array.new(layers.length) { MlxLm::KVCache.new } +
            Array.new(layers.length) { MlxLm::RotatingKVCache.new(max_size: @args.loop_window_size) }
        end
      end

      Models.register("iquestloopcoder", Model, ModelArgs)
    end

    IQuestLoopCoder = Iquestloopcoder unless const_defined?(:IQuestLoopCoder)
  end
end
