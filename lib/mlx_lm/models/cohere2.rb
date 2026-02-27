module MlxLm
  module Models
    module Cohere2
      class ModelArgs < BaseModelArgs
        field :model_type, default: "cohere2"
        field :hidden_size, default: 4096
        field :head_dim, default: 128
        field :num_hidden_layers, default: 32
        field :intermediate_size, default: 14336
        field :num_attention_heads, default: 32
        field :num_key_value_heads, default: 8
        field :rope_theta, default: 50_000.0
        field :vocab_size, default: 256000
        field :layer_norm_eps, default: 1e-5
        field :logit_scale, default: 0.0625
        field :attention_bias, default: false
        field :layer_norm_bias, default: false
        field :sliding_window, default: 4096
        field :sliding_window_pattern, default: 4

        def initialize(**kwargs)
          super
          @num_key_value_heads ||= @num_attention_heads
        end
      end

      class Attention < MLX::NN::Module
        def initialize(args, layer_idx)
          super()
          dim = args.hidden_size
          @n_heads = args.num_attention_heads
          @n_kv_heads = args.num_key_value_heads
          @head_dim = args.head_dim
          if (@head_dim * @n_heads) != dim
            raise ArgumentError,
              "hidden_size must equal num_attention_heads * head_dim (got #{dim} and #{@n_heads} * #{@head_dim})"
          end
          @scale = @head_dim**(-0.5)

          bias = args.attention_bias
          self.q_proj = MLX::NN::Linear.new(dim, @n_heads * @head_dim, bias: bias)
          self.k_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: bias)
          self.v_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: bias)
          self.o_proj = MLX::NN::Linear.new(@n_heads * @head_dim, dim, bias: bias)

          self.rope = MLX::NN::RoPE.new(@head_dim, traditional: true, base: args.rope_theta)
          @use_sliding_window = ((layer_idx + 1) % args.sliding_window_pattern) != 0
        end

        def call(x, mask: nil, cache: nil)
          mx = MLX::Core
          b, l, _d = x.shape

          queries = q_proj.call(x).reshape([b, l, @n_heads, @head_dim]).transpose([0, 2, 1, 3])
          keys = k_proj.call(x).reshape([b, l, @n_kv_heads, @head_dim]).transpose([0, 2, 1, 3])
          values = v_proj.call(x).reshape([b, l, @n_kv_heads, @head_dim]).transpose([0, 2, 1, 3])

          if @use_sliding_window
            if cache
              queries = rope.call(queries, offset: cache.offset)
              keys = rope.call(keys, offset: cache.offset)
            else
              queries = rope.call(queries)
              keys = rope.call(keys)
            end
          end

          keys, values = cache.update_and_fetch(keys, values) if cache

          sdpa_type = queries.dtype == mx.float16 ? mx.float32 : queries.dtype
          output = mx.scaled_dot_product_attention(
            queries.astype(sdpa_type),
            keys,
            values,
            @scale,
            mask
          ).astype(queries.dtype)

          output = output.transpose([0, 2, 1, 3]).reshape([b, l, @n_heads * @head_dim])
          o_proj.call(output)
        end
      end

      class MLP < MLX::NN::Module
        def initialize(dim, hidden_dim)
          super()
          self.gate_proj = MLX::NN::Linear.new(dim, hidden_dim, bias: false)
          self.up_proj = MLX::NN::Linear.new(dim, hidden_dim, bias: false)
          self.down_proj = MLX::NN::Linear.new(hidden_dim, dim, bias: false)
        end

        def call(x)
          down_proj.call(Activations.swiglu(gate_proj.call(x), up_proj.call(x)))
        end
      end

      class TransformerBlock < MLX::NN::Module
        def initialize(args, layer_idx)
          super()
          self.self_attn = Attention.new(args, layer_idx)
          self.mlp = MLP.new(args.hidden_size, args.intermediate_size)
          self.input_layernorm = MLX::NN::LayerNorm.new(
            args.hidden_size,
            eps: args.layer_norm_eps,
            bias: args.layer_norm_bias
          )
        end

        def call(x, mask: nil, cache: nil)
          h = input_layernorm.call(x)
          attn_h = self_attn.call(h, mask: mask, cache: cache)
          ff_h = mlp.call(h)
          attn_h + ff_h + x
        end
      end

      class Cohere2Model < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          @window_size = args.sliding_window

          self.embed_tokens = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
          self.layers = Array.new(args.num_hidden_layers) { |i| TransformerBlock.new(args, i) }
          self.norm = MLX::NN::LayerNorm.new(
            args.hidden_size,
            eps: args.layer_norm_eps,
            bias: args.layer_norm_bias
          )
        end

        def call(inputs, cache: nil)
          h = embed_tokens.call(inputs)
          layer_cache = cache || [nil] * layers.length

          pattern = @args.sliding_window_pattern
          full_mask = _create_attention_mask(h, layer_cache[pattern - 1])
          swa_mask = _create_attention_mask(h, layer_cache[0], window_size: @window_size)

          layers.each_with_index do |layer, i|
            is_global = (i % pattern) == (pattern - 1)
            mask = is_global ? full_mask : swa_mask
            h = layer.call(h, mask: mask, cache: layer_cache[i])
          end

          norm.call(h)
        end

        private

        def _create_attention_mask(h, cache, window_size: nil)
          n = h.shape[1]
          offset = cache ? cache.offset : 0

          if window_size
            if cache || n > window_size
              return _create_causal_mask(n, offset, window_size)
            end
            return nil if n == 1

            return "causal"
          end

          return nil if n == 1

          "causal"
        end

        def _create_causal_mask(n, offset, window_size = nil)
          mx = MLX::Core
          rinds = mx.arange(offset + n)
          linds = offset.zero? ? rinds : mx.arange(offset, offset + n)

          linds = mx.expand_dims(linds, 1)
          rinds = mx.expand_dims(rinds, 0)
          mask = mx.greater_equal(linds, rinds)

          if window_size
            mask = mx.logical_and(mask, mx.less(linds, mx.add(rinds, window_size)))
          end

          mask
        end
      end

      class Model < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.model = Cohere2Model.new(args)
        end

        def call(inputs, cache: nil)
          out = model.call(inputs, cache: cache)
          out = model.embed_tokens.as_linear(out)
          out * @args.logit_scale
        end

        def make_cache
          caches = []
          @args.num_hidden_layers.times do |i|
            is_global = (i % @args.sliding_window_pattern) == (@args.sliding_window_pattern - 1)
            if is_global
              caches << MlxLm::KVCache.new
            else
              caches << MlxLm::RotatingKVCache.new(max_size: @args.sliding_window, keep: 0)
            end
          end
          caches
        end

        def sanitize(weights)
          weights.reject { |k, _| k.include?("self_attn.rotary_emb.inv_freq") }
        end

        def layers
          model.layers
        end
      end

      Models.register("cohere2", Model, ModelArgs)
    end
  end
end
