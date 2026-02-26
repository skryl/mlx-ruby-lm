module MlxLm
  module Models
    module Exaone4
      class ModelArgs < BaseModelArgs
        field :model_type, default: "exaone4"
        field :hidden_size
        field :num_hidden_layers
        field :intermediate_size
        field :num_attention_heads
        field :rms_norm_eps
        field :vocab_size
        field :num_key_value_heads
        field :max_position_embeddings
        field :rope_theta
        field :head_dim
        field :tie_word_embeddings
        field :rope_scaling, default: nil
        field :sliding_window, default: nil
        field :sliding_window_pattern, default: nil

        def initialize(**kwargs)
          super
          @num_key_value_heads ||= @num_attention_heads
          @head_dim ||= @hidden_size / @num_attention_heads
        end
      end

      class Attention < MLX::NN::Module
        attr_reader :is_local

        def initialize(args, is_local)
          super()

          dim = args.hidden_size
          @n_heads = args.num_attention_heads
          @n_kv_heads = args.num_key_value_heads
          @head_dim = args.head_dim
          @scale = @head_dim**(-0.5)

          self.q_proj = MLX::NN::Linear.new(dim, @n_heads * @head_dim, bias: false)
          self.k_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: false)
          self.v_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: false)
          self.o_proj = MLX::NN::Linear.new(@n_heads * @head_dim, dim, bias: false)

          self.q_norm = MLX::NN::RMSNorm.new(@head_dim, eps: args.rms_norm_eps)
          self.k_norm = MLX::NN::RMSNorm.new(@head_dim, eps: args.rms_norm_eps)

          @is_local = is_local || false
          @use_rope = is_local.nil? || @is_local
          if @use_rope
            self.rope = MlxLm::Models.initialize_rope(
              @head_dim,
              args.rope_theta,
              false,
              args.rope_scaling,
              max_position_embeddings: args.max_position_embeddings
            )
          end
        end

        def call(x, mask: nil, cache: nil)
          mx = MLX::Core
          b, l, _d = x.shape

          queries = q_proj.call(x)
          keys = k_proj.call(x)
          values = v_proj.call(x)

          queries = q_norm.call(queries.reshape([b, l, @n_heads, @head_dim])).transpose([0, 2, 1, 3])
          keys = k_norm.call(keys.reshape([b, l, @n_kv_heads, @head_dim])).transpose([0, 2, 1, 3])
          values = values.reshape([b, l, @n_kv_heads, @head_dim]).transpose([0, 2, 1, 3])

          if cache
            if @use_rope
              queries = rope.call(queries, offset: cache.offset)
              keys = rope.call(keys, offset: cache.offset)
            end
            keys, values = cache.update_and_fetch(keys, values)
          elsif @use_rope
            queries = rope.call(queries)
            keys = rope.call(keys)
          end

          output = mx.scaled_dot_product_attention(queries, keys, values, @scale, mask)
          output = output.transpose([0, 2, 1, 3]).reshape([b, l, @n_heads * @head_dim])
          o_proj.call(output)
        end
      end

      class MLP < MLX::NN::Module
        def initialize(dim, hidden_dim)
          super()
          self.gate_proj = MLX::NN::Linear.new(dim, hidden_dim, bias: false)
          self.down_proj = MLX::NN::Linear.new(hidden_dim, dim, bias: false)
          self.up_proj = MLX::NN::Linear.new(dim, hidden_dim, bias: false)
        end

        def call(x)
          down_proj.call(Activations.swiglu(gate_proj.call(x), up_proj.call(x)))
        end
      end

      class TransformerBlock < MLX::NN::Module
        def initialize(args, is_local:)
          super()
          self.self_attn = Attention.new(args, is_local)
          self.mlp = MLP.new(args.hidden_size, args.intermediate_size)
          self.post_attention_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
          self.post_feedforward_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
        end

        def call(x, mask: nil, cache: nil)
          r = self_attn.call(x, mask: mask, cache: cache)
          h = x + post_attention_layernorm.call(r)
          r = mlp.call(h)
          h + post_feedforward_layernorm.call(r)
        end
      end

      class ExaoneModel < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.vocab_size = args.vocab_size
          self.num_hidden_layers = args.num_hidden_layers
          self.embed_tokens = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)

          pattern = args.sliding_window_pattern
          self.layers = Array.new(args.num_hidden_layers) do |i|
            is_local = pattern ? (pattern[i % pattern.length] == "L") : nil
            TransformerBlock.new(args, is_local: is_local)
          end

          if pattern
            self.swa_idx = pattern.index("L")
            self.full_idx = pattern.index("G")
          else
            self.swa_idx = nil
            self.full_idx = 0
          end

          self.window_size = args.sliding_window
          self.norm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
        end

        def call(inputs, cache: nil)
          h = embed_tokens.call(inputs)
          layer_cache = cache || [nil] * layers.length

          global_mask = _create_attention_mask(h, layer_cache[full_idx])
          if !swa_idx.nil?
            swa_mask = _create_attention_mask(
              h,
              layer_cache[swa_idx],
              window_size: window_size
            )
          else
            swa_mask = nil
          end

          layers.each_with_index do |layer, i|
            mask = layer.self_attn.is_local ? swa_mask : global_mask
            h = layer.call(h, mask: mask, cache: layer_cache[i])
          end

          norm.call(h)
        end

        private

        def _create_attention_mask(h, cache = nil, window_size: nil)
          n = h.shape[1]
          if cache && cache.respond_to?(:make_mask)
            return cache.make_mask(n, window_size: window_size)
          end
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
      end

      class Model < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.model_type = args.model_type
          self.model = ExaoneModel.new(args)
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

        def make_cache
          layers.map do |layer|
            if layer.self_attn.is_local
              RotatingKVCache.new(max_size: @args.sliding_window, keep: 0)
            else
              KVCache.new
            end
          end
        end

        def layers
          model.layers
        end
      end

      Models.register("exaone4", Model, ModelArgs)
    end
  end
end
