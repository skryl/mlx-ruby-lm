module MlxLm
  module Models
    module OLMo3
      class ModelArgs < BaseModelArgs
        field :model_type, default: "olmo3"
        field :hidden_size
        field :num_hidden_layers
        field :intermediate_size
        field :num_attention_heads
        field :rms_norm_eps
        field :vocab_size
        field :max_position_embeddings
        field :sliding_window
        field :rope_theta
        field :attention_bias, default: false
        field :layer_types, default: nil
        field :num_key_value_heads, default: nil
        field :head_dim, default: nil
        field :rope_scaling, default: nil
        field :tie_word_embeddings, default: false

        def initialize(**kwargs)
          super
          @num_key_value_heads ||= @num_attention_heads
          @head_dim ||= @hidden_size / @num_attention_heads
          @layer_types ||= Array.new(@num_hidden_layers) do |i|
            ((i + 1) % 4).zero? ? "full_attention" : "sliding_attention"
          end
        end
      end

      class Olmo3Attention < MLX::NN::Module
        def initialize(args, layer_idx:)
          super()
          @num_attention_heads = args.num_attention_heads
          @num_key_value_heads = args.num_key_value_heads
          @head_dim = args.head_dim
          @scale = @head_dim**(-0.5)

          self.q_proj = MLX::NN::Linear.new(
            args.hidden_size,
            @num_attention_heads * @head_dim,
            bias: args.attention_bias
          )
          self.k_proj = MLX::NN::Linear.new(
            args.hidden_size,
            @num_key_value_heads * @head_dim,
            bias: args.attention_bias
          )
          self.v_proj = MLX::NN::Linear.new(
            args.hidden_size,
            @num_key_value_heads * @head_dim,
            bias: args.attention_bias
          )
          self.o_proj = MLX::NN::Linear.new(
            @num_attention_heads * @head_dim,
            args.hidden_size,
            bias: args.attention_bias
          )

          self.q_norm = MLX::NN::RMSNorm.new(
            @num_attention_heads * @head_dim,
            eps: args.rms_norm_eps
          )
          self.k_norm = MLX::NN::RMSNorm.new(
            @num_key_value_heads * @head_dim,
            eps: args.rms_norm_eps
          )

          if args.layer_types[layer_idx] != "full_attention"
            self.rope = MLX::NN::RoPE.new(@head_dim, traditional: false, base: args.rope_theta)
          else
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

          queries = q_norm.call(q_proj.call(x))
          keys = k_norm.call(k_proj.call(x))
          values = v_proj.call(x)

          queries = queries.reshape([b, l, @num_attention_heads, @head_dim]).transpose([0, 2, 1, 3])
          keys = keys.reshape([b, l, @num_key_value_heads, @head_dim]).transpose([0, 2, 1, 3])
          values = values.reshape([b, l, @num_key_value_heads, @head_dim]).transpose([0, 2, 1, 3])

          if cache
            queries = rope.call(queries, offset: cache.offset)
            keys = rope.call(keys, offset: cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
          else
            queries = rope.call(queries)
            keys = rope.call(keys)
          end

          output = mx.scaled_dot_product_attention(queries, keys, values, @scale, mask)
          output = output.transpose([0, 2, 1, 3]).reshape([b, l, @num_attention_heads * @head_dim])
          o_proj.call(output)
        end
      end

      class Olmo3MLP < MLX::NN::Module
        def initialize(args)
          super()
          self.gate_proj = MLX::NN::Linear.new(args.hidden_size, args.intermediate_size, bias: false)
          self.down_proj = MLX::NN::Linear.new(args.intermediate_size, args.hidden_size, bias: false)
          self.up_proj = MLX::NN::Linear.new(args.hidden_size, args.intermediate_size, bias: false)
        end

        def call(x)
          down_proj.call(Activations.swiglu(gate_proj.call(x), up_proj.call(x)))
        end
      end

      class Olmo3DecoderLayer < MLX::NN::Module
        def initialize(args, layer_idx:)
          super()
          self.self_attn = Olmo3Attention.new(args, layer_idx: layer_idx)
          self.mlp = Olmo3MLP.new(args)
          self.post_attention_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
          self.post_feedforward_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
        end

        def call(x, mask: nil, cache: nil)
          r = post_attention_layernorm.call(self_attn.call(x, mask: mask, cache: cache))
          h = x + r
          r = post_feedforward_layernorm.call(mlp.call(h))
          h + r
        end
      end

      class Olmo3Model < MLX::NN::Module
        attr_reader :layer_types

        def initialize(args)
          super()
          @sliding_window = args.sliding_window
          @layer_types = args.layer_types

          self.embed_tokens = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
          self.layers = Array.new(args.num_hidden_layers) do |i|
            Olmo3DecoderLayer.new(args, layer_idx: i)
          end
          self.norm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)

          self.swa_idx = @layer_types.index("sliding_attention") || 0
          self.ga_idx = @layer_types.index("full_attention") || 0
        end

        def call(inputs, cache: nil)
          h = embed_tokens.call(inputs)
          layer_cache = cache || [nil] * layers.length

          full_mask = _create_attention_mask(h, layer_cache[ga_idx])
          sliding_window_mask = _create_attention_mask(
            h,
            layer_cache[swa_idx],
            window_size: @sliding_window
          )

          layers.each_with_index do |layer, i|
            mask = @layer_types[i] == "full_attention" ? full_mask : sliding_window_mask
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

          if window_size
            offset = 0
            if cache
              offset = cache.offset
              if cache.instance_variable_defined?(:@max_size)
                max_size = cache.instance_variable_get(:@max_size)
                offset = [max_size - 1, offset].min if max_size && max_size > 0
              end
            end
            return _create_causal_mask(n, offset: offset, window_size: window_size) if offset + n > window_size
          end
          return nil if n == 1

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
        attr_reader :args

        def initialize(args)
          super()
          @args = args
          self.model_type = args.model_type
          self.model = Olmo3Model.new(args)
          unless args.tie_word_embeddings
            self.lm_head = MLX::NN::Linear.new(args.hidden_size, args.vocab_size, bias: false)
          end
        end

        def call(inputs, cache: nil)
          out = model.call(inputs, cache: cache)
          if args.tie_word_embeddings
            model.embed_tokens.as_linear(out)
          else
            lm_head.call(out)
          end
        end

        def layers
          model.layers
        end

        def make_cache
          model.layer_types.map do |layer_type|
            if layer_type == "full_attention"
              KVCache.new
            else
              RotatingKVCache.new(max_size: args.sliding_window)
            end
          end
        end
      end

      Models.register("olmo3", Model, ModelArgs)
    end
  end
end
