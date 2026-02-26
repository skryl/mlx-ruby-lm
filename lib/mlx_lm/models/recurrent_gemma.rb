require_relative "cache"

module MlxLm
  module Models
    module RecurrentGemma
      class ModelArgs < BaseModelArgs
        field :model_type, default: "recurrent_gemma"
        field :attention_bias
        field :conv1d_width
        field :hidden_size
        field :intermediate_size
        field :logits_soft_cap
        field :num_attention_heads
        field :num_hidden_layers
        field :num_key_value_heads
        field :rms_norm_eps
        field :rope_theta
        field :attention_window_size
        field :vocab_size
        field :embeddings_scale_by_sqrt_dim, default: true
        field :block_types, default: nil
        field :_block_types, default: nil

        def initialize(**kwargs)
          super
          @block_types ||= @_block_types
          @block_types ||= ["recurrent", "attention"]
        end
      end

      class RMSNorm < MLX::NN::Module
        def initialize(dims, eps: 1e-5)
          super()
          self.weight = MLX::Core.ones([dims])
          @eps = eps
        end

        def call(x)
          mx = MLX::Core
          mean_sq = mx.mean(x * x, -1, keepdims: true)
          norm = x * mx.rsqrt(mean_sq + @eps)
          norm * (weight + 1.0)
        end
      end

      class RGLRU < MLX::NN::Module
        def initialize(width:, num_heads:)
          super()
          @width = width
          @num_heads = num_heads
          @head_dim = @width / @num_heads

          mx = MLX::Core
          self.recurrent_param = mx.zeros([@width])
          self.input_gate_weight = mx.zeros([@num_heads, @head_dim, @head_dim])
          self.input_gate_bias = mx.zeros([@num_heads, @head_dim])
          self.recurrent_gate_weight = mx.zeros([@num_heads, @head_dim, @head_dim])
          self.recurrent_gate_bias = mx.zeros([@num_heads, @head_dim])
        end

        def call(x, cache: nil)
          mx = MLX::Core
          b, l, _ = x.shape

          gate_x = _apply_block_linear(x, input_gate_weight, input_gate_bias, batch: b, seq: l)
          gate_a = _apply_block_linear(x, recurrent_gate_weight, recurrent_gate_bias, batch: b, seq: l)

          log_a = -8.0 * gate_a * MLX::NN.softplus(recurrent_param)
          a = mx.exp(log_a)
          a_square = mx.exp(2.0 * log_a)

          gated_x = x * gate_x
          multiplier = mx.sqrt(1.0 - a_square)
          if cache.nil?
            first = mx.ones([b, 1, @width], multiplier.dtype)
            if l == 1
              multiplier = first
            else
              rest = mx.split(multiplier, [1], 1)[1]
              multiplier = mx.concatenate([first, rest], 1)
            end
          end

          normalized_x = gated_x * multiplier.astype(x.dtype)
          _rnn_scan(normalized_x, a, cache)
        end

        private

        def _apply_block_linear(h, w, b, batch:, seq:)
          mx = MLX::Core
          h = h.reshape([batch, seq, @num_heads, @head_dim]).transpose([0, 2, 1, 3])
          h = mx.matmul(h, w).transpose([0, 2, 1, 3]) + b
          mx.sigmoid(h.reshape([batch, seq, @width]))
        end

        def _rnn_scan(x, a, h0)
          mx = MLX::Core
          b, l, d = x.shape

          if l == 1
            if h0.nil?
              return x, _slice_step(x, 0)
            end

            y = a * mx.expand_dims(h0, 1) + x
            return y, _slice_step(y, 0)
          end

          h_t = h0 || mx.zeros([b, d], x.dtype)
          ys = []
          l.times do |t|
            h_t = _slice_step(a, t) * h_t + _slice_step(x, t)
            ys << h_t
          end
          [mx.stack(ys, 1), h_t]
        end

        def _slice_step(array, idx)
          mx = MLX::Core
          idx_arr = mx.array([idx], dtype: mx.int32)
          mx.squeeze(mx.take(array, idx_arr, 1), 1)
        end
      end

      class RecurrentBlock < MLX::NN::Module
        def initialize(width:, num_heads:, lru_width: nil, conv1d_temporal_width: 4)
          super()
          @width = width
          @num_heads = num_heads
          @lru_width = lru_width || width
          @conv1d_temporal_width = conv1d_temporal_width

          self.linear_y = MLX::NN::Linear.new(width, @lru_width)
          self.linear_x = MLX::NN::Linear.new(width, @lru_width)
          self.linear_out = MLX::NN::Linear.new(@lru_width, width)
          self.conv_1d = MLX::NN::Conv1d.new(
            @lru_width,
            @lru_width,
            @conv1d_temporal_width,
            groups: @lru_width,
            bias: true,
            padding: 0
          )
          self.rg_lru = RGLRU.new(width: @lru_width, num_heads: @num_heads)
        end

        def call(x, cache: nil, mask: nil)
          _ = mask
          mx = MLX::Core

          y = MLX::NN.gelu_approx(linear_y.call(x))
          x = linear_x.call(x)

          conv_cache = _read_cache(cache, 0)
          rnn_cache = _read_cache(cache, 1)

          x = if conv_cache
            mx.concatenate([conv_cache, x], 1)
          else
            mx.pad(x, [[0, 0], [@conv1d_temporal_width - 1, 0], [0, 0]])
          end

          conv_input = x
          x = conv_1d.call(x)
          _write_cache(cache, 0, _tail_cache(conv_input))

          x, last_h = rg_lru.call(x, cache: rnn_cache)
          _write_cache(cache, 1, last_h)

          linear_out.call(x * y)
        end

        private

        def _tail_cache(full_x)
          mx = MLX::Core
          n_keep = @conv1d_temporal_width - 1
          return mx.zeros([full_x.shape[0], 0, full_x.shape[2]], full_x.dtype) if n_keep <= 0

          split_at = full_x.shape[1] - n_keep
          mx.split(full_x, [split_at], 1)[1]
        end

        def _read_cache(cache, idx)
          if cache.is_a?(MlxLm::ArraysCache) || cache.is_a?(Array)
            cache[idx]
          else
            nil
          end
        end

        def _write_cache(cache, idx, value)
          return unless cache.is_a?(MlxLm::ArraysCache) || cache.is_a?(Array)

          cache[idx] = value
        end
      end

      class LocalAttentionBlock < MLX::NN::Module
        def initialize(width:, num_heads:, window_size:)
          super()
          @width = width
          @num_heads = num_heads
          @window_size = window_size
          @scale = (width / num_heads)**(-0.5)
          @head_dim = @width / @num_heads

          self.q_proj = MLX::NN::Linear.new(@width, @width, bias: false)
          self.k_proj = MLX::NN::Linear.new(@width, @head_dim, bias: false)
          self.v_proj = MLX::NN::Linear.new(@width, @head_dim, bias: false)
          self.o_proj = MLX::NN::Linear.new(@width, @width, bias: true)
          self.rope = MLX::NN::RoPE.new(@head_dim / 2, traditional: false)
        end

        def call(x, cache: nil, mask: nil)
          mx = MLX::Core
          b, l, _ = x.shape

          queries = q_proj.call(x)
          keys = k_proj.call(x)
          values = v_proj.call(x)

          queries = queries.reshape([b, l, @num_heads, @head_dim]).transpose([0, 2, 1, 3])
          keys = keys.reshape([b, l, 1, @head_dim]).transpose([0, 2, 1, 3])
          values = values.reshape([b, l, 1, @head_dim]).transpose([0, 2, 1, 3])

          if cache
            queries = rope.call(queries, offset: cache.offset)
            keys = rope.call(keys, offset: cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
          else
            queries = rope.call(queries)
            keys = rope.call(keys)
          end

          output = mx.scaled_dot_product_attention(queries, keys, values, @scale, mask)
          output = output.transpose([0, 2, 1, 3]).reshape([b, l, @width])
          o_proj.call(output)
        end
      end

      class MLPBlock < MLX::NN::Module
        def initialize(width:, expanded_width:)
          super()
          hidden = expanded_width / 2
          self.up_proj = MLX::NN::Linear.new(width, hidden)
          self.gate_proj = MLX::NN::Linear.new(width, hidden)
          self.down_proj = MLX::NN::Linear.new(hidden, width)
        end

        def call(x)
          down_proj.call(MLX::NN.gelu_approx(gate_proj.call(x)) * up_proj.call(x))
        end
      end

      class ResidualBlock < MLX::NN::Module
        attr_reader :temporal_block_type

        def initialize(
          width:,
          mlp_expanded_width:,
          num_heads:,
          attention_window_size:,
          temporal_block_type:,
          lru_width: nil,
          conv1d_temporal_width: 4
        )
          super()
          @temporal_block_type = temporal_block_type

          self.temporal_pre_norm = RMSNorm.new(width)
          self.temporal_block = if temporal_block_type == "recurrent"
            RecurrentBlock.new(
              width: width,
              num_heads: num_heads,
              lru_width: lru_width,
              conv1d_temporal_width: conv1d_temporal_width
            )
          else
            LocalAttentionBlock.new(
              width: width,
              num_heads: num_heads,
              window_size: attention_window_size
            )
          end

          self.channel_pre_norm = RMSNorm.new(width)
          self.mlp_block = MLPBlock.new(width: width, expanded_width: mlp_expanded_width)
        end

        def call(x, cache: nil, mask: nil)
          raw_x = x
          x = temporal_block.call(temporal_pre_norm.call(raw_x), cache: cache, mask: mask)
          residual = x + raw_x
          x = mlp_block.call(channel_pre_norm.call(residual))
          x + residual
        end
      end

      class Griffin < MLX::NN::Module
        attr_reader :window_size, :swa_idx

        def initialize(config)
          super()
          @config = config
          @scale_by_sqrt_dim = config.embeddings_scale_by_sqrt_dim

          block_types = Array(config.block_types)
          block_types = ["recurrent"] if block_types.empty?

          self.embed_tokens = MLX::NN::Embedding.new(config.vocab_size, config.hidden_size)
          self.layers = Array.new(config.num_hidden_layers) do |i|
            ResidualBlock.new(
              width: config.hidden_size,
              mlp_expanded_width: config.intermediate_size,
              num_heads: config.num_attention_heads,
              attention_window_size: config.attention_window_size,
              temporal_block_type: block_types[i % block_types.length],
              lru_width: nil,
              conv1d_temporal_width: config.conv1d_width
            )
          end
          self.final_norm = RMSNorm.new(config.hidden_size, eps: config.rms_norm_eps)

          @window_size = config.attention_window_size
          @swa_idx = block_types.index("attention") || 0
        end

        def call(tokens, cache: nil)
          x = embed_tokens.call(tokens)
          x = x * Math.sqrt(x.shape[-1]) if @scale_by_sqrt_dim

          layer_cache = cache || [nil] * layers.length
          mask = _create_attention_mask(x, layer_cache[@swa_idx], window_size: @window_size)

          layers.each_with_index do |block, i|
            x = block.call(x, mask: mask, cache: layer_cache[i])
          end

          final_norm.call(x)
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

        def initialize(config)
          super()
          @args = config
          @tie_word_embeddings = false
          self.model_type = config.model_type
          self.model = Griffin.new(config)
          self.lm_head = MLX::NN::Linear.new(config.hidden_size, config.vocab_size, bias: false)
        end

        def call(tokens, cache: nil)
          mx = MLX::Core
          logits = model.call(tokens, cache: cache)
          logits = if @tie_word_embeddings || lm_head.nil?
            model.embed_tokens.as_linear(logits)
          else
            lm_head.call(logits)
          end

          c = args.logits_soft_cap
          logits = mx.tanh(logits / c) * c if c && c != 0
          logits
        end

        def layers
          model.layers
        end

        def sanitize(weights)
          mx = MLX::Core
          sanitized = {}
          weights.each do |key, value|
            current = value
            if key.include?("conv_1d.weight") && value.shape[-1] != 1
              current = mx.swapaxes(value, 1, 2)
            end
            sanitized[key] = current
          end

          unless sanitized.key?("lm_head.weight")
            @tie_word_embeddings = true
            self.lm_head = nil
          end

          sanitized
        end

        def make_cache
          layers.map do |layer|
            if layer.temporal_block_type == "recurrent"
              MlxLm::ArraysCache.new(2)
            else
              MlxLm::RotatingKVCache.new(max_size: args.attention_window_size)
            end
          end
        end
      end

      Models.register("recurrent_gemma", Model, ModelArgs)
    end
  end
end
