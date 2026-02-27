module MlxLm
  module Models
    module BaichuanM1
      class ModelArgs < BaseModelArgs
        field :vocab_size
        field :hidden_size
        field :intermediate_size
        field :num_hidden_layers
        field :num_attention_heads
        field :num_key_value_heads
        field :rope_theta
        field :sliding_window
        field :sliding_window_layers
        field :conv_window
        field :rms_norm_eps
        field :model_type, default: "baichuan_m1"
        field :num_swa_attention_heads, default: nil
        field :num_swa_key_value_heads, default: nil
        field :tie_word_embeddings, default: false
      end

      class Attention < MLX::NN::Module
        def initialize(config, layer_idx: nil)
          super()

          raise ArgumentError, "Layer index must be provided to Attention module." if layer_idx.nil?

          swa_layers = config.sliding_window_layers || []
          @is_swa = swa_layers.include?(layer_idx)

          @num_heads = if @is_swa && config.num_swa_attention_heads
            config.num_swa_attention_heads
          else
            config.num_attention_heads
          end

          @num_kv_heads = if @is_swa && config.num_swa_key_value_heads
            config.num_swa_key_value_heads
          else
            config.num_key_value_heads
          end

          @hidden_size = config.hidden_size
          @head_dim = @hidden_size / @num_heads

          unless (@head_dim * @num_heads) == @hidden_size
            raise ArgumentError, "hidden_size must be divisible by num_heads"
          end

          @scale = @head_dim**(-0.5)

          self.w_pack = MLX::NN::Linear.new(
            config.hidden_size,
            @hidden_size + 2 * @num_kv_heads * @head_dim,
            bias: false
          )
          self.o_proj = MLX::NN::Linear.new(
            @num_heads * @head_dim,
            config.hidden_size,
            bias: false
          )

          self.rope = MLX::NN::RoPE.new(@head_dim, traditional: false, base: config.rope_theta)

          @conv_window = config.conv_window
          raise ArgumentError, "conv_window must be 2" unless @conv_window == 2

          mx = MLX::Core
          self.conv_k = mx.zeros([1, 1, @num_kv_heads, 1, @conv_window])
          self.conv_v = mx.zeros([1, 1, @num_kv_heads, 1, @conv_window])
        end

        def call(x, mask: nil, cache: nil)
          mx = MLX::Core
          b, l, d = x.shape

          proj = w_pack.call(x)
          q, k, v = mx.split(proj, [d, d + @num_kv_heads * @head_dim], -1)

          q = q.reshape([b, l, @num_heads, @head_dim]).transpose([0, 2, 1, 3])
          k = k.reshape([b, l, @num_kv_heads, @head_dim]).transpose([0, 2, 1, 3])
          v = v.reshape([b, l, @num_kv_heads, @head_dim]).transpose([0, 2, 1, 3])

          layer_cache = cache || [nil, nil]
          conv_cache = layer_cache[0]
          kv_cache = layer_cache[1]

          if conv_cache
            offset = kv_cache.offset
            last_k = conv_cache[0]
            last_v = conv_cache[1]
          else
            offset = 0
            last_k = nil
            last_v = nil
          end

          k_init = k
          v_init = v

          k = _custom_convolution(k, conv_k, state: last_k)
          v = _custom_convolution(v, conv_v, state: last_v)
          q = rope.call(q, offset: offset)
          k = rope.call(k, offset: offset)

          if conv_cache
            k, v = kv_cache.update_and_fetch(k, v)
            if l > 0
              conv_cache[0] = mx.split(k_init, [l - 1], 2)[1]
              conv_cache[1] = mx.split(v_init, [l - 1], 2)[1]
            end
          end

          out = mx.scaled_dot_product_attention(q, k, v, @scale, mask)
          out = out.transpose([0, 2, 1, 3]).reshape([b, l, @num_heads * @head_dim])
          o_proj.call(out)
        end

        private

        def _custom_convolution(u, weights, state: nil)
          mx = MLX::Core
          b, h, l, d = u.shape

          weights = weights.reshape([1, h, @conv_window, 1, 1])
          w0 = mx.take(weights, 0, 2)
          w1 = mx.take(weights, 1, 2)

          state ||= mx.zeros([b, h, 1, d], u.dtype)
          if l > 1
            u_prev = mx.concatenate([state, mx.split(u, [l - 1], 2)[0]], 2)
          else
            u_prev = state
          end

          mx.add(mx.multiply(u_prev, w0), mx.multiply(u, w1))
        end
      end

      class MLP < MLX::NN::Module
        def initialize(config)
          super()
          self.gate_proj = MLX::NN::Linear.new(
            config.hidden_size,
            config.intermediate_size,
            bias: false
          )
          self.up_proj = MLX::NN::Linear.new(
            config.hidden_size,
            config.intermediate_size,
            bias: false
          )
          self.down_proj = MLX::NN::Linear.new(
            config.intermediate_size,
            config.hidden_size,
            bias: false
          )
        end

        def call(x)
          down_proj.call(Activations.swiglu(gate_proj.call(x), up_proj.call(x)))
        end
      end

      class DecoderLayer < MLX::NN::Module
        def initialize(config, layer_idx)
          super()
          self.self_attn = Attention.new(config, layer_idx: layer_idx)
          self.mlp = MLP.new(config)
          self.input_layernorm = MLX::NN::RMSNorm.new(config.hidden_size, eps: config.rms_norm_eps)
          self.post_attention_layernorm = MLX::NN::RMSNorm.new(config.hidden_size, eps: config.rms_norm_eps)
        end

        def call(x, mask: nil, cache: nil)
          r = self_attn.call(input_layernorm.call(x), mask: mask, cache: cache)
          h = x + r
          r = mlp.call(post_attention_layernorm.call(h))
          h + r
        end
      end

      class BaichuanModel < MLX::NN::Module
        def initialize(config)
          super()
          @config = config
          @sliding_window = config.sliding_window
          @swa_layers = config.sliding_window_layers || []

          self.embed_tokens = MLX::NN::Embedding.new(config.vocab_size, config.hidden_size)
          self.layers = Array.new(config.num_hidden_layers) { |i| DecoderLayer.new(config, i) }
          self.norm = MLX::NN::RMSNorm.new(config.hidden_size, eps: config.rms_norm_eps)

          self.first_swa_idx = @swa_layers.empty? ? nil : @swa_layers[0]
          self.first_global_idx = nil
          config.num_hidden_layers.times do |i|
            next if @swa_layers.include?(i)

            self.first_global_idx = i
            break
          end
        end

        def call(inputs, cache: nil)
          x = embed_tokens.call(inputs)
          layer_cache = cache || Array.new(layers.length) { [nil, nil] }

          c_global = first_global_idx.nil? ? nil : layer_cache[first_global_idx][1]
          c_swa = first_swa_idx.nil? ? nil : layer_cache[first_swa_idx][1]

          global_mask = _create_attention_mask(x, c_global)
          swa_mask = _create_attention_mask(x, c_swa, window_size: @sliding_window)

          layers.each_with_index do |layer, i|
            mask = @swa_layers.include?(i) ? swa_mask : global_mask
            x = layer.call(x, mask: mask, cache: layer_cache[i])
          end

          norm.call(x)
        end

        private

        def _create_attention_mask(x, cache = nil, window_size: nil)
          n = x.shape[1]
          return cache.make_mask(n, window_size: window_size) if cache && cache.respond_to?(:make_mask)
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
        def initialize(config)
          super()
          @config = config
          self.model_type = config.model_type
          self.model = BaichuanModel.new(config)
          @tie_word_embeddings = config.tie_word_embeddings
          unless @tie_word_embeddings
            self.lm_head = MLX::NN::Linear.new(config.hidden_size, config.vocab_size, bias: false)
          end
        end

        def make_cache
          caches = []
          swa_layers = @config.sliding_window_layers || []
          @config.num_hidden_layers.times do |i|
            is_swa = swa_layers.include?(i)
            conv_cache = MlxLm::ArraysCache.new(2)
            kv_cache = if is_swa
              MlxLm::RotatingKVCache.new(max_size: @config.sliding_window)
            else
              MlxLm::KVCache.new
            end
            caches << MlxLm::CacheList.new(conv_cache, kv_cache)
          end
          caches
        end

        def sanitize(weights)
          mx = MLX::Core
          is_quantized = weights.key?("lm_head.scales")

          if !is_quantized && weights.key?("lm_head.weight")
            w = weights["lm_head.weight"]
            dtype = w.dtype
            w = w.astype(mx.float32)
            norm = mx.norm(w, nil, -1, true)
            w = (w / (norm + 1e-7)).astype(dtype)
            weights["lm_head.weight"] = w
          end

          weights
        end

        def call(inputs, cache: nil)
          out = model.call(inputs, cache: cache)
          if @tie_word_embeddings
            model.embed_tokens.as_linear(out)
          else
            lm_head.call(out)
          end
        end

        def layers
          model.layers
        end
      end

      Models.register("baichuan_m1", Model, ModelArgs)
    end
  end
end
