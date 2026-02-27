require_relative "cache"
require_relative "rope_utils"

module MlxLm
  module Models
    module Gemma3Text
      class ModelArgs < BaseModelArgs
        field :model_type, default: "gemma3_text"
        field :hidden_size, default: 1152
        field :num_hidden_layers, default: 26
        field :intermediate_size, default: 6912
        field :num_attention_heads, default: 4
        field :head_dim, default: 256
        field :rms_norm_eps, default: 1.0e-6
        field :vocab_size, default: 262144
        field :num_key_value_heads, default: 1
        field :rope_theta, default: 1_000_000.0
        field :rope_local_base_freq, default: 10_000.0
        field :query_pre_attn_scalar, default: 256.0
        field :sliding_window, default: 512
        field :sliding_window_pattern, default: 6
        field :max_position_embeddings, default: 32768
        field :rope_scaling, default: nil
      end

      class RMSNorm < MLX::NN::Module
        def initialize(dims:, eps: 1e-6)
          super()
          self.weight = MLX::Core.ones([dims])
          @eps = eps
        end

        def call(x)
          mx = MLX::Core
          x_sq = x * x
          mean_sq = mx.mean(x_sq, -1, keepdims: true)
          x * mx.rsqrt(mean_sq + @eps) * (1.0 + weight)
        end
      end

      class Attention < MLX::NN::Module
        def initialize(args, layer_idx)
          super()
          dim = args.hidden_size
          @n_heads = args.num_attention_heads
          @n_kv_heads = args.num_key_value_heads
          @head_dim = args.head_dim
          @scale = args.query_pre_attn_scalar**(-0.5)
          pattern = [args.sliding_window_pattern.to_i, 1].max
          @is_sliding = ((layer_idx + 1) % pattern) != 0

          self.q_proj = MLX::NN::Linear.new(dim, @n_heads * @head_dim, bias: false)
          self.k_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: false)
          self.v_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: false)
          self.o_proj = MLX::NN::Linear.new(@n_heads * @head_dim, dim, bias: false)

          self.q_norm = RMSNorm.new(dims: @head_dim, eps: args.rms_norm_eps)
          self.k_norm = RMSNorm.new(dims: @head_dim, eps: args.rms_norm_eps)

          if @is_sliding
            self.rope = MlxLm::Models.initialize_rope(
              @head_dim,
              args.rope_local_base_freq,
              false
            )
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

          queries = q_proj.call(x).reshape([b, l, @n_heads, @head_dim]).transpose([0, 2, 1, 3])
          keys = k_proj.call(x).reshape([b, l, @n_kv_heads, @head_dim]).transpose([0, 2, 1, 3])
          values = v_proj.call(x).reshape([b, l, @n_kv_heads, @head_dim]).transpose([0, 2, 1, 3])

          queries = q_norm.call(queries)
          keys = k_norm.call(keys)

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
          down_proj.call(MLX::NN.gelu_approx(gate_proj.call(x)) * up_proj.call(x))
        end
      end

      class TransformerBlock < MLX::NN::Module
        def initialize(args, layer_idx)
          super()
          self.self_attn = Attention.new(args, layer_idx)
          self.mlp = MLP.new(args.hidden_size, args.intermediate_size)
          self.input_layernorm = RMSNorm.new(dims: args.hidden_size, eps: args.rms_norm_eps)
          self.post_attention_layernorm = RMSNorm.new(dims: args.hidden_size, eps: args.rms_norm_eps)
          self.pre_feedforward_layernorm = RMSNorm.new(dims: args.hidden_size, eps: args.rms_norm_eps)
          self.post_feedforward_layernorm = RMSNorm.new(dims: args.hidden_size, eps: args.rms_norm_eps)
        end

        def call(x, mask: nil, cache: nil)
          r = self_attn.call(input_layernorm.call(x), mask: mask, cache: cache)
          h = clip_residual(x, post_attention_layernorm.call(r))
          r = mlp.call(pre_feedforward_layernorm.call(h))
          clip_residual(h, post_feedforward_layernorm.call(r))
        end

        private

        def clip_residual(x, y)
          mx = MLX::Core
          return x + y unless x.dtype == mx.float16

          bound = mx.finfo(mx.float16).max
          mx.clip(
            x.astype(mx.float32) + y.astype(mx.float32),
            -bound,
            bound
          ).astype(mx.float16)
        end
      end

      class Gemma3Model < MLX::NN::Module
        attr_reader :sliding_window_pattern

        def initialize(args)
          super()
          @args = args
          @window_size = args.sliding_window
          @sliding_window_pattern = [args.sliding_window_pattern.to_i, 1].max
          self.embed_tokens = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
          self.layers = Array.new(args.num_hidden_layers) do |layer_idx|
            TransformerBlock.new(args, layer_idx)
          end
          self.norm = RMSNorm.new(dims: args.hidden_size, eps: args.rms_norm_eps)
        end

        def call(inputs, cache: nil, input_embeddings: nil)
          h = input_embeddings || embed_tokens.call(inputs)
          h = h * Math.sqrt(@args.hidden_size)
          layer_cache = cache || [nil] * layers.length

          global_idx = sliding_window_pattern - 1
          global_mask = _create_attention_mask(h, layer_cache[global_idx])
          sliding_window_mask = if sliding_window_pattern > 1
            _create_attention_mask(h, layer_cache[0], window_size: @window_size)
          else
            nil
          end

          layers.each_with_index do |layer, i|
            is_global = (i % sliding_window_pattern) == (sliding_window_pattern - 1)
            mask = is_global ? global_mask : sliding_window_mask
            h = layer.call(h, mask: mask, cache: layer_cache[i])
          end

          norm.call(h)
        end

        private

        def _create_attention_mask(h, cache = nil, window_size: nil)
          n = h.shape[1]
          return cache.make_mask(n) if cache && cache.respond_to?(:make_mask)

          if window_size
            offset = cache ? cache.offset : 0
            if cache && cache.instance_variable_defined?(:@max_size)
              max_size = cache.instance_variable_get(:@max_size)
              offset = [max_size - 1, offset].min if max_size && max_size > 0
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
          @tie_word_embeddings = false
          self.model_type = args.model_type
          self.model = Gemma3Model.new(args)
          self.lm_head = MLX::NN::Linear.new(args.hidden_size, args.vocab_size, bias: false)
        end

        def call(inputs, cache: nil, input_embeddings: nil)
          out = model.call(inputs, cache: cache, input_embeddings: input_embeddings)
          if @tie_word_embeddings || lm_head.nil?
            model.embed_tokens.as_linear(out)
          else
            lm_head.call(out)
          end
        end

        def sanitize(weights)
          sanitized = weights.reject { |k, _| k.include?("self_attn.rotary_emb.inv_freq") }
          unless sanitized.key?("lm_head.weight")
            @tie_word_embeddings = true
            self.lm_head = nil
          end
          sanitized
        end

        def layers
          model.layers
        end

        def make_cache
          pattern = [@args.sliding_window_pattern.to_i, 1].max
          max_size = @args.sliding_window || @args.max_position_embeddings || 1
          Array.new(@args.num_hidden_layers) do |i|
            is_global = (i % pattern) == (pattern - 1)
            if is_global
              MlxLm::KVCache.new
            else
              MlxLm::RotatingKVCache.new(max_size: max_size, keep: 0)
            end
          end
        end
      end

      Models.register("gemma3_text", Model, ModelArgs)
    end
  end
end
