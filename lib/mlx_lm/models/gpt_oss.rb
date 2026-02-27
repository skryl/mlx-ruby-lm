require_relative "cache"
require_relative "rope_utils"
require_relative "switch_layers"

module MlxLm
  module Models
    module GptOss
      class ModelArgs < BaseModelArgs
        field :model_type, default: "gpt_oss"
        field :num_hidden_layers, default: 36
        field :num_local_experts, default: 128
        field :num_experts_per_tok, default: 4
        field :vocab_size, default: 201_088
        field :rms_norm_eps, default: 1e-5
        field :hidden_size, default: 2880
        field :intermediate_size, default: 2880
        field :head_dim, default: 64
        field :num_attention_heads, default: 64
        field :num_key_value_heads, default: 8
        field :sliding_window, default: 128
        field :rope_theta, default: 150_000
        field :rope_scaling, default: nil
        field :layer_types, default: nil

        def initialize(**kwargs)
          super
          @layer_types ||= Array.new(@num_hidden_layers) do |i|
            i.even? ? "sliding_attention" : "full_attention"
          end
        end
      end

      class AttentionBlock < MLX::NN::Module
        def initialize(config)
          super()
          @head_dim = config.head_dim
          @num_attention_heads = config.num_attention_heads
          @num_key_value_heads = config.num_key_value_heads
          @sm_scale = 1.0 / Math.sqrt(@head_dim)

          self.q_proj = MLX::NN::Linear.new(
            config.hidden_size,
            @num_attention_heads * @head_dim,
            bias: true
          )
          self.k_proj = MLX::NN::Linear.new(
            config.hidden_size,
            @num_key_value_heads * @head_dim,
            bias: true
          )
          self.v_proj = MLX::NN::Linear.new(
            config.hidden_size,
            @num_key_value_heads * @head_dim,
            bias: true
          )
          self.o_proj = MLX::NN::Linear.new(
            @num_attention_heads * @head_dim,
            config.hidden_size,
            bias: true
          )

          self.rope = MlxLm::Models.initialize_rope(
            @head_dim,
            config.rope_theta,
            false,
            config.rope_scaling
          )
        end

        def call(x, mask:, cache: nil)
          mx = MLX::Core
          b, l, _d = x.shape

          q = q_proj.call(x).reshape([b, l, @num_attention_heads, @head_dim]).transpose([0, 2, 1, 3])
          k = k_proj.call(x).reshape([b, l, @num_key_value_heads, @head_dim]).transpose([0, 2, 1, 3])
          v = v_proj.call(x).reshape([b, l, @num_key_value_heads, @head_dim]).transpose([0, 2, 1, 3])

          if cache
            q = rope.call(q, offset: cache.offset)
            k = rope.call(k, offset: cache.offset)
            k, v = cache.update_and_fetch(k, v)
          else
            q = rope.call(q)
            k = rope.call(k)
          end

          out = mx.scaled_dot_product_attention(q, k, v, @sm_scale, mask)
          out = out.transpose([0, 2, 1, 3]).reshape([b, l, @num_attention_heads * @head_dim])
          o_proj.call(out)
        end
      end

      class MLPBlock < MLX::NN::Module
        def initialize(config)
          super()
          @num_local_experts = config.num_local_experts
          @num_experts_per_tok = config.num_experts_per_tok

          self.experts = SwitchLayers::SwitchGLU.new(
            config.hidden_size,
            config.intermediate_size,
            @num_local_experts,
            bias: true
          )
          self.router = MLX::NN::Linear.new(
            config.hidden_size,
            @num_local_experts,
            bias: true
          )
        end

        def call(x)
          mx = MLX::Core

          gates = router.call(x)
          k = [@num_experts_per_tok, @num_local_experts].min
          inds = mx.stop_gradient(mx.argpartition(gates * -1.0, k - 1, -1))
          take_ids = mx.array((0...k).to_a, dtype: mx.int32)
          inds = mx.take(inds, take_ids, -1)
          expert_weights = mx.take_along_axis(gates, inds, -1)
          expert_weights = mx.softmax(expert_weights.astype(mx.float32), -1).astype(expert_weights.dtype)

          x = experts.call(x, inds)
          x = x * mx.expand_dims(expert_weights, -1)
          mx.sum(x, -2)
        end
      end

      class TransformerBlock < MLX::NN::Module
        def initialize(config)
          super()
          self.self_attn = AttentionBlock.new(config)
          self.mlp = MLPBlock.new(config)
          self.input_layernorm = MLX::NN::RMSNorm.new(config.hidden_size, eps: config.rms_norm_eps)
          self.post_attention_layernorm = MLX::NN::RMSNorm.new(config.hidden_size, eps: config.rms_norm_eps)
        end

        def call(x, mask:, cache: nil)
          h = x + self_attn.call(input_layernorm.call(x), mask: mask, cache: cache)
          h + mlp.call(post_attention_layernorm.call(h))
        end
      end

      class GptOssMoeModel < MLX::NN::Module
        attr_reader :layer_types

        def initialize(args)
          super()
          @window_size = args.sliding_window
          @layer_types = args.layer_types

          self.embed_tokens = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
          self.norm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
          self.layers = Array.new(args.num_hidden_layers) { TransformerBlock.new(args) }

          @swa_idx = @layer_types.index("sliding_attention") || 0
          @ga_idx = @layer_types.index("full_attention") || 0
        end

        def call(inputs, cache: nil, input_embeddings: nil)
          x = input_embeddings || embed_tokens.call(inputs)
          layer_cache = cache || [nil] * layers.length

          full_mask = _create_attention_mask(x, layer_cache[@ga_idx])
          swa_mask = _create_attention_mask(
            x,
            layer_cache[@swa_idx],
            window_size: @window_size
          )

          layers.each_with_index do |layer, i|
            layer_type = @layer_types[i]
            mask = layer_type == "full_attention" ? full_mask : swa_mask
            x = layer.call(x, mask: mask, cache: layer_cache[i])
          end

          norm.call(x)
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
          self.model = GptOssMoeModel.new(args)
          self.lm_head = MLX::NN::Linear.new(args.hidden_size, args.vocab_size, bias: false)
        end

        def call(inputs, cache: nil, input_embeddings: nil)
          lm_head.call(model.call(inputs, cache: cache, input_embeddings: input_embeddings))
        end

        def sanitize(weights)
          return weights if weights.keys.any? { |key| key.include?("gate_proj.weight") }

          result = {}
          weights.each do |key, value|
            if key.include?("gate_up_proj") && !key.include?("bias")
              normalized_key, normalized_value = _normalize_moe_weight_param(key, value)
              split_axis = normalized_value.shape.length - 2
              result[normalized_key.sub("gate_up_proj", "gate_proj")] = _take_every_other(
                normalized_value,
                start: 0,
                axis: split_axis
              )
              result[normalized_key.sub("gate_up_proj", "up_proj")] = _take_every_other(
                normalized_value,
                start: 1,
                axis: split_axis
              )
            elsif key.include?("down_proj") && !key.include?("bias")
              normalized_key, normalized_value = _normalize_moe_weight_param(key, value)
              result[normalized_key] = normalized_value
            elsif key.include?("gate_up_proj_bias")
              split_axis = value.shape.length - 1
              result[key.sub("gate_up_proj_bias", "gate_proj.bias")] = _take_every_other(
                value,
                start: 0,
                axis: split_axis
              )
              result[key.sub("gate_up_proj_bias", "up_proj.bias")] = _take_every_other(
                value,
                start: 1,
                axis: split_axis
              )
            elsif key.include?("down_proj_bias")
              result[key.sub("down_proj_bias", "down_proj.bias")] = value
            else
              result[key] = value
            end
          end

          result
        end

        def layers
          model.layers
        end

        def make_cache
          model.layer_types.map do |layer_type|
            if layer_type == "full_attention"
              MlxLm::KVCache.new
            else
              MlxLm::RotatingKVCache.new(max_size: @args.sliding_window)
            end
          end
        end

        private

        def _normalize_moe_weight_param(key, value)
          mx = MLX::Core
          normalized_key = key
          normalized_value = value

          if key.include?("_blocks")
            normalized_value = mx.flatten(value.view(mx.uint32), -2, -1)
            normalized_key = normalized_key.sub("_blocks", ".weight")
          end
          if key.include?("_scales")
            normalized_key = normalized_key.sub("_scales", ".scales")
          end

          [normalized_key, normalized_value]
        end

        def _take_every_other(value, start:, axis:)
          mx = MLX::Core
          indices = (start...value.shape[axis]).step(2).to_a
          take_ids = mx.array(indices, dtype: mx.int32)
          mx.take(value, take_ids, axis)
        end
      end

      Models.register("gpt_oss", Model, ModelArgs)
    end
  end
end
