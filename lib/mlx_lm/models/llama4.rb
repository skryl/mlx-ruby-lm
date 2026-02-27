require_relative "activations"
require_relative "cache"
require_relative "rope_utils"
require_relative "switch_layers"

module MlxLm
  module Models
    module Llama4
      class TextArgs < BaseModelArgs
        field :model_type, default: "llama4_text"
        field :attention_bias, default: false
        field :attention_chunk_size, default: 1024
        field :head_dim, default: nil
        field :hidden_size
        field :interleave_moe_layer_step, default: 1
        field :intermediate_size
        field :intermediate_size_mlp, default: nil
        field :max_position_embeddings, default: 4096
        field :num_attention_heads
        field :num_experts_per_tok, default: 1
        field :num_hidden_layers
        field :num_key_value_heads, default: nil
        field :num_local_experts, default: 1
        field :rms_norm_eps, default: 1e-5
        field :rope_scaling, default: nil
        field :rope_theta, default: 10_000.0
        field :use_qk_norm, default: false
        field :vocab_size
        field :attn_temperature_tuning, default: 4
        field :floor_scale, default: 8192
        field :attn_scale, default: 0.1

        def initialize(**kwargs)
          super
          @num_key_value_heads ||= @num_attention_heads
          @head_dim ||= @hidden_size / @num_attention_heads
          @intermediate_size_mlp ||= @intermediate_size
          @attention_chunk_size = [@attention_chunk_size.to_i, 1].max
          @interleave_moe_layer_step = [@interleave_moe_layer_step.to_i, 1].max
        end
      end

      class ModelArgs < BaseModelArgs
        field :model_type, default: "llama4"
        field :text_config, default: nil

        def self.from_dict(params)
          has_text_config = params.key?("text_config") || params.key?(:text_config)
          return super if has_text_config

          new(model_type: params["model_type"] || params[:model_type], text_config: params)
        end

        def initialize(**kwargs)
          super
          @text_config = _to_text_args(@text_config || {})
        end

        private

        def _to_text_args(config)
          return config if config.is_a?(TextArgs)

          normalized = {}
          config.each { |key, value| normalized[key.to_s] = value }
          TextArgs.from_dict(normalized)
        end
      end

      class Attention < MLX::NN::Module
        def initialize(args, layer_idx)
          super()

          dim = args.hidden_size
          @n_heads = args.num_attention_heads
          @n_kv_heads = args.num_key_value_heads
          @head_dim = args.head_dim
          @scale = @head_dim**(-0.5)
          @use_rope = ((layer_idx + 1) % 4) != 0
          @attn_temperature_tuning = args.attn_temperature_tuning
          @floor_scale = args.floor_scale
          @attn_scale = args.attn_scale
          @use_qk_norm = args.use_qk_norm && @use_rope

          self.q_proj = MLX::NN::Linear.new(dim, @n_heads * @head_dim, bias: args.attention_bias)
          self.k_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: args.attention_bias)
          self.v_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: args.attention_bias)
          self.o_proj = MLX::NN::Linear.new(@n_heads * @head_dim, dim, bias: args.attention_bias)

          if @use_rope
            self.rope = MlxLm::Models.initialize_rope(
              @head_dim,
              args.rope_theta,
              true,
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

          offset = cache ? cache.offset : 0
          if @use_rope
            queries = rope.call(queries, offset: offset)
            keys = rope.call(keys, offset: offset)
          end

          if @use_qk_norm
            queries = mx.rms_norm(queries, nil, 1e-6)
            keys = mx.rms_norm(keys, nil, 1e-6)
          end

          if @attn_temperature_tuning && !@use_rope
            attn_scales = (mx.log(mx.floor(mx.arange(offset + 1, offset + l + 1) / @floor_scale) + 1.0) * @attn_scale) + 1.0
            queries = (queries * attn_scales.reshape([l, 1])).astype(queries.dtype)
          end

          keys, values = cache.update_and_fetch(keys, values) if cache

          output = mx.scaled_dot_product_attention(queries, keys, values, @scale, mask)
          output = output.transpose([0, 2, 1, 3]).reshape([b, l, @n_heads * @head_dim])
          o_proj.call(output)
        end
      end

      class MLP < MLX::NN::Module
        def initialize(args, intermediate_size = nil)
          super()
          dim = args.hidden_size
          hidden_dim = intermediate_size || args.intermediate_size

          self.gate_proj = MLX::NN::Linear.new(dim, hidden_dim, bias: false)
          self.down_proj = MLX::NN::Linear.new(hidden_dim, dim, bias: false)
          self.up_proj = MLX::NN::Linear.new(dim, hidden_dim, bias: false)
        end

        def call(x)
          down_proj.call(Activations.swiglu(gate_proj.call(x), up_proj.call(x)))
        end
      end

      class MoE < MLX::NN::Module
        def initialize(args)
          super()
          @top_k = args.num_experts_per_tok
          raise ArgumentError, "Only 1 expert per token supported" unless @top_k == 1

          @num_experts = args.num_local_experts
          self.experts = SwitchLayers::SwitchGLU.new(
            args.hidden_size,
            args.intermediate_size,
            @num_experts
          )
          self.router = MLX::NN::Linear.new(args.hidden_size, @num_experts, bias: false)
          self.shared_expert = MLP.new(args)
        end

        def call(x)
          mx = MLX::Core
          logits = router.call(x)

          indices = mx.argpartition(logits * -1.0, @top_k - 1, -1)
          take_ids = mx.array((0...@top_k).to_a, dtype: mx.int32)
          indices = mx.take(indices, take_ids, -1)
          scores = mx.take_along_axis(logits, indices, -1)
          scores = mx.sigmoid(scores.astype(mx.float32)).astype(x.dtype)

          out = mx.squeeze(experts.call(x * scores, indices), 2)
          out + shared_expert.call(x)
        end
      end

      class TransformerBlock < MLX::NN::Module
        def initialize(args, layer_idx)
          super()
          self.self_attn = Attention.new(args, layer_idx)
          is_moe_layer = (layer_idx % args.interleave_moe_layer_step) == (args.interleave_moe_layer_step - 1)
          if is_moe_layer
            self.feed_forward = MoE.new(args)
          else
            self.feed_forward = MLP.new(args, args.intermediate_size_mlp)
          end
          self.input_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
          self.post_attention_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
        end

        def call(x, mask: nil, cache: nil)
          r = self_attn.call(input_layernorm.call(x), mask: mask, cache: cache)
          h = x + r
          r = feed_forward.call(post_attention_layernorm.call(h))
          h + r
        end
      end

      class LlamaModel < MLX::NN::Module
        def initialize(args)
          super()
          @attention_chunk_size = args.attention_chunk_size
          self.embed_tokens = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
          self.layers = Array.new(args.num_hidden_layers) { |i| TransformerBlock.new(args, i) }
          self.norm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
        end

        def call(inputs, cache: nil)
          mx = MLX::Core
          h = embed_tokens.call(inputs)
          layer_cache = cache || Array.new(layers.length)

          if cache
            cache.each_with_index do |c, idx|
              next unless ((idx + 1) % 4) != 0
              next unless c && c.respond_to?(:maybe_trim_front)

              c.maybe_trim_front
            end
            first_cache = cache[0]
            start = first_cache&.respond_to?(:start_position) ? first_cache.start_position : 0
            offset = first_cache&.respond_to?(:offset) ? first_cache.offset : 0
          else
            start = 0
            offset = 0
          end

          finish = offset + h.shape[1]
          linds = mx.arange(start, finish)
          rinds = mx.arange(offset, finish).reshape([h.shape[1], 1])

          block_pos = mx.abs(
            mx.floor_divide(linds, @attention_chunk_size) -
            mx.floor_divide(rinds, @attention_chunk_size)
          )
          token_pos = mx.less_equal(linds, rinds)
          chunk_mask = mx.logical_and(mx.equal(block_pos, 0), token_pos)
          global_mask = _create_attention_mask(h, layer_cache[3])

          layers.each_with_index do |layer, idx|
            use_chunked_attention = ((idx + 1) % 4) != 0
            mask = use_chunked_attention ? chunk_mask : global_mask
            h = layer.call(h, mask: mask, cache: layer_cache[idx])
          end

          norm.call(h)
        end

        private

        def _create_attention_mask(h, cache = nil)
          return cache.make_mask(h.shape[1]) if cache && cache.respond_to?(:make_mask)
          return nil if h.shape[1] == 1

          "causal"
        end
      end

      class LanguageModel < MLX::NN::Module
        def initialize(args)
          super()
          self.model = LlamaModel.new(args)
          self.lm_head = MLX::NN::Linear.new(args.hidden_size, args.vocab_size, bias: false)
        end

        def call(inputs, cache: nil)
          lm_head.call(model.call(inputs, cache: cache))
        end
      end

      class Model < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.model_type = args.model_type
          self.language_model = LanguageModel.new(args.text_config)
        end

        def call(inputs, cache: nil)
          language_model.call(inputs, cache: cache)
        end

        def sanitize(weights)
          mx = MLX::Core

          sanitized = {}
          weights.each do |key, value|
            next if _multimodal_key?(key)

            sanitized[key] = value
          end

          @args.text_config.num_hidden_layers.to_i.times do |layer_idx|
            prefix = "language_model.model.layers.#{layer_idx}.feed_forward.experts"

            gate_up = _pop_first(
              sanitized,
              ["#{prefix}.gate_up_proj", "#{prefix}.gate_up_proj.weight"]
            )
            if gate_up
              split = gate_up.shape[-1] / 2
              gate_proj, up_proj = mx.split(gate_up, [split], -1)
              sanitized["#{prefix}.gate_proj.weight"] = mx.swapaxes(gate_proj, 1, 2)
              sanitized["#{prefix}.up_proj.weight"] = mx.swapaxes(up_proj, 1, 2)
            end

            down_proj = _pop_first(
              sanitized,
              ["#{prefix}.down_proj", "#{prefix}.down_proj.weight"]
            )
            if down_proj
              sanitized["#{prefix}.down_proj.weight"] = mx.swapaxes(down_proj, 1, 2)
            end
          end

          sanitized
        end

        def layers
          language_model.model.layers
        end

        def make_cache
          chunk_size = [@args.text_config.attention_chunk_size.to_i, 1].max
          Array.new(layers.length) do |i|
            if ((i + 1) % 4) != 0
              MlxLm::ChunkedKVCache.new(chunk_size)
            else
              MlxLm::KVCache.new
            end
          end
        end

        private

        def _pop_first(weights, keys)
          keys.each do |key|
            return weights.delete(key) if weights.key?(key)
          end
          nil
        end

        def _multimodal_key?(key)
          key_name = key.to_s
          key_name.include?("vision_model") ||
            key_name.include?("vision_tower") ||
            key_name.include?("multi_modal_projector")
        end
      end

      Models.register("llama4", Model, ModelArgs)
    end
  end
end
