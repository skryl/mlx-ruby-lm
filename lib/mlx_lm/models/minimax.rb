require_relative "switch_layers"

module MlxLm
  module Models
    module Minimax
      class ModelArgs < BaseModelArgs
        field :model_type, default: "minimax"
        field :hidden_size
        field :intermediate_size
        field :num_attention_heads
        field :num_key_value_heads
        field :max_position_embeddings
        field :num_experts_per_tok
        field :num_local_experts
        field :shared_intermediate_size
        field :num_hidden_layers
        field :rms_norm_eps
        field :rope_theta
        field :rotary_dim
        field :vocab_size
        field :tie_word_embeddings, default: false
        field :scoring_func, default: "sigmoid"
        field :head_dim, default: nil
        field :use_qk_norm, default: true

        def initialize(**kwargs)
          super
          @num_key_value_heads ||= @num_attention_heads
          @head_dim ||= @hidden_size / @num_attention_heads
          @rotary_dim ||= @head_dim
        end
      end

      class Attention < MLX::NN::Module
        def initialize(args)
          super()
          dim = args.hidden_size
          @num_attention_heads = args.num_attention_heads
          @num_key_value_heads = args.num_key_value_heads
          @head_dim = args.head_dim
          @scale = @head_dim**(-0.5)
          @use_qk_norm = args.use_qk_norm

          self.q_proj = MLX::NN::Linear.new(dim, @num_attention_heads * @head_dim, bias: false)
          self.k_proj = MLX::NN::Linear.new(dim, @num_key_value_heads * @head_dim, bias: false)
          self.v_proj = MLX::NN::Linear.new(dim, @num_key_value_heads * @head_dim, bias: false)
          self.o_proj = MLX::NN::Linear.new(@num_attention_heads * @head_dim, dim, bias: false)

          if @use_qk_norm
            self.q_norm = MLX::NN::RMSNorm.new(@head_dim * @num_attention_heads, eps: args.rms_norm_eps)
            self.k_norm = MLX::NN::RMSNorm.new(@head_dim * @num_key_value_heads, eps: args.rms_norm_eps)
          end

          self.rope = MLX::NN::RoPE.new(args.rotary_dim, traditional: false, base: args.rope_theta)
        end

        def call(x, mask: nil, cache: nil)
          mx = MLX::Core
          b, l, _d = x.shape

          queries = q_proj.call(x)
          keys = k_proj.call(x)
          values = v_proj.call(x)

          if @use_qk_norm
            queries = q_norm.call(queries)
            keys = k_norm.call(keys)
          end

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

      class SparseMoeBlock < MLX::NN::Module
        def initialize(args)
          super()
          mx = MLX::Core
          @num_experts_per_tok = args.num_experts_per_tok
          @num_local_experts = args.num_local_experts

          self.gate = MLX::NN::Linear.new(args.hidden_size, @num_local_experts, bias: false)
          self.switch_mlp = SwitchLayers::SwitchGLU.new(args.hidden_size, args.intermediate_size, @num_local_experts)
          self.e_score_correction_bias = mx.zeros([@num_local_experts])
        end

        def call(x)
          mx = MLX::Core

          gates = gate.call(x.astype(mx.float32))
          orig_scores = mx.sigmoid(gates)
          scores = orig_scores + e_score_correction_bias

          k = [[@num_experts_per_tok.to_i, 1].max, @num_local_experts.to_i].min
          inds = mx.argpartition(scores * -1.0, k - 1, -1)
          take_ids = mx.array((0...k).to_a, dtype: mx.int32)
          inds = mx.take(inds, take_ids, -1)

          scores = mx.take_along_axis(orig_scores, inds, -1)
          scores = scores / (mx.expand_dims(mx.sum(scores, -1), -1) + 1e-20)
          scores = scores.astype(x.dtype)

          y = switch_mlp.call(x, inds)
          mx.sum(y * mx.expand_dims(scores, -1), -2)
        end
      end

      class DecoderLayer < MLX::NN::Module
        def initialize(args)
          super()
          self.self_attn = Attention.new(args)
          self.block_sparse_moe = SparseMoeBlock.new(args)
          self.input_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
          self.post_attention_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
        end

        def call(x, mask: nil, cache: nil)
          h = x + self_attn.call(input_layernorm.call(x), mask: mask, cache: cache)
          h + block_sparse_moe.call(post_attention_layernorm.call(h))
        end
      end

      class MiniMaxModel < MLX::NN::Module
        def initialize(args)
          super()
          self.embed_tokens = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
          self.layers = Array.new(args.num_hidden_layers) { DecoderLayer.new(args) }
          self.norm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
        end

        def call(inputs, mask: nil, cache: nil)
          h = embed_tokens.call(inputs)
          layer_cache = cache || [nil] * layers.length
          local_mask = mask || _create_attention_mask(h, layer_cache[0])

          layers.each_with_index do |layer, i|
            h = layer.call(h, mask: local_mask, cache: layer_cache[i])
          end

          norm.call(h)
        end

        private

        def _create_attention_mask(h, cache)
          n = h.shape[1]
          return cache.make_mask(n) if cache && cache.respond_to?(:make_mask)
          return nil if n == 1

          "causal"
        end
      end

      class Model < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.model_type = args.model_type
          self.model = MiniMaxModel.new(args)
          self.lm_head = MLX::NN::Linear.new(args.hidden_size, args.vocab_size, bias: false) unless args.tie_word_embeddings
        end

        def call(inputs, mask: nil, cache: nil)
          out = model.call(inputs, mask: mask, cache: cache)
          if @args.tie_word_embeddings
            model.embed_tokens.as_linear(out)
          else
            lm_head.call(out)
          end
        end

        def sanitize(weights)
          mx = MLX::Core
          dequantized = {}

          weights.each do |key, value|
            if key.include?("weight_scale_inv")
              weight_key = key.sub("_scale_inv", "")
              next unless weights.key?(weight_key)

              dequantized[weight_key] = _dequant(weights[weight_key], value)
            elsif !dequantized.key?(key)
              dequantized[key] = value
            end
          end

          result = dequantized
          return result unless result.key?("model.layers.0.block_sparse_moe.experts.0.w1.weight")

          mapping = {
            "w1" => "gate_proj",
            "w2" => "down_proj",
            "w3" => "up_proj",
          }
          experts_count = @args.num_local_experts.to_i
          return result if experts_count <= 0

          @args.num_hidden_layers.times do |layer_idx|
            prefix = "model.layers.#{layer_idx}"
            mapping.each do |old_name, new_name|
              first_key = "#{prefix}.block_sparse_moe.experts.0.#{old_name}.weight"
              next unless result.key?(first_key)

              expert_keys = (0...experts_count).map do |expert_idx|
                "#{prefix}.block_sparse_moe.experts.#{expert_idx}.#{old_name}.weight"
              end
              next unless expert_keys.all? { |k| result.key?(k) }

              stacked = expert_keys.map { |k| result.delete(k) }
              result["#{prefix}.block_sparse_moe.switch_mlp.#{new_name}.weight"] = mx.stack(stacked)
            end
          end

          result
        end

        def layers
          model.layers
        end

        def cast_predicate
          lambda { |key| !key.include?("e_score_correction_bias") }
        end

        def quant_predicate
          lambda do |path, _|
            if path.end_with?("block_sparse_moe.gate")
              { group_size: 64, bits: 8 }
            else
              true
            end
          end
        end

        private

        def _dequant(weight, scale_inv)
          mx = MLX::Core
          dtype = mx.bfloat16
          block_size = 128

          dequantized = mx.from_fp8(weight, dtype: dtype)
          m, n = dequantized.shape
          pad_bottom = block_size * scale_inv.shape[0] - m
          pad_side = block_size * scale_inv.shape[1] - n

          dequantized = mx.pad(dequantized, [[0, pad_bottom], [0, pad_side]])
          dequantized = dequantized.reshape([
            (m + pad_bottom) / block_size,
            block_size,
            (n + pad_side) / block_size,
            block_size,
          ])

          scaled = dequantized * scale_inv.reshape([scale_inv.shape[0], 1, scale_inv.shape[1], 1])
          scaled = scaled.reshape([m + pad_bottom, n + pad_side])
          scaled = mx.split(scaled, [m], 0)[0]
          scaled = mx.split(scaled, [n], 1)[0]
          scaled.astype(dtype)
        rescue StandardError
          weight
        end
      end

      Models.register("minimax", Model, ModelArgs)
    end
  end
end
