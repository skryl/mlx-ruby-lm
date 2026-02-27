require_relative "activations"

module MlxLm
  module Models
    module Dbrx
      class ModelArgs < BaseModelArgs
        field :model_type, default: "dbrx"
        field :vocab_size, default: 32_000
        field :d_model, default: 6144
        field :ffn_config, default: {}
        field :attn_config, default: {}
        field :n_layers, default: 40
        field :n_heads, default: 48

        def initialize(**kwargs)
          super
          @ffn_config ||= {}
          @attn_config ||= {}
        end
      end

      class Attention < MLX::NN::Module
        def initialize(args)
          super()

          @num_heads = args.n_heads
          @d_model = args.d_model
          @head_dim = @d_model / args.n_heads
          @num_key_value_heads = _attn_value(args.attn_config, "kv_n_heads", args.n_heads).to_i
          @clip_qkv = _attn_value(args.attn_config, "clip_qkv", 8.0).to_f
          @rope_theta = _attn_value(args.attn_config, "rope_theta", 10_000.0).to_f
          @scale = @head_dim**(-0.5)

          self.wqkv = MLX::NN::Linear.new(
            args.d_model,
            (@num_key_value_heads * 2 + @num_heads) * @head_dim,
            bias: false
          )
          self.out_proj = MLX::NN::Linear.new(args.d_model, args.d_model, bias: false)
          self.rope = MLX::NN::RoPE.new(@head_dim, traditional: false, base: @rope_theta)
        end

        def call(x, mask: nil, cache: nil)
          mx = MLX::Core
          b, l, _d = x.shape

          qkv = wqkv.call(x)
          qkv = mx.clip(qkv, -@clip_qkv, @clip_qkv)

          splits = [@d_model, @d_model + @head_dim * @num_key_value_heads]
          queries, keys, values = mx.split(qkv, splits, -1)

          queries = queries.reshape([b, l, @num_heads, @head_dim]).transpose([0, 2, 1, 3])
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
          output = output.transpose([0, 2, 1, 3]).reshape([b, l, @d_model])
          out_proj.call(output)
        end

        private

        def _attn_value(config, key, default = nil)
          return default if config.nil?
          return config[key] if config.key?(key)

          config.fetch(key.to_sym, default)
        end
      end

      class NormAttnNorm < MLX::NN::Module
        def initialize(args)
          super()
          self.norm_1 = MLX::NN::LayerNorm.new(args.d_model, bias: false)
          self.norm_2 = MLX::NN::LayerNorm.new(args.d_model, bias: false)
          self.attn = Attention.new(args)
        end

        def call(x, mask: nil, cache: nil)
          h = attn.call(norm_1.call(x), mask: mask, cache: cache)
          residual = x + h
          [residual, norm_2.call(residual)]
        end
      end

      class MLP < MLX::NN::Module
        def initialize(d_model, ffn_dim)
          super()
          self.v1 = MLX::NN::Linear.new(d_model, ffn_dim, bias: false)
          self.w1 = MLX::NN::Linear.new(d_model, ffn_dim, bias: false)
          self.w2 = MLX::NN::Linear.new(ffn_dim, d_model, bias: false)
        end

        def call(x)
          w2.call(Activations.swiglu(w1.call(x), v1.call(x)))
        end
      end

      class Router < MLX::NN::Module
        def initialize(d_model, num_experts)
          super()
          self.layer = MLX::NN::Linear.new(d_model, num_experts, bias: false)
        end

        def call(x)
          layer.call(x)
        end
      end

      class SparseMoeBlock < MLX::NN::Module
        def initialize(args)
          super()
          @d_model = args.d_model
          @ffn_dim = _ffn_value(args.ffn_config, "ffn_hidden_size", args.d_model * 4).to_i
          @num_experts = _ffn_value(args.ffn_config, "moe_num_experts", 1).to_i
          @num_experts_per_tok = _ffn_value(args.ffn_config, "moe_top_k", 1).to_i

          self.router = Router.new(@d_model, @num_experts)
          self.experts = Array.new(@num_experts) { MLP.new(@d_model, @ffn_dim) }
        end

        def call(x)
          mx = MLX::Core

          top_k = [[@num_experts_per_tok, 1].max, @num_experts].min
          orig_shape = x.shape
          token_count = orig_shape[0...-1].reduce(1, :*)
          flat_x = x.reshape([token_count, orig_shape[-1]])

          gates = router.call(flat_x)
          gates = mx.softmax(gates.astype(mx.float32), -1)

          inds = mx.stop_gradient(mx.argpartition(gates * -1.0, top_k - 1, -1))
          take_ids = mx.array((0...top_k).to_a, dtype: mx.int32)
          inds = mx.take(inds, take_ids, -1)
          scores = mx.take_along_axis(gates, inds, -1)
          scores = scores / mx.expand_dims(mx.sum(scores, -1), -1)
          scores = scores.astype(flat_x.dtype)

          expert_ids = inds.to_a
          expert_scores = scores.to_a

          outputs = Array.new(flat_x.shape[0]) do |token_idx|
            token_ids = mx.array([token_idx], dtype: mx.int32)
            token_state = mx.squeeze(mx.take(flat_x, token_ids, 0), 0)

            token_out = nil
            expert_ids[token_idx].each_with_index do |expert_idx, score_idx|
              expert_out = experts[expert_idx.to_i].call(token_state)
              weighted = expert_out * expert_scores[token_idx][score_idx].to_f
              token_out = token_out.nil? ? weighted : (token_out + weighted)
            end

            token_out
          end

          mx.stack(outputs, 0).reshape(orig_shape)
        end

        private

        def _ffn_value(config, key, default = nil)
          return default if config.nil?
          return config[key] if config.key?(key)

          config.fetch(key.to_sym, default)
        end
      end

      class DecoderLayer < MLX::NN::Module
        def initialize(args)
          super()
          self.ffn = SparseMoeBlock.new(args)
          self.norm_attn_norm = NormAttnNorm.new(args)
        end

        def call(x, mask: nil, cache: nil)
          residual, hidden = norm_attn_norm.call(x, mask: mask, cache: cache)
          ffn.call(hidden) + residual
        end
      end

      class DbrxModel < MLX::NN::Module
        def initialize(args)
          super()
          self.wte = MLX::NN::Embedding.new(args.vocab_size, args.d_model)
          self.blocks = Array.new(args.n_layers) { DecoderLayer.new(args) }
          self.norm_f = MLX::NN::LayerNorm.new(args.d_model, bias: false)
        end

        def call(inputs, cache: nil)
          h = wte.call(inputs)
          layer_cache = cache || [nil] * blocks.length
          mask = _create_attention_mask(h, layer_cache[0])

          blocks.each_with_index do |layer, layer_idx|
            h = layer.call(h, mask: mask, cache: layer_cache[layer_idx])
          end

          norm_f.call(h)
        end

        private

        def _create_attention_mask(hidden, cache)
          return cache.make_mask(hidden.shape[1]) if cache && cache.respond_to?(:make_mask)
          return nil if hidden.shape[1] == 1

          "causal"
        end
      end

      class Model < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.model_type = args.model_type
          self.transformer = DbrxModel.new(args)
          self.lm_head = MLX::NN::Linear.new(args.d_model, args.vocab_size, bias: false)
        end

        def call(inputs, cache: nil)
          out = transformer.call(inputs, cache: cache)
          lm_head.call(out)
        end

        def layers
          transformer.blocks
        end

        def sanitize(weights)
          mx = MLX::Core
          num_experts = _ffn_value(@args.ffn_config, "moe_num_experts", 0).to_i
          return weights if num_experts <= 0

          pattern = "experts.mlp"
          sanitized = {}

          weights.each do |key, value|
            unless key.include?(pattern)
              sanitized[key] = value
              next
            end

            split_weights = mx.split(value, num_experts, 0)
            split_weights.each_with_index do |slice, expert_idx|
              expert_key = _expert_weight_key(key, expert_idx)
              if key.end_with?("w2") || key.end_with?("w2.weight")
                slice = slice.transpose([1, 0])
              end
              sanitized[expert_key] = slice
            end
          end

          sanitized
        end

        private

        def _expert_weight_key(key, expert_idx)
          base = key.end_with?(".weight") ? key.sub(/\.weight\z/, "") : key
          "#{base.sub('.mlp', ".#{expert_idx}")}.weight"
        end

        def _ffn_value(config, key, default = nil)
          return default if config.nil?
          return config[key] if config.key?(key)

          config.fetch(key.to_sym, default)
        end
      end

      Models.register("dbrx", Model, ModelArgs)
    end
  end
end
