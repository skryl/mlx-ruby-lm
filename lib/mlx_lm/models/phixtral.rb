require_relative "switch_layers"

module MlxLm
  module Models
    module Phixtral
      class ModelArgs < BaseModelArgs
        field :model_type, default: "phixtral"
        field :num_vocab, default: 51_200
        field :model_dim, default: 2_560
        field :num_heads, default: 32
        field :num_layers, default: 32
        field :rotary_dim, default: 32
        field :num_experts_per_tok, default: 2
        field :num_local_experts, default: 4
      end

      class RoPEAttention < MLX::NN::Module
        def initialize(dims, num_heads, rotary_dim)
          super()
          @num_heads = num_heads
          @head_dim = dims / num_heads
          @scale = @head_dim**(-0.5)

          self.rope = MLX::NN::RoPE.new(rotary_dim, traditional: false)
          self.wqkv = MLX::NN::Linear.new(dims, 3 * dims)
          self.out_proj = MLX::NN::Linear.new(dims, dims)
        end

        def call(x, mask: nil, cache: nil)
          mx = MLX::Core
          b, l, d = x.shape

          qkv = wqkv.call(x)
          queries, keys, values = mx.split(qkv, [d, 2 * d], -1)

          queries = queries.reshape([b, l, @num_heads, @head_dim]).transpose([0, 2, 1, 3])
          keys = keys.reshape([b, l, @num_heads, @head_dim]).transpose([0, 2, 1, 3])
          values = values.reshape([b, l, @num_heads, @head_dim]).transpose([0, 2, 1, 3])

          if cache
            queries = rope.call(queries, offset: cache.offset)
            keys = rope.call(keys, offset: cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
          else
            queries = rope.call(queries)
            keys = rope.call(keys)
          end

          queries = queries.astype(mx.float32)
          output = mx.scaled_dot_product_attention(queries, keys, values, @scale, mask).astype(values.dtype)
          output = output.transpose([0, 2, 1, 3]).reshape([b, l, d])
          out_proj.call(output)
        end
      end

      class MOE < MLX::NN::Module
        def initialize(args, dim, hidden_dim)
          super()
          @num_experts = args.num_local_experts
          @num_experts_per_tok = args.num_experts_per_tok

          self.switch_mlp = SwitchLayers::SwitchMLP.new(
            dim,
            hidden_dim,
            @num_experts,
            bias: true
          )
          self.gate = MLX::NN::Linear.new(args.model_dim, @num_experts, bias: false)
        end

        def call(x)
          mx = MLX::Core
          k = @num_experts_per_tok

          gates = gate.call(x)
          inds = mx.stop_gradient(mx.argpartition(gates * -1.0, k - 1, -1))
          take_ids = mx.array((0...k).to_a, dtype: mx.int32)
          inds = mx.take(inds, take_ids, -1)

          scores = mx.take_along_axis(gates, inds, -1)
          scores = mx.softmax(scores.astype(mx.float32), -1).astype(gates.dtype)

          y = switch_mlp.call(x, inds)
          mx.sum(y * mx.expand_dims(scores, -1), -2)
        end
      end

      class ParallelBlock < MLX::NN::Module
        def initialize(config)
          super()
          dims = config.model_dim
          mlp_dims = dims * 4

          self.mixer = RoPEAttention.new(dims, config.num_heads, config.rotary_dim)
          self.ln = MLX::NN::LayerNorm.new(dims)
          self.moe = MOE.new(config, dims, mlp_dims)
        end

        def call(x, mask: nil, cache: nil)
          h = ln.call(x)
          attn_h = mixer.call(h, mask: mask, cache: cache)
          ff_h = moe.call(h)
          attn_h + ff_h + x
        end
      end

      class Embd < MLX::NN::Module
        def initialize(config)
          super()
          self.wte = MLX::NN::Embedding.new(config.num_vocab, config.model_dim)
        end

        def call(x)
          wte.call(x)
        end
      end

      class TransformerDecoder < MLX::NN::Module
        def initialize(config)
          super()
          self.embd = Embd.new(config)
          self.h = Array.new(config.num_layers) { ParallelBlock.new(config) }
        end

        def call(x, mask: nil, cache: nil)
          hidden = embd.call(x)
          layer_cache = cache || [nil] * h.length

          h.each_with_index do |layer, i|
            hidden = layer.call(hidden, mask: mask, cache: layer_cache[i])
          end

          hidden
        end
      end

      class OutputHead < MLX::NN::Module
        def initialize(config)
          super()
          self.ln = MLX::NN::LayerNorm.new(config.model_dim)
          self.linear = MLX::NN::Linear.new(config.model_dim, config.num_vocab)
        end

        def call(inputs)
          linear.call(ln.call(inputs))
        end
      end

      class Model < MLX::NN::Module
        def initialize(config)
          super()
          @args = config

          self.model_type = config.model_type
          self.transformer = TransformerDecoder.new(config)
          self.lm_head = OutputHead.new(config)
        end

        def call(x, mask: nil, cache: nil)
          local_mask = mask || _create_attention_mask(x, cache)
          y = transformer.call(x, mask: local_mask, cache: cache)
          lm_head.call(y)
        end

        def sanitize(weights)
          first_key = "transformer.h.0.moe.mlp.0.fc1.weight"
          return weights unless weights.key?(first_key)

          mx = MLX::Core
          result = weights.dup

          @args.num_layers.times do |layer_idx|
            prefix = "transformer.h.#{layer_idx}"
            %w[fc1 fc2].each do |proj|
              %w[weight scales biases bias].each do |suffix|
                expert_keys = (0...@args.num_local_experts).map do |expert_idx|
                  "#{prefix}.moe.mlp.#{expert_idx}.#{proj}.#{suffix}"
                end
                next unless expert_keys.all? { |k| result.key?(k) }

                stacked = expert_keys.map { |k| result.delete(k) }
                result["#{prefix}.moe.switch_mlp.#{proj}.#{suffix}"] = mx.stack(stacked)
              end
            end
          end

          result
        end

        def layers
          transformer.h
        end

        private

        def _create_attention_mask(tokens, cache)
          first_cache = cache.is_a?(Array) ? cache[0] : cache
          return first_cache.make_mask(tokens.shape[1]) if first_cache && first_cache.respond_to?(:make_mask)
          return nil if tokens.shape[1] == 1

          "causal"
        end
      end

      Models.register("phixtral", Model, ModelArgs)
    end
  end
end
