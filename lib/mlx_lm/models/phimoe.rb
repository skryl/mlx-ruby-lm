require_relative "rope_utils"
require_relative "switch_layers"

module MlxLm
  module Models
    module PhiMoe
      class ModelArgs < BaseModelArgs
        field :model_type, default: "phimoe"
        field :vocab_size, default: 32064
        field :hidden_size, default: 4096
        field :intermediate_size, default: 6400
        field :num_hidden_layers, default: 32
        field :num_attention_heads, default: 32
        field :num_key_value_heads, default: 8
        field :max_position_embeddings, default: 131072
        field :original_max_position_embeddings, default: 4096
        field :rms_norm_eps, default: 1e-6
        field :rope_scaling, default: nil
        field :num_local_experts, default: 16
        field :num_experts_per_tok, default: 2
        field :rope_theta, default: 10_000.0
      end

      class Attention < MLX::NN::Module
        def initialize(args)
          super()

          dim = args.hidden_size
          @n_heads = args.num_attention_heads
          @n_kv_heads = args.num_key_value_heads
          @head_dim = dim / @n_heads
          @scale = @head_dim**(-0.5)

          self.q_proj = MLX::NN::Linear.new(dim, @n_heads * @head_dim, bias: true)
          self.k_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: true)
          self.v_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: true)
          self.o_proj = MLX::NN::Linear.new(@n_heads * @head_dim, dim, bias: true)

          scaling = args.rope_scaling || {}
          self.rope = SuScaledRoPE.new(
            @head_dim,
            base: args.rope_theta,
            max_position_embeddings: args.max_position_embeddings,
            original_max_position_embeddings: args.original_max_position_embeddings,
            short_factor: _config_value(scaling, "short_factor", 1.0),
            long_factor: _config_value(scaling, "long_factor", 1.0),
            short_mscale: _config_value(scaling, "short_mscale"),
            long_mscale: _config_value(scaling, "long_mscale")
          )
        end

        def call(x, mask: nil, cache: nil)
          mx = MLX::Core
          b, l, _d = x.shape

          queries = q_proj.call(x).reshape([b, l, @n_heads, @head_dim]).transpose([0, 2, 1, 3])
          keys = k_proj.call(x).reshape([b, l, @n_kv_heads, @head_dim]).transpose([0, 2, 1, 3])
          values = v_proj.call(x).reshape([b, l, @n_kv_heads, @head_dim]).transpose([0, 2, 1, 3])

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

        private

        def _config_value(config, key, default = nil)
          return default if config.nil?
          return config[key] if config.key?(key)

          config.fetch(key.to_sym, default)
        end
      end

      class PhiMoESparseMoeBlock < MLX::NN::Module
        def initialize(args)
          super()

          @hidden_dim = args.hidden_size
          @ffn_dim = args.intermediate_size
          @num_experts = args.num_local_experts
          @top_k = args.num_experts_per_tok

          self.gate = MLX::NN::Linear.new(@hidden_dim, @num_experts, bias: false)
          self.switch_mlp = SwitchLayers::SwitchGLU.new(@hidden_dim, @ffn_dim, @num_experts)
        end

        def call(x)
          mx = MLX::Core

          k = [@top_k, @num_experts].min
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

      class DecoderLayer < MLX::NN::Module
        def initialize(args)
          super()
          self.self_attn = Attention.new(args)
          self.block_sparse_moe = PhiMoESparseMoeBlock.new(args)
          self.input_layernorm = MLX::NN::LayerNorm.new(args.hidden_size, eps: args.rms_norm_eps)
          self.post_attention_layernorm = MLX::NN::LayerNorm.new(args.hidden_size, eps: args.rms_norm_eps)
        end

        def call(x, mask: nil, cache: nil)
          residual = x
          hidden_states = input_layernorm.call(x)
          hidden_states = self_attn.call(hidden_states, mask: mask, cache: cache)
          hidden_states = residual + hidden_states

          residual = hidden_states
          hidden_states = post_attention_layernorm.call(hidden_states)
          hidden_states = block_sparse_moe.call(hidden_states)
          residual + hidden_states
        end
      end

      class PhiMoEModel < MLX::NN::Module
        def initialize(args)
          super()
          self.embed_tokens = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
          self.layers = Array.new(args.num_hidden_layers) { DecoderLayer.new(args) }
          self.norm = MLX::NN::LayerNorm.new(args.hidden_size, eps: args.rms_norm_eps)
        end

        def call(inputs, cache: nil)
          h = embed_tokens.call(inputs)
          layer_cache = cache || [nil] * layers.length

          mask = nil
          mask = "causal" if h.shape[1] > 1

          layers.each_with_index do |layer, layer_idx|
            h = layer.call(h, mask: mask, cache: layer_cache[layer_idx])
          end

          norm.call(h)
        end
      end

      class Model < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.model_type = args.model_type
          self.model = PhiMoEModel.new(args)
          self.lm_head = MLX::NN::Linear.new(args.hidden_size, args.vocab_size, bias: true)
        end

        def call(inputs, cache: nil)
          lm_head.call(model.call(inputs, cache: cache))
        end

        def sanitize(weights)
          return weights unless weights.key?("model.layers.0.block_sparse_moe.experts.0.w1.weight")

          mx = MLX::Core
          result = weights.dup

          @args.num_hidden_layers.times do |layer_idx|
            prefix = "model.layers.#{layer_idx}"
            [["w1", "gate_proj"], ["w2", "down_proj"], ["w3", "up_proj"]].each do |source, target|
              %w[weight scales biases].each do |param|
                first_key = "#{prefix}.block_sparse_moe.experts.0.#{source}.#{param}"
                next unless result.key?(first_key)

                expert_keys = (0...@args.num_local_experts).map do |expert_idx|
                  "#{prefix}.block_sparse_moe.experts.#{expert_idx}.#{source}.#{param}"
                end
                next unless expert_keys.all? { |key| result.key?(key) }

                stacked = expert_keys.map { |key| result.delete(key) }
                result["#{prefix}.block_sparse_moe.switch_mlp.#{target}.#{param}"] = mx.stack(stacked)
              end
            end
          end

          result
        end

        def layers
          model.layers
        end
      end

      Models.register("phimoe", Model, ModelArgs)
    end
  end
end
