require_relative "activations"
require_relative "qwen2"
require_relative "switch_layers"

module MlxLm
  module Models
    module Qwen2Moe
      class ModelArgs < Qwen2::ModelArgs
        field :model_type, default: "qwen2_moe"
        field :num_key_value_heads, default: nil
        field :num_experts_per_tok
        field :num_experts
        field :moe_intermediate_size
        field :shared_expert_intermediate_size
        field :tie_word_embeddings, default: false

        def initialize(**kwargs)
          super
          validate_rope_scaling!
        end

        private

        def validate_rope_scaling!
          return unless @rope_scaling

          required_keys = %w[factor type]
          unless required_keys.all? { |key| _rope_scaling_has_key?(key) }
            raise ArgumentError, "rope_scaling must contain keys #{required_keys}"
          end

          return if _rope_scaling_value("type") == "linear"

          raise ArgumentError, "rope_scaling 'type' currently only supports 'linear'"
        end

        def _rope_scaling_has_key?(key)
          @rope_scaling.key?(key) || @rope_scaling.key?(key.to_sym)
        end

        def _rope_scaling_value(key)
          return @rope_scaling[key] if @rope_scaling.key?(key)

          @rope_scaling[key.to_sym]
        end
      end

      class SharedExpertMLP < MLX::NN::Module
        def initialize(dim, hidden_dim)
          super()
          self.gate_proj = MLX::NN::Linear.new(dim, hidden_dim, bias: false)
          self.down_proj = MLX::NN::Linear.new(hidden_dim, dim, bias: false)
          self.up_proj = MLX::NN::Linear.new(dim, hidden_dim, bias: false)
        end

        def call(x)
          down_proj.call(Activations.swiglu(gate_proj.call(x), up_proj.call(x)))
        end
      end

      class SparseMoeBlock < MLX::NN::Module
        def initialize(args)
          super()
          dim = args.hidden_size
          intermediate_size = args.moe_intermediate_size
          shared_expert_intermediate_size = args.shared_expert_intermediate_size

          @num_experts = args.num_experts
          @top_k = args.num_experts_per_tok

          self.gate = MLX::NN::Linear.new(dim, @num_experts, bias: false)
          self.switch_mlp = SwitchLayers::SwitchGLU.new(dim, intermediate_size, @num_experts)

          self.shared_expert = SharedExpertMLP.new(dim, shared_expert_intermediate_size)
          self.shared_expert_gate = MLX::NN::Linear.new(dim, 1, bias: false)
        end

        def call(x)
          mx = MLX::Core

          gates = gate.call(x)
          gates = mx.softmax(gates.astype(mx.float32), -1).astype(gates.dtype)

          k = [@top_k, @num_experts].min
          inds = mx.stop_gradient(mx.argpartition(gates * -1.0, k - 1, -1))
          take_ids = mx.array((0...k).to_a, dtype: mx.int32)
          inds = mx.take(inds, take_ids, -1)
          scores = mx.take_along_axis(gates, inds, -1)

          y = switch_mlp.call(x, inds)
          y = mx.sum(y * mx.expand_dims(scores, -1), -2)

          shared_expert_output = shared_expert.call(x)
          shared_expert_output = mx.sigmoid(shared_expert_gate.call(x)) * shared_expert_output

          y + shared_expert_output
        end
      end

      class DecoderLayer < MLX::NN::Module
        def initialize(args)
          super()
          self.self_attn = Qwen2::Attention.new(args)
          self.mlp = SparseMoeBlock.new(args)
          self.input_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
          self.post_attention_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
        end

        def call(x, mask: nil, cache: nil)
          r = self_attn.call(input_layernorm.call(x), mask: mask, cache: cache)
          h = x + r
          r = mlp.call(post_attention_layernorm.call(h))
          h + r
        end
      end

      class Qwen2MoeModel < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.embed_tokens = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
          self.layers = Array.new(args.num_hidden_layers) { DecoderLayer.new(args) }
          self.norm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
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
          self.model = Qwen2MoeModel.new(args)
          self.lm_head = MLX::NN::Linear.new(args.hidden_size, args.vocab_size, bias: false)
        end

        def call(inputs, cache: nil)
          lm_head.call(model.call(inputs, cache: cache))
        end

        def sanitize(weights)
          return weights unless weights.key?("model.layers.0.mlp.experts.0.up_proj.weight")

          mx = MLX::Core
          result = weights.dup

          @args.num_hidden_layers.times do |layer_idx|
            prefix = "model.layers.#{layer_idx}"
            %w[up_proj down_proj gate_proj].each do |projection|
              %w[weight scales biases].each do |param|
                first_key = "#{prefix}.mlp.experts.0.#{projection}.#{param}"
                next unless result.key?(first_key)

                expert_keys = (0...@args.num_experts).map do |expert_idx|
                  "#{prefix}.mlp.experts.#{expert_idx}.#{projection}.#{param}"
                end
                next unless expert_keys.all? { |key| result.key?(key) }

                stacked = expert_keys.map { |key| result.delete(key) }
                result["#{prefix}.mlp.switch_mlp.#{projection}.#{param}"] = mx.stack(stacked)
              end
            end
          end

          result
        end

        def layers
          model.layers
        end
      end

      Models.register("qwen2_moe", Model, ModelArgs)
    end
  end
end
