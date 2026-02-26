require_relative "activations"
require_relative "rope_utils"
require_relative "ernie4_5"

module MlxLm
  module Models
    module Ernie45Moe
      class ModelArgs < Ernie45::ModelArgs
        field :model_type, default: "ernie4_5_moe"
        field :moe_num_experts, default: 0
        field :moe_layer_start_index, default: 0
        field :moe_intermediate_size, default: 0
        field :moe_capacity, default: []
        field :moe_k, default: 1
        field :moe_layer_interval, default: 1
        field :moe_use_aux_free, default: false
        field :moe_num_shared_experts, default: 0
        field :moe_layer_end_index, default: nil
        field :moe_gate_act, default: "softmax"

        def initialize(**kwargs)
          super
          @moe_capacity = Array(@moe_capacity).dup
        end
      end

      class Model < Ernie45::Model
        REMOVE_PATTERNS = [
          "mtp_block.",
          "mtp_linear_proj.",
          "mtp_hidden_norm.",
          "mtp_emb_norm.",
          "e_score_correction_bias",
        ].freeze

        EXPERT_PROJ_NAMES = %w[gate_proj down_proj up_proj].freeze

        def sanitize(weights)
          result = weights.reject do |key, _|
            REMOVE_PATTERNS.any? { |pattern| key.include?(pattern) }
          end

          stack_expert_weights!(result)
        end

        private

        def stack_expert_weights!(weights)
          mx = MLX::Core
          num_experts = @args.moe_num_experts.to_i
          return weights if num_experts <= 0

          @args.num_hidden_layers.times do |layer_idx|
            prefix = "model.layers.#{layer_idx}.mlp"

            EXPERT_PROJ_NAMES.each do |proj_name|
              expert_weights = pop_complete_expert_weights(weights, prefix, proj_name, num_experts)
              next unless expert_weights

              weights["#{prefix}.switch_mlp.#{proj_name}.weight"] = mx.stack(expert_weights)
            end
          end

          weights
        end

        def pop_complete_expert_weights(weights, prefix, proj_name, num_experts)
          first_key = expert_weight_key(prefix, 0, proj_name)
          return nil unless weights.key?(first_key)

          popped = []
          num_experts.times do |expert_idx|
            key = expert_weight_key(prefix, expert_idx, proj_name)
            unless weights.key?(key)
              restore_popped_weights!(weights, prefix, proj_name, popped)
              return nil
            end
            popped << weights.delete(key)
          end
          popped
        end

        def restore_popped_weights!(weights, prefix, proj_name, popped)
          popped.each_with_index do |tensor, idx|
            weights[expert_weight_key(prefix, idx, proj_name)] = tensor
          end
        end

        def expert_weight_key(prefix, expert_idx, proj_name)
          "#{prefix}.experts.#{expert_idx}.#{proj_name}.weight"
        end
      end

      Models.register("ernie4_5_moe", Model, ModelArgs)
    end
  end
end
