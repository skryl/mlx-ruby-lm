module MlxLm
  module Models
    module Qwen35Moe
      class ModelArgs < Qwen35::ModelArgs
        field :model_type, default: "qwen3_5_moe"
      end

      class Model < Qwen35::Model
        def sanitize(weights)
          remapped = remap_language_model_weights(weights)
          rewrite_moe_expert_weights(remapped)
          language_model.sanitize(remapped)
        end

        private

        def rewrite_moe_expert_weights(weights)
          mx = MLX::Core

          layers.length.times do |layer_idx|
            prefix = "language_model.model.layers.#{layer_idx}.mlp"
            gate_up_key = _first_existing_key(
              weights,
              ["#{prefix}.experts.gate_up_proj", "#{prefix}.experts.gate_up_proj.weight"]
            )
            down_proj_key = _first_existing_key(
              weights,
              ["#{prefix}.experts.down_proj", "#{prefix}.experts.down_proj.weight"]
            )

            next unless gate_up_key && down_proj_key

            gate_up = weights.delete(gate_up_key)
            down_proj = weights.delete(down_proj_key)
            mid = gate_up.shape[-2] / 2
            gate_proj, up_proj = mx.split(gate_up, [mid], -2)

            weights["#{prefix}.switch_mlp.gate_proj.weight"] = gate_proj
            weights["#{prefix}.switch_mlp.up_proj.weight"] = up_proj
            weights["#{prefix}.switch_mlp.down_proj.weight"] = down_proj
          end

          weights
        end

        def _first_existing_key(weights, candidates)
          candidates.find { |key| weights.key?(key) }
        end
      end

      Models.register("qwen3_5_moe", Model, ModelArgs)
    end
  end
end
