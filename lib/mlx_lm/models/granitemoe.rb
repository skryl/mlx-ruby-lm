require_relative "granite"

module MlxLm
  module Models
    module GraniteMoe
      class ModelArgs < Granite::ModelArgs
        field :model_type, default: "granitemoe"
        field :num_local_experts
        field :num_experts_per_tok
      end

      class Model < Granite::Model
        def sanitize(weights)
          result = weights.dup
          rewrite_legacy_moe_weights(result)
          result.delete("lm_head.weight") if @args.tie_word_embeddings
          result
        end

        private

        def rewrite_legacy_moe_weights(weights)
          mx = MLX::Core

          layers.length.times do |layer_idx|
            prefix = "model.layers.#{layer_idx}.block_sparse_moe"
            input_key = _first_existing_key(
              weights,
              ["#{prefix}.input_linear.weight", "#{prefix}.input_linear"]
            )
            output_key = _first_existing_key(
              weights,
              ["#{prefix}.output_linear.weight", "#{prefix}.output_linear"]
            )
            next unless input_key && output_key

            input_linear = weights.delete(input_key)
            output_linear = weights.delete(output_key)
            mid = input_linear.shape[1] / 2
            gate_proj, up_proj = mx.split(input_linear, [mid], 1)

            weights["#{prefix}.switch_mlp.gate_proj.weight"] = gate_proj
            weights["#{prefix}.switch_mlp.up_proj.weight"] = up_proj
            weights["#{prefix}.switch_mlp.down_proj.weight"] = output_linear
          end

          weights
        end

        def _first_existing_key(weights, candidates)
          candidates.find { |key| weights.key?(key) }
        end
      end

      Models.register("granitemoe", Model, ModelArgs)
    end
  end
end
