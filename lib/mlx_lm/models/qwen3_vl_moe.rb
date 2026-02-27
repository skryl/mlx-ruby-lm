require_relative "qwen3_moe"

module MlxLm
  module Models
    module Qwen3VLMoe
      class ModelArgs < BaseModelArgs
        field :model_type, default: "qwen3_vl_moe"
        field :text_config, default: nil

        def self.from_dict(params)
          has_text_config = params.key?("text_config") || params.key?(:text_config)
          return super if has_text_config

          new(model_type: params["model_type"] || params[:model_type], text_config: params)
        end
      end

      class Model < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.model_type = args.model_type
          self.language_model = Qwen3Moe::Model.new(Qwen3Moe::ModelArgs.from_dict(args.text_config))
        end

        def call(inputs, cache: nil, input_embeddings: nil)
          language_model.call(inputs, cache: cache, input_embeddings: input_embeddings)
        end

        def sanitize(weights)
          nested = MLX::Utils.tree_unflatten(weights.to_a)
          nested.delete("visual") if nested.is_a?(Hash)

          language_model_tree = {}
          if nested.is_a?(Hash)
            language_model_node = nested["language_model"]
            if language_model_node.is_a?(Hash)
              language_model_tree["model"] = language_model_node["model"] if language_model_node.key?("model")
              language_model_tree["lm_head"] = language_model_node["lm_head"] if language_model_node.key?("lm_head")
            end
          end

          flattened = MLX::Utils.tree_flatten({ "language_model" => language_model_tree }, destination: {})
          sanitized = flattened.is_a?(Hash) ? flattened : {}
          rewrite_moe_expert_weights(sanitized)
          sanitized
        end

        def layers
          language_model.model.layers
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
            mid = gate_up.shape[-1] / 2
            gate_proj, up_proj = mx.split(gate_up, [mid], -1)

            weights["#{prefix}.switch_mlp.gate_proj.weight"] = mx.swapaxes(gate_proj, -2, -1)
            weights["#{prefix}.switch_mlp.up_proj.weight"] = mx.swapaxes(up_proj, -2, -1)
            weights["#{prefix}.switch_mlp.down_proj.weight"] = mx.swapaxes(down_proj, -2, -1)
          end

          weights
        end

        def _first_existing_key(weights, candidates)
          candidates.find { |key| weights.key?(key) }
        end
      end

      Models.register("qwen3_vl_moe", Model, ModelArgs)
    end
  end
end
