module MlxLm
  module Models
    module Qwen3VL
      class ModelArgs < BaseModelArgs
        field :model_type, default: "qwen3_vl"
        field :text_config, default: nil

        def self.from_dict(params)
          return super if params.key?("text_config")

          new(model_type: params["model_type"], text_config: params)
        end
      end

      class Model < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.model_type = args.model_type
          self.language_model = Qwen3::Model.new(Qwen3::ModelArgs.from_dict(args.text_config))
        end

        def call(inputs, cache: nil, input_embeddings: nil)
          language_model.call(inputs, cache: cache, input_embeddings: input_embeddings)
        end

        def sanitize(weights)
          nested = MLX::Utils.tree_unflatten(weights.to_a)
          nested.delete("vision_tower") if nested.is_a?(Hash)

          flattened = MLX::Utils.tree_flatten(nested, destination: {})
          sanitized = {}
          flattened.each do |key, value|
            sanitized_key = key.start_with?("language_model.") ? key : "language_model.#{key}"
            sanitized[sanitized_key] = value
          end
          sanitized
        end

        def layers
          language_model.layers
        end
      end

      Models.register("qwen3_vl", Model, ModelArgs)
    end
  end
end
