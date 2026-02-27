module MlxLm
  module Models
    module Qwen2VL
      class ModelArgs < BaseModelArgs
        field :model_type, default: "qwen2_vl"
        field :text_config

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
          self.language_model = Qwen2::Model.new(Qwen2::ModelArgs.from_dict(args.text_config))
        end

        def call(inputs, cache: nil, input_embeddings: nil)
          language_model.call(inputs, cache: cache)
        end

        def sanitize(weights)
          sanitized = {}
          weights.each do |key, value|
            next if key == "visual" || key.start_with?("visual.")
            next if key == "vision_tower" || key.start_with?("vision_tower.")

            mapped_key = key.start_with?("language_model.") ? key : "language_model.#{key}"
            sanitized[mapped_key] = value
          end
          sanitized
        end

        def layers
          language_model.model.layers
        end
      end

      Models.register("qwen2_vl", Model, ModelArgs)
    end
  end
end
