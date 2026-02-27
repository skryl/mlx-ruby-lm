require_relative "llama"

module MlxLm
  module Models
    module Mistral3
      class ModelArgs < BaseModelArgs
        field :model_type, default: "mistral3"
        field :text_config, default: nil

        def initialize(**kwargs)
          super
          @text_config = (@text_config || {}).dup
          @text_config["tie_word_embeddings"] = false unless @text_config.key?("tie_word_embeddings")
        end
      end

      class Model < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.model_type = args.model_type

          text_config = args.text_config || {}
          text_model_type = text_config["model_type"]

          if text_model_type == "ministral3" && Models::REGISTRY.key?("ministral3")
            model_class, args_class = Models.get_classes(text_config)
            self.language_model = model_class.new(args_class.from_dict(text_config))
          else
            self.language_model = Llama::Model.new(Llama::ModelArgs.from_dict(text_config))
          end
        end

        def call(inputs, cache: nil, input_embeddings: nil)
          supports_input_embeddings = language_model.method(:call).parameters.any? do |_, name|
            name == :input_embeddings
          end

          if supports_input_embeddings
            language_model.call(inputs, cache: cache, input_embeddings: input_embeddings)
          else
            language_model.call(inputs, cache: cache)
          end
        end

        def sanitize(weights)
          result = {}
          language_weights = {}

          weights.each do |k, v|
            next if k == "vision_tower" || k.start_with?("vision_tower.")
            next if k == "multi_modal_projector" || k.start_with?("multi_modal_projector.")

            if k.start_with?("language_model.")
              language_weights[k.delete_prefix("language_model.")] = v
            else
              result[k] = v
            end
          end

          sanitized_language = if language_model.respond_to?(:sanitize)
            language_model.sanitize(language_weights)
          else
            language_weights
          end

          sanitized_language.each do |k, v|
            result["language_model.#{k}"] = v
          end

          result
        end

        def layers
          return language_model.model.layers if language_model.respond_to?(:model) && language_model.model.respond_to?(:layers)

          language_model.layers
        end
      end

      Models.register("mistral3", Model, ModelArgs)
    end
  end
end
