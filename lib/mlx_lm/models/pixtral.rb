module MlxLm
  module Models
    module Pixtral
      class ModelArgs < BaseModelArgs
        field :model_type, default: "pixtral"
        field :text_config

        def initialize(**kwargs)
          super
          @text_config ||= {}
          @text_config["tie_word_embeddings"] = false
          unless @text_config.key?("num_attention_heads") || @text_config.key?(:num_attention_heads)
            @text_config["num_attention_heads"] = 32
          end
        end
      end

      class Model < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.model_type = args.model_type
          self.language_model = Llama::Model.new(Llama::ModelArgs.from_dict(args.text_config))
        end

        def call(inputs, cache: nil, input_embeddings: nil)
          language_model.call(inputs, cache: cache)
        end

        def sanitize(weights)
          weights.reject do |key, _|
            key == "vision_tower" ||
              key.start_with?("vision_tower.") ||
              key == "multi_modal_projector" ||
              key.start_with?("multi_modal_projector.")
          end
        end

        def layers
          language_model.model.layers
        end
      end

      Models.register("pixtral", Model, ModelArgs)
    end
  end
end
