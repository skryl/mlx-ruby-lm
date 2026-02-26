module MlxLm
  module Models
    module Qwen35
      class ModelArgs < BaseModelArgs
        field :model_type, default: "qwen3_5"
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
          self.language_model = Qwen3::Model.new(Qwen3::ModelArgs.from_dict(_text_config_for_qwen3(args)))
        end

        def call(inputs, cache: nil, input_embeddings: nil)
          language_model.call(inputs, cache: cache, input_embeddings: input_embeddings)
        end

        def sanitize(weights)
          language_model.sanitize(remap_language_model_weights(weights))
        end

        def layers
          language_model.layers
        end

        protected

        def remap_language_model_weights(weights)
          remapped = {}
          weights.each do |key, value|
            next if key.start_with?("model.visual")

            mapped_key = if key.start_with?("model.language_model")
              key.sub("model.language_model", "language_model.model")
            elsif key.start_with?("language_model.")
              key
            else
              "language_model.#{key}"
            end
            remapped[mapped_key] = value
          end
          remapped
        end

        private

        def _text_config_for_qwen3(args)
          config = {}
          (args.text_config || {}).each { |key, value| config[key.to_s] = value }
          config["model_type"] ||= args.model_type
          config["tie_word_embeddings"] = false unless config.key?("tie_word_embeddings")
          config
        end
      end

      Models.register("qwen3_5", Model, ModelArgs)
    end
  end
end
