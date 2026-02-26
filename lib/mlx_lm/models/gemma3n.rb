require_relative "gemma2"

module MlxLm
  module Models
    module Gemma3n
      class ModelArgs < BaseModelArgs
        field :model_type, default: "gemma3n"
        field :text_config, default: nil

        def self.from_dict(params)
          has_text_config = params.key?("text_config") || params.key?(:text_config)
          return super if has_text_config

          new(model_type: params["model_type"] || params[:model_type], text_config: params)
        end

        def initialize(**kwargs)
          super
          @text_config = (@text_config || {}).dup
        end
      end

      class Model < MLX::NN::Module
        MULTIMODAL_MODEL_PREFIXES = %w[
          model.vision_tower
          model.audio_tower
          model.embed_audio
          model.embed_vision
        ].freeze

        def initialize(args)
          super()
          @args = args
          self.model_type = args.model_type
          self.language_model = Gemma2::Model.new(Gemma2::ModelArgs.from_dict(_text_config_for_gemma2(args)))
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
          weights.reject do |key, _|
            MULTIMODAL_MODEL_PREFIXES.any? { |prefix| key == prefix || key.start_with?("#{prefix}.") }
          end
        end

        def layers
          language_model.layers
        end

        def make_cache
          return language_model.make_cache if language_model.respond_to?(:make_cache)

          nil
        end

        private

        def _text_config_for_gemma2(args)
          config = {}
          (args.text_config || {}).each { |key, value| config[key.to_s] = value }
          config["model_type"] ||= args.model_type
          config
        end
      end

      Models.register("gemma3n", Model, ModelArgs)
    end
  end
end
