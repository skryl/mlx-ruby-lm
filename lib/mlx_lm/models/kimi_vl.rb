require_relative "deepseek"

module MlxLm
  module Models
    module KimiVL
      class ModelArgs < BaseModelArgs
        field :model_type, default: "kimi_vl"
        field :text_config, default: nil

        def self.from_dict(params)
          has_text_config = params.key?("text_config") || params.key?(:text_config)
          return super if has_text_config

          model_type = params["model_type"] || params[:model_type] || "kimi_vl"
          new(model_type: model_type, text_config: params)
        end

        def initialize(**kwargs)
          super
          @text_config = _stringify_keys(@text_config || {})
          @text_config["model_type"] ||= "deepseek"
        end

        private

        def _stringify_keys(hash)
          hash.each_with_object({}) do |(key, value), out|
            out[key.to_s] = value
          end
        end
      end

      class Model < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.model_type = args.model_type
          self.language_model = DeepSeek::Model.new(
            DeepSeek::ModelArgs.from_dict(args.text_config)
          )
        end

        def call(inputs, cache: nil, input_embeddings: nil)
          language_model.call(inputs, cache: cache)
        end

        def sanitize(weights)
          language_weights = {}
          flat_weights = weights.is_a?(Hash) ? weights : weights.to_h

          flat_weights.each do |key, value|
            next if _drop_key?(key)

            normalized_key = key.start_with?("language_model.") ? key.delete_prefix("language_model.") : key
            language_weights[normalized_key] = value
          end

          sanitized_language = if language_model.respond_to?(:sanitize)
            language_model.sanitize(language_weights)
          else
            language_weights
          end

          sanitized_language.each_with_object({}) do |(key, value), out|
            out["language_model.#{key}"] = value
          end
        end

        def model
          language_model.model
        end

        def layers
          model.layers
        end

        def cast_predicate
          lambda { |key| !key.include?("e_score_correction_bias") }
        end

        private

        def _drop_key?(key)
          key.include?("vision_tower") ||
            key.include?("multi_modal_projector") ||
            key.include?("rotary_emb")
        end
      end

      Models.register("kimi_vl", Model, ModelArgs)
    end
  end
end
