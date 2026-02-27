require_relative "gemma3_text"

module MlxLm
  module Models
    module Gemma3
      class ModelArgs < BaseModelArgs
        field :model_type, default: "gemma3"
        field :text_config, default: nil
        field :vocab_size, default: 262208

        def self.from_dict(params)
          has_text_config = params.key?("text_config") || params.key?(:text_config)
          return super if has_text_config

          model_type = params["model_type"] || params[:model_type] || "gemma3"
          vocab_size = params["vocab_size"] || params[:vocab_size] || 262208
          new(model_type: model_type, text_config: params, vocab_size: vocab_size)
        end

        def initialize(**kwargs)
          super
          @text_config = _stringify_keys(@text_config || {})
          @text_config["vocab_size"] = @vocab_size
          @text_config["num_attention_heads"] ||= 8
          @text_config["num_key_value_heads"] ||= 4
          @text_config["model_type"] ||= "gemma3_text"
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
          self.language_model = Gemma3Text::Model.new(
            Gemma3Text::ModelArgs.from_dict(args.text_config)
          )
        end

        def call(inputs, cache: nil, input_embeddings: nil)
          language_model.call(
            inputs,
            cache: cache,
            input_embeddings: input_embeddings
          )
        end

        def sanitize(weights)
          flat_weights = weights.is_a?(Hash) ? weights : weights.to_h
          nested = MLX::Utils.tree_unflatten(flat_weights.to_a)

          if nested.is_a?(Hash)
            nested.delete("vision_tower")
            nested.delete("multi_modal_projector")

            language_tree = nested["language_model"] || {}
            language_weights = MLX::Utils.tree_flatten(language_tree, destination: {})
            sanitized_language = language_model.sanitize(language_weights)
            nested["language_model"] = MLX::Utils.tree_unflatten(sanitized_language.to_a)
          end

          MLX::Utils.tree_flatten(nested, destination: {})
        end

        def layers
          language_model.layers
        end

        def make_cache
          language_model.make_cache
        end
      end

      Models.register("gemma3", Model, ModelArgs)
    end
  end
end
