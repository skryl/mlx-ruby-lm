require_relative "lfm2"

module MlxLm
  module Models
    module Lfm2VL
      class ModelArgs < BaseModelArgs
        field :model_type, default: "lfm2-vl"
        field :text_config, default: nil

        def self.from_dict(params)
          has_text_config = params.key?("text_config") || params.key?(:text_config)
          return super if has_text_config

          new(model_type: params["model_type"] || params[:model_type], text_config: params)
        end

        def initialize(**kwargs)
          super
          @text_config = _stringify_keys(@text_config || {})
          @text_config["tie_word_embeddings"] = false
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
          self.language_model = Lfm2::Model.new(Lfm2::ModelArgs.from_dict(args.text_config))
        end

        def call(inputs, cache: nil, input_embeddings: nil)
          language_model.call(inputs, cache: cache, input_embeddings: input_embeddings)
        end

        def sanitize(weights)
          nested = MLX::Utils.tree_unflatten(weights.to_a)
          if nested.is_a?(Hash)
            nested.delete("vision_tower")
            nested.delete("multi_modal_projector")
          end
          MLX::Utils.tree_flatten(nested, destination: {})
        end

        def layers
          language_model.layers
        end

        def make_cache
          return language_model.make_cache if language_model.respond_to?(:make_cache)

          nil
        end
      end

      Models.register("lfm2-vl", Model, ModelArgs)
    end
  end
end
