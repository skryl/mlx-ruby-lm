require_relative "qwen3"

module MlxLm
  module Models
    module Lfm2
      class ModelArgs < BaseModelArgs
        field :model_type, default: "lfm2"
        field :vocab_size, default: 32000
        field :hidden_size, default: 4096
        field :num_hidden_layers, default: 32
        field :num_attention_heads, default: 32
        field :num_key_value_heads, default: nil
        field :max_position_embeddings, default: 2048
        field :norm_eps, default: 1e-6
        field :conv_bias, default: false
        field :conv_L_cache, default: 4
        field :block_dim, default: nil
        field :block_ff_dim, default: nil
        field :block_multiple_of, default: 256
        field :block_ffn_dim_multiplier, default: nil
        field :block_auto_adjust_ff_dim, default: false
        field :rope_theta, default: 1_000_000.0
        field :rope_parameters, default: nil
        field :full_attn_idxs, default: nil
        field :layer_types, default: nil
        field :tie_word_embeddings, default: true

        def initialize(**kwargs)
          super
          rope_theta_from_params = _rope_theta_from_parameters
          @rope_theta = rope_theta_from_params unless rope_theta_from_params.nil?
          @num_key_value_heads ||= @num_attention_heads
          @block_dim ||= @hidden_size
          @block_ff_dim ||= @block_dim * 4
          @full_attn_idxs ||= _full_attn_idxs_from_layer_types
        end

        private

        def _rope_theta_from_parameters
          return nil unless @rope_parameters.is_a?(Hash)

          @rope_parameters["rope_theta"] || @rope_parameters[:rope_theta]
        end

        def _full_attn_idxs_from_layer_types
          return [] unless @layer_types.is_a?(Array)

          @layer_types.each_with_index.filter_map do |layer_type, i|
            i if layer_type.to_s == "full_attention"
          end
        end
      end

      class Model < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.model_type = args.model_type
          self.language_model = Qwen3::Model.new(Qwen3::ModelArgs.from_dict(_qwen3_config(args)))
        end

        def call(inputs, cache: nil, input_embeddings: nil)
          language_model.call(inputs, cache: cache, input_embeddings: input_embeddings)
        end

        def sanitize(weights)
          sanitized = {}
          weights.each do |name, param|
            current = param
            if name.include?("conv.weight") && _transpose_conv_weight?(param)
              current = MLX::Core.swapaxes(param, 1, 2)
            end
            sanitized[name] = current
          end
          sanitized
        end

        def layers
          language_model.layers
        end

        def make_cache
          return language_model.make_cache if language_model.respond_to?(:make_cache)
          return nil unless defined?(MlxLm::KVCache)

          Array.new(layers.length) { MlxLm::KVCache.new }
        end

        private

        def _transpose_conv_weight?(param)
          return false unless param.respond_to?(:shape)
          return false unless param.shape.is_a?(Array)
          return false unless param.shape.length >= 3

          param.shape[-1] > param.shape[1]
        end

        def _qwen3_config(args)
          {
            "model_type" => "qwen3",
            "hidden_size" => args.hidden_size,
            "num_hidden_layers" => args.num_hidden_layers,
            "intermediate_size" => args.block_ff_dim,
            "num_attention_heads" => args.num_attention_heads,
            "num_key_value_heads" => args.num_key_value_heads,
            "rms_norm_eps" => args.norm_eps,
            "vocab_size" => args.vocab_size,
            "rope_theta" => args.rope_theta,
            "max_position_embeddings" => args.max_position_embeddings,
            "tie_word_embeddings" => args.tie_word_embeddings,
          }
        end
      end

      Models.register("lfm2", Model, ModelArgs)
    end
  end
end
