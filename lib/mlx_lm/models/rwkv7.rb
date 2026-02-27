require_relative "recurrent_gemma"

module MlxLm
  module Models
    module Rwkv7
      class ModelArgs < BaseModelArgs
        field :model_type, default: "rwkv7"
        field :vocab_size
        field :hidden_size
        field :intermediate_size
        field :norm_eps, default: 1e-5
        field :head_dim
        field :num_hidden_layers
        field :a_low_rank_dim, default: nil
        field :v_low_rank_dim, default: nil
        field :gate_low_rank_dim, default: nil
        field :decay_low_rank_dim, default: nil
        field :tie_word_embeddings, default: false
        field :rope_theta, default: 10_000.0
        field :attention_window_size, default: 128
        field :block_types, default: nil
        field :num_attention_heads, default: nil
        field :num_key_value_heads, default: nil

        def initialize(**kwargs)
          super
          if @num_attention_heads.nil? && !@hidden_size.nil? && !@head_dim.nil? && @head_dim.to_i > 0
            @num_attention_heads = @hidden_size / @head_dim
          end
          @num_attention_heads ||= 1
          @num_key_value_heads ||= @num_attention_heads
          @block_types ||= Array.new(@num_hidden_layers.to_i, "recurrent")
        end

        def to_recurrent_gemma_dict
          {
            "model_type" => @model_type,
            "attention_bias" => false,
            "conv1d_width" => 3,
            "hidden_size" => @hidden_size,
            "intermediate_size" => @intermediate_size,
            "logits_soft_cap" => nil,
            "num_attention_heads" => @num_attention_heads,
            "num_hidden_layers" => @num_hidden_layers,
            "num_key_value_heads" => @num_key_value_heads,
            "rms_norm_eps" => @norm_eps,
            "rope_theta" => @rope_theta,
            "attention_window_size" => @attention_window_size,
            "vocab_size" => @vocab_size,
            "embeddings_scale_by_sqrt_dim" => false,
            "block_types" => @block_types,
          }
        end
      end

      class Model < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.model_type = args.model_type
          self.wrapped_model = RecurrentGemma::Model.new(
            RecurrentGemma::ModelArgs.from_dict(args.to_recurrent_gemma_dict)
          )
        end

        def call(inputs, cache: nil)
          wrapped_model.call(inputs, cache: cache)
        end

        def sanitize(weights)
          remapped = {}
          weights.each do |key, value|
            remapped[_remap_weight_key(key)] = value
          end
          wrapped_model.sanitize(remapped)
        end

        def layers
          wrapped_model.layers
        end

        def make_cache
          return wrapped_model.make_cache if wrapped_model.respond_to?(:make_cache)

          nil
        end

        private

        def _remap_weight_key(key)
          mapped = key.dup
          mapped = mapped.gsub(/\Ablocks\./, "model.layers.")
          mapped = mapped.gsub(".time_mix.", ".temporal_block.")
          mapped
        end
      end

      Models.register("rwkv7", Model, ModelArgs)
    end
  end
end
