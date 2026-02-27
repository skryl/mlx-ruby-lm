require_relative "recurrent_gemma"

module MlxLm
  module Models
    module FalconH1
      class ModelArgs < BaseModelArgs
        field :model_type, default: "falcon_h1"
        field :attention_bias, default: false
        field :head_dim, default: 64
        field :hidden_size, default: 1024
        field :intermediate_size, default: 2048
        field :max_position_embeddings, default: 131_072
        field :mamba_d_conv, default: 4
        field :num_attention_heads, default: 8
        field :num_hidden_layers, default: 36
        field :num_key_value_heads, default: 2
        field :rms_norm_eps, default: 1e-5
        field :rope_theta, default: 100_000_000_000.0
        field :vocab_size, default: 32_784
        field :tie_word_embeddings, default: true
        field :logits_soft_cap, default: nil
        field :attention_window_size, default: nil
        field :block_types, default: nil
      end

      class Model < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.model_type = args.model_type
          self.language_model = RecurrentGemma::Model.new(
            RecurrentGemma::ModelArgs.from_dict(_to_recurrent_gemma_config(args))
          )

          if args.tie_word_embeddings
            language_model.instance_variable_set(:@tie_word_embeddings, true)
            language_model.lm_head = nil if language_model.respond_to?(:lm_head=)
          end
        end

        def call(inputs, cache: nil)
          language_model.call(inputs, cache: cache)
        end

        def sanitize(weights)
          remapped = {}
          weights.each do |key, value|
            remapped[_remap_weight_key(key)] = value
          end
          language_model.sanitize(remapped)
        end

        def layers
          language_model.layers
        end

        def make_cache
          return language_model.make_cache if language_model.respond_to?(:make_cache)

          nil
        end

        private

        def _to_recurrent_gemma_config(args)
          {
            "model_type" => args.model_type,
            "attention_bias" => args.attention_bias,
            "conv1d_width" => args.mamba_d_conv || 4,
            "hidden_size" => args.hidden_size,
            "intermediate_size" => args.intermediate_size,
            "logits_soft_cap" => args.logits_soft_cap,
            "num_attention_heads" => args.num_attention_heads,
            "num_hidden_layers" => args.num_hidden_layers,
            "num_key_value_heads" => args.num_key_value_heads || args.num_attention_heads,
            "rms_norm_eps" => args.rms_norm_eps,
            "rope_theta" => args.rope_theta,
            "attention_window_size" => args.attention_window_size || [args.max_position_embeddings.to_i, 128].min,
            "vocab_size" => args.vocab_size,
            "embeddings_scale_by_sqrt_dim" => false,
            "block_types" => args.block_types || ["recurrent", "attention"],
          }
        end

        def _remap_weight_key(key)
          mapped = key.dup
          mapped = mapped.gsub(".mamba.conv1d.", ".temporal_block.conv_1d.")
          mapped = mapped.gsub(".mamba.out_proj.", ".temporal_block.linear_out.")
          mapped = mapped.gsub(".mamba.in_proj.", ".temporal_block.linear_x.")
          mapped = mapped.gsub(".self_attn.", ".temporal_block.")
          mapped = mapped.gsub(".feed_forward.", ".mlp_block.")
          mapped = mapped.gsub(".input_layernorm.", ".temporal_pre_norm.")
          mapped = mapped.gsub(".pre_ff_layernorm.", ".channel_pre_norm.")
          mapped = mapped.gsub("model.final_layernorm.", "model.final_norm.")
          mapped
        end
      end

      Models.register("falcon_h1", Model, ModelArgs)
    end
  end
end
