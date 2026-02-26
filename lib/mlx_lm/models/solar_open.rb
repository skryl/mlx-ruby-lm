require_relative "deepseek"

module MlxLm
  module Models
    module SolarOpen
      class ModelArgs < BaseModelArgs
        field :model_type, default: "solar_open"
        field :vocab_size
        field :hidden_size
        field :intermediate_size
        field :moe_intermediate_size
        field :num_hidden_layers
        field :num_attention_heads
        field :num_key_value_heads
        field :head_dim
        field :n_shared_experts
        field :n_routed_experts
        field :routed_scaling_factor
        field :num_experts_per_tok
        field :first_k_dense_replace
        field :norm_topk_prob
        field :max_position_embeddings
        field :rms_norm_eps
        field :rope_theta
        field :tie_word_embeddings
        field :partial_rotary_factor
        field :rope_scaling, default: nil
        field :attention_bias, default: false
        field :use_qk_norm, default: false
        field :n_group, default: 1
        field :topk_group, default: 1
        field :scoring_func, default: "sigmoid"
        field :topk_method, default: "noaux_tc"
      end

      class Model < DeepSeek::Model
        def initialize(args)
          super(DeepSeek::ModelArgs.from_dict(_to_deepseek_config(args)))
          self.model_type = args.model_type
        end

        def sanitize(weights)
          sanitized = super(weights)
          mpt_prefix = "model.layers.#{@args.num_hidden_layers}"
          sanitized.reject do |k, _|
            k == mpt_prefix || k.start_with?("#{mpt_prefix}.")
          end
        end

        private

        def _to_deepseek_config(args)
          {
            "model_type" => args.model_type,
            "vocab_size" => args.vocab_size,
            "hidden_size" => args.hidden_size,
            "intermediate_size" => args.intermediate_size,
            "moe_intermediate_size" => args.moe_intermediate_size,
            "num_hidden_layers" => args.num_hidden_layers,
            "num_attention_heads" => args.num_attention_heads,
            "num_key_value_heads" => args.num_key_value_heads,
            "n_shared_experts" => args.n_shared_experts,
            "n_routed_experts" => args.n_routed_experts,
            "num_experts_per_tok" => args.num_experts_per_tok,
            "first_k_dense_replace" => args.first_k_dense_replace,
            "moe_layer_freq" => 1,
            "max_position_embeddings" => args.max_position_embeddings,
            "rms_norm_eps" => args.rms_norm_eps,
            "rope_theta" => args.rope_theta,
            "rope_scaling" => args.rope_scaling,
            "attention_bias" => args.attention_bias,
          }
        end
      end

      Models.register("solar_open", Model, ModelArgs)
    end
  end
end
