require_relative "deepseek_v2"

module MlxLm
  module Models
    module DeepseekV3
      class ModelArgs < DeepseekV2::ModelArgs
        field :model_type, default: "deepseek_v3"
        field :topk_method, default: "noaux_tc"
        field :scoring_func, default: "sigmoid"
        field :norm_topk_prob, default: true
        field :n_group, default: 1
        field :topk_group, default: 1
        field :num_experts_per_tok, default: 1
      end

      class Model < DeepseekV2::Model
        def sanitize(weights)
          super(weights).reject do |key, _|
            key_name = key.to_s
            key_name.start_with?("model.layers.61") || key_name.include?("rotary_emb.inv_freq")
          end
        end

        def cast_predicate
          ->(key) { !key.to_s.include?("e_score_correction_bias") }
        end
      end

      Models.register("deepseek_v3", Model, ModelArgs)
    end

    DeepSeekV3 = DeepseekV3 unless const_defined?(:DeepSeekV3)
  end
end
