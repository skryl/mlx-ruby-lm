require_relative "deepseek"

module MlxLm
  module Models
    module DeepseekV32
      class ModelArgs < DeepSeek::ModelArgs
        field :model_type, default: "deepseek_v32"
        field :index_head_dim, default: 128
        field :index_n_heads, default: 64
        field :index_topk, default: 2048
        field :routed_scaling_factor, default: 1.0
        field :kv_lora_rank, default: 512
        field :q_lora_rank, default: 1536
        field :qk_rope_head_dim, default: 64
        field :v_head_dim, default: 128
        field :qk_nope_head_dim, default: 128
        field :topk_method, default: "noaux_tc"
        field :scoring_func, default: "sigmoid"
        field :norm_topk_prob, default: true
        field :n_group, default: 1
        field :topk_group, default: 1
      end

      class Model < DeepSeek::Model
        def sanitize(weights)
          sanitized = super(weights)
          drop_mtp_layer_weights(sanitized)
        end

        private

        def drop_mtp_layer_weights(weights)
          cutoff = @args.num_hidden_layers.to_i

          weights.reject do |key, _|
            match = key.match(/\Amodel\.layers\.(\d+)(?:\.|\z)/)
            match && match[1].to_i >= cutoff
          end
        end
      end

      Models.register("deepseek_v32", Model, ModelArgs)
    end
  end
end
