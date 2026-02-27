require_relative "deepseek"

module MlxLm
  module Models
    module DeepseekV2
      class ModelArgs < BaseModelArgs
        field :model_type, default: "deepseek_v2"
        field :vocab_size, default: 102_400
        field :hidden_size, default: 4096
        field :intermediate_size, default: 11_008
        field :moe_intermediate_size, default: 1407
        field :num_hidden_layers, default: 30
        field :num_attention_heads, default: 32
        field :num_key_value_heads, default: 32
        field :n_shared_experts, default: nil
        field :n_routed_experts, default: nil
        field :routed_scaling_factor, default: 1.0
        field :kv_lora_rank, default: 512
        field :q_lora_rank, default: 1536
        field :qk_rope_head_dim, default: 64
        field :v_head_dim, default: 128
        field :qk_nope_head_dim, default: 128
        field :topk_method, default: "gready"
        field :n_group, default: nil
        field :topk_group, default: nil
        field :num_experts_per_tok, default: nil
        field :moe_layer_freq, default: 1
        field :first_k_dense_replace, default: 0
        field :max_position_embeddings, default: 2048
        field :rms_norm_eps, default: 1e-6
        field :rope_theta, default: 10_000.0
        field :rope_scaling, default: nil
        field :attention_bias, default: false

        def initialize(**kwargs)
          super
          @num_key_value_heads ||= @num_attention_heads
        end
      end

      class Model < DeepSeek::Model
        def initialize(args)
          super(DeepSeek::ModelArgs.from_dict(_to_deepseek_config(args)))
          self.model_type = args.model_type
        end

        def sanitize(weights)
          _stack_expert_weights(weights.dup)
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
            "moe_layer_freq" => args.moe_layer_freq,
            "first_k_dense_replace" => args.first_k_dense_replace,
            "max_position_embeddings" => args.max_position_embeddings,
            "rms_norm_eps" => args.rms_norm_eps,
            "rope_theta" => args.rope_theta,
            "rope_scaling" => args.rope_scaling,
            "attention_bias" => args.attention_bias,
          }
        end

        def _stack_expert_weights(weights)
          num_experts = @args.n_routed_experts.to_i
          return weights if num_experts <= 0

          mx = MLX::Core
          projections = %w[gate_proj down_proj up_proj].freeze
          params = %w[weight scales biases].freeze

          @args.num_hidden_layers.times do |layer_idx|
            prefix = "model.layers.#{layer_idx}.mlp"
            projections.each do |projection|
              params.each do |param|
                expert_keys = (0...num_experts).map do |expert_idx|
                  "#{prefix}.experts.#{expert_idx}.#{projection}.#{param}"
                end
                next unless expert_keys.all? { |key| weights.key?(key) }

                stacked = expert_keys.map { |key| weights.delete(key) }
                weights["#{prefix}.switch_mlp.#{projection}.#{param}"] = mx.stack(stacked)
              end
            end
          end

          weights
        end
      end

      Models.register("deepseek_v2", Model, ModelArgs)
    end

    DeepSeekV2 = DeepseekV2 unless const_defined?(:DeepSeekV2)
  end
end
