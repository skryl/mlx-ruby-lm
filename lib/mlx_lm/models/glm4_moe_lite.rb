require_relative "glm4_moe"

module MlxLm
  module Models
    module Glm4MoeLite
      class ModelArgs < BaseModelArgs
        field :model_type, default: "glm4_moe_lite"
        field :vocab_size, default: 154_880
        field :hidden_size, default: 2048
        field :intermediate_size, default: 10_240
        field :moe_intermediate_size, default: 1536
        field :num_hidden_layers, default: 47
        field :num_attention_heads, default: 20
        field :num_key_value_heads, default: 20
        field :n_shared_experts, default: 1
        field :n_routed_experts, default: 64
        field :routed_scaling_factor, default: 1.8
        field :kv_lora_rank, default: 512
        field :q_lora_rank, default: 768
        field :qk_rope_head_dim, default: 64
        field :qk_nope_head_dim, default: 192
        field :v_head_dim, default: 256
        field :topk_method, default: "noaux_tc"
        field :scoring_func, default: "sigmoid"
        field :norm_topk_prob, default: true
        field :n_group, default: 1
        field :topk_group, default: 1
        field :num_experts_per_tok, default: 4
        field :moe_layer_freq, default: 1
        field :first_k_dense_replace, default: 1
        field :max_position_embeddings, default: 202_752
        field :rms_norm_eps, default: 1e-5
        field :rope_theta, default: 1_000_000.0
        field :rope_scaling, default: nil
        field :attention_bias, default: false
        field :attention_dropout, default: 0.0
        field :partial_rotary_factor, default: 1.0
        field :tie_word_embeddings, default: false
        field :num_nextn_predict_layers, default: 1
        field :quantization, default: nil

        def initialize(**kwargs)
          super
          @num_key_value_heads ||= @num_attention_heads
        end
      end

      class Model < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.model_type = args.model_type
          self.language_model = Glm4Moe::Model.new(
            Glm4Moe::ModelArgs.from_dict(_to_glm4_moe_config(args))
          )
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

        def _to_glm4_moe_config(args)
          inferred_head_dim = args.qk_nope_head_dim.to_i + args.qk_rope_head_dim.to_i
          inferred_head_dim = args.hidden_size / args.num_attention_heads if inferred_head_dim <= 0

          {
            "model_type" => args.model_type,
            "vocab_size" => args.vocab_size,
            "hidden_size" => args.hidden_size,
            "intermediate_size" => args.intermediate_size,
            "max_position_embeddings" => args.max_position_embeddings,
            "moe_intermediate_size" => args.moe_intermediate_size,
            "norm_topk_prob" => args.norm_topk_prob,
            "num_attention_heads" => args.num_attention_heads,
            "n_group" => args.n_group,
            "head_dim" => inferred_head_dim,
            "topk_group" => args.topk_group,
            "n_shared_experts" => args.n_shared_experts,
            "n_routed_experts" => args.n_routed_experts,
            "routed_scaling_factor" => args.routed_scaling_factor,
            "num_experts_per_tok" => args.num_experts_per_tok,
            "first_k_dense_replace" => args.first_k_dense_replace,
            "num_hidden_layers" => args.num_hidden_layers,
            "num_key_value_heads" => args.num_key_value_heads,
            "rms_norm_eps" => args.rms_norm_eps,
            "rope_theta" => args.rope_theta,
            "rope_scaling" => args.rope_scaling,
            "use_qk_norm" => false,
            "tie_word_embeddings" => args.tie_word_embeddings,
            "attention_bias" => args.attention_bias,
            "partial_rotary_factor" => args.partial_rotary_factor,
            "scoring_func" => args.scoring_func,
            "topk_method" => args.topk_method,
          }
        end

        def _remap_weight_key(key)
          mapped = key.dup
          mapped = mapped.gsub(".self_attn.embed_q.", ".self_attn.q_proj.")
          mapped = mapped.gsub(".self_attn.unembed_out.", ".self_attn.v_proj.")
          mapped = mapped.gsub(".self_attn.kv_a_proj_with_mqa.", ".self_attn.k_proj.")
          mapped = mapped.gsub(".self_attn.q_a_proj.", ".self_attn.q_proj.")
          mapped = mapped.gsub(".self_attn.q_b_proj.", ".self_attn.q_proj.")
          mapped
        end
      end

      Models.register("glm4_moe_lite", Model, ModelArgs)
    end
  end
end
