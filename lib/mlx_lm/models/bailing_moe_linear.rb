require_relative "bailing_moe"

module MlxLm
  module Models
    module BailingMoeLinear
      class ModelArgs < BailingMoe::ModelArgs
        field :model_type, default: "bailing_moe_linear"
        field :layer_group_size, default: nil
        field :group_norm_size, default: nil
        field :use_rmsnorm, default: nil
        field :head_dim, default: nil
        field :rope_traditional, default: false

        def to_bailing_moe_dict
          {
            "model_type" => @model_type,
            "hidden_size" => @hidden_size,
            "intermediate_size" => @intermediate_size,
            "max_position_embeddings" => @max_position_embeddings,
            "moe_intermediate_size" => @moe_intermediate_size,
            "num_experts" => @num_experts,
            "num_shared_experts" => @num_shared_experts,
            "norm_topk_prob" => @norm_topk_prob,
            "num_attention_heads" => @num_attention_heads,
            "num_experts_per_tok" => @num_experts_per_tok,
            "num_hidden_layers" => @num_hidden_layers,
            "num_key_value_heads" => @num_key_value_heads,
            "rms_norm_eps" => @rms_norm_eps,
            "rope_theta" => @rope_theta,
            "vocab_size" => @vocab_size,
            "first_k_dense_replace" => @first_k_dense_replace,
            "rope_scaling" => @rope_scaling,
            "use_bias" => @use_bias,
            "use_qkv_bias" => @use_qkv_bias,
            "norm_head" => @norm_head,
            "norm_softmax" => @norm_softmax,
            "use_qk_norm" => @use_qk_norm,
            "tie_word_embeddings" => @tie_word_embeddings,
            "partial_rotary_factor" => @partial_rotary_factor,
            "rotary_dim" => @rotary_dim,
            "moe_router_enable_expert_bias" => @moe_router_enable_expert_bias,
            "moe_router_enable_routed_scaling" => @moe_router_enable_routed_scaling,
            "routed_scaling_factor" => @routed_scaling_factor,
            "score_function" => @score_function,
            "n_group" => @n_group,
            "topk_group" => @topk_group,
            "moe_shared_expert_intermediate_size" => @moe_shared_expert_intermediate_size,
            "moe_router_enable_shared_expert" => @moe_router_enable_shared_expert,
          }
        end
      end

      class Model < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.model_type = args.model_type
          self.wrapped_model = BailingMoe::Model.new(BailingMoe::ModelArgs.from_dict(args.to_bailing_moe_dict))
        end

        def call(inputs, cache: nil)
          wrapped_model.call(inputs, cache: cache)
        end

        def sanitize(weights)
          wrapped_model.sanitize(weights)
        end

        def layers
          wrapped_model.layers
        end

        def make_cache
          return nil unless wrapped_model.respond_to?(:make_cache)

          wrapped_model.make_cache
        end

        def cast_predicate
          wrapped_model.cast_predicate
        end

        def quant_predicate
          wrapped_model.quant_predicate
        end
      end

      Models.register("bailing_moe_linear", Model, ModelArgs)
    end
  end
end
