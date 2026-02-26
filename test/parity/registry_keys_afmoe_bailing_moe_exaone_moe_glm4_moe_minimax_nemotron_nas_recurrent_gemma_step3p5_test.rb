# frozen_string_literal: true

require_relative "../test_helper"

class Phase25IntegrationRegistryTest < Minitest::Test
  MODEL_TYPES = %w[
    afmoe
    bailing_moe
    exaone_moe
    glm4_moe
    minimax
    nemotron-nas
    recurrent_gemma
    step3p5
  ].freeze

  def test_phase25_model_keys_resolve_with_tiny_configs
    MODEL_TYPES.each do |model_type|
      assert MlxLm::Models::REGISTRY.key?(model_type), "#{model_type} should be registered"

      model_class, args_class = MlxLm::Models.get_classes(tiny_config(model_type))

      assert_kind_of Class, model_class, "#{model_type} should resolve to a model class"
      assert_kind_of Class, args_class, "#{model_type} should resolve to a model args class"
      assert_instance_of args_class, args_class.from_dict(tiny_config(model_type))
    end
  end

  private

  def tiny_config(model_type)
    case model_type
    when "afmoe"
      {
        "model_type" => "afmoe",
        "vocab_size" => 64,
        "hidden_size" => 16,
        "intermediate_size" => 32,
        "moe_intermediate_size" => 24,
        "num_hidden_layers" => 1,
        "num_attention_heads" => 2,
        "num_key_value_heads" => 1,
        "head_dim" => 8,
        "max_position_embeddings" => 128,
        "rms_norm_eps" => 1e-5,
        "rope_theta" => 10_000.0,
        "num_experts" => 2,
        "num_experts_per_tok" => 1,
        "num_shared_experts" => 1,
        "num_dense_layers" => 1,
        "route_norm" => true,
        "route_scale" => 1.0,
        "score_func" => "sigmoid",
        "n_group" => 1,
        "topk_group" => 1,
        "sliding_window" => 8,
        "mup_enabled" => false,
        "layer_types" => ["full_attention"],
      }
    when "bailing_moe"
      {
        "model_type" => "bailing_moe",
        "hidden_size" => 16,
        "intermediate_size" => 32,
        "max_position_embeddings" => 128,
        "moe_intermediate_size" => 24,
        "num_experts" => 2,
        "num_shared_experts" => 1,
        "norm_topk_prob" => true,
        "num_attention_heads" => 2,
        "num_experts_per_tok" => 1,
        "num_hidden_layers" => 1,
        "num_key_value_heads" => 1,
        "rms_norm_eps" => 1e-5,
        "rope_theta" => 10_000.0,
        "vocab_size" => 64,
        "first_k_dense_replace" => 0,
        "use_bias" => false,
        "use_qkv_bias" => false,
        "score_function" => "softmax",
        "n_group" => 1,
        "topk_group" => 1,
      }
    when "exaone_moe"
      {
        "model_type" => "exaone_moe",
        "vocab_size" => 64,
        "hidden_size" => 16,
        "intermediate_size" => 32,
        "moe_intermediate_size" => 24,
        "num_hidden_layers" => 1,
        "num_attention_heads" => 2,
        "num_key_value_heads" => 1,
        "head_dim" => 8,
        "num_experts" => 2,
        "num_experts_per_tok" => 1,
        "num_shared_experts" => 1,
        "rms_norm_eps" => 1e-5,
        "max_position_embeddings" => 128,
        "sliding_window" => 8,
        "layer_types" => ["full_attention"],
        "is_moe_layer" => [true],
        "n_group" => 1,
        "topk_group" => 1,
        "routed_scaling_factor" => 1.0,
        "norm_topk_prob" => true,
        "rope_theta" => 10_000.0,
      }
    when "glm4_moe"
      {
        "model_type" => "glm4_moe",
        "vocab_size" => 64,
        "hidden_size" => 16,
        "intermediate_size" => 32,
        "max_position_embeddings" => 128,
        "moe_intermediate_size" => 24,
        "norm_topk_prob" => true,
        "num_attention_heads" => 2,
        "n_group" => 1,
        "head_dim" => 8,
        "topk_group" => 1,
        "n_shared_experts" => 1,
        "n_routed_experts" => 2,
        "routed_scaling_factor" => 1.0,
        "num_experts_per_tok" => 1,
        "first_k_dense_replace" => 0,
        "num_hidden_layers" => 1,
        "num_key_value_heads" => 1,
        "rms_norm_eps" => 1e-5,
        "rope_theta" => 10_000.0,
        "use_qk_norm" => true,
        "attention_bias" => false,
        "partial_rotary_factor" => 1.0,
      }
    when "minimax"
      {
        "model_type" => "minimax",
        "hidden_size" => 16,
        "intermediate_size" => 32,
        "num_attention_heads" => 2,
        "num_key_value_heads" => 1,
        "max_position_embeddings" => 128,
        "num_experts_per_tok" => 1,
        "num_local_experts" => 2,
        "shared_intermediate_size" => 24,
        "num_hidden_layers" => 1,
        "rms_norm_eps" => 1e-5,
        "rope_theta" => 10_000.0,
        "rotary_dim" => 8,
        "vocab_size" => 64,
        "use_qk_norm" => true,
      }
    when "nemotron-nas"
      {
        "model_type" => "nemotron-nas",
        "hidden_size" => 16,
        "num_hidden_layers" => 1,
        "num_attention_heads" => 2,
        "rms_norm_eps" => 1e-5,
        "vocab_size" => 64,
        "hidden_act" => "silu",
        "attention_bias" => false,
        "mlp_bias" => false,
        "rope_theta" => 10_000.0,
        "max_position_embeddings" => 128,
        "block_configs" => [
          {
            "attention" => { "no_op" => true },
            "ffn" => { "replace_with_linear" => true },
          },
        ],
      }
    when "recurrent_gemma"
      {
        "model_type" => "recurrent_gemma",
        "hidden_size" => 16,
        "attention_bias" => false,
        "conv1d_width" => 3,
        "intermediate_size" => 32,
        "logits_soft_cap" => 1.0,
        "num_attention_heads" => 2,
        "num_hidden_layers" => 1,
        "num_key_value_heads" => 1,
        "rms_norm_eps" => 1e-5,
        "rope_theta" => 10_000.0,
        "attention_window_size" => 8,
        "vocab_size" => 64,
        "block_types" => ["recurrent"],
      }
    else
      {
        "model_type" => "step3p5",
        "hidden_size" => 16,
        "num_hidden_layers" => 1,
        "vocab_size" => 64,
        "num_attention_heads" => 2,
        "num_attention_groups" => 1,
        "head_dim" => 8,
        "intermediate_size" => 32,
        "rms_norm_eps" => 1e-5,
        "rope_theta" => 10_000.0,
        "sliding_window" => 8,
        "layer_types" => ["full_attention"],
        "partial_rotary_factors" => [1.0],
        "attention_other_setting" => {
          "num_attention_heads" => 2,
          "num_attention_groups" => 1,
        },
        "moe_num_experts" => 2,
        "moe_top_k" => 1,
        "moe_intermediate_size" => 24,
        "share_expert_dim" => 24,
        "moe_layers_enum" => "0",
      }
    end
  end
end
