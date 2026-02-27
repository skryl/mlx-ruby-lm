# frozen_string_literal: true

require_relative "../test_helper"

class Afm7BailingMoeLinearFalconH1Glm4MoeLiteTest < Minitest::Test
  MODEL_TYPES = %w[
    afm7
    bailing_moe_linear
    falcon_h1
    glm4_moe_lite
  ].freeze

  def test_phase26_model_keys_resolve_with_tiny_configs
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
    when "afm7"
      {
        "model_type" => "afm7",
        "vocab_size" => 64,
        "hidden_dim" => 16,
        "num_layers" => 1,
        "num_kv_reuse_layers" => 0,
        "num_heads" => 2,
        "num_kv_heads" => 1,
        "hidden_dim_scale_factor" => 2.0,
        "rms_norm_eps" => 1e-5,
        "rope_theta" => 10_000.0,
        "max_position_embeddings" => 128,
      }
    when "bailing_moe_linear"
      {
        "model_type" => "bailing_moe_linear",
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
        "layer_group_size" => 1,
        "group_norm_size" => 1,
      }
    when "falcon_h1"
      {
        "model_type" => "falcon_h1",
        "hidden_size" => 16,
        "intermediate_size" => 32,
        "num_hidden_layers" => 1,
        "num_attention_heads" => 2,
        "num_key_value_heads" => 1,
        "mamba_d_conv" => 3,
        "rms_norm_eps" => 1e-5,
        "rope_theta" => 10_000.0,
        "vocab_size" => 64,
        "max_position_embeddings" => 128,
      }
    else
      {
        "model_type" => "glm4_moe_lite",
        "vocab_size" => 64,
        "hidden_size" => 16,
        "intermediate_size" => 32,
        "moe_intermediate_size" => 24,
        "num_hidden_layers" => 1,
        "num_attention_heads" => 2,
        "num_key_value_heads" => 1,
        "n_shared_experts" => 1,
        "n_routed_experts" => 2,
        "routed_scaling_factor" => 1.0,
        "kv_lora_rank" => 8,
        "q_lora_rank" => 8,
        "qk_rope_head_dim" => 4,
        "qk_nope_head_dim" => 4,
        "v_head_dim" => 8,
        "topk_method" => "noaux_tc",
        "scoring_func" => "sigmoid",
        "norm_topk_prob" => true,
        "n_group" => 1,
        "topk_group" => 1,
        "num_experts_per_tok" => 1,
        "moe_layer_freq" => 1,
        "first_k_dense_replace" => 0,
        "max_position_embeddings" => 128,
        "rms_norm_eps" => 1e-5,
        "rope_theta" => 10_000.0,
        "attention_bias" => false,
        "partial_rotary_factor" => 1.0,
      }
    end
  end
end
