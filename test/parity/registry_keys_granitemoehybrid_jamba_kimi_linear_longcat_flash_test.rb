# frozen_string_literal: true

require_relative "../test_helper"

class Phase27IntegrationRegistryTest < Minitest::Test
  MODEL_TYPES = %w[
    granitemoehybrid
    jamba
    kimi_linear
    longcat_flash
  ].freeze

  def test_phase27_model_keys_resolve_with_tiny_configs
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
    when "granitemoehybrid"
      {
        "model_type" => "granitemoehybrid",
        "vocab_size" => 64,
        "hidden_size" => 16,
        "intermediate_size" => 32,
        "num_hidden_layers" => 1,
        "max_position_embeddings" => 128,
        "num_attention_heads" => 2,
        "num_key_value_heads" => 1,
        "attention_bias" => false,
        "embedding_multiplier" => 1.0,
        "attention_multiplier" => 1.0,
        "logits_scaling" => 1.0,
        "residual_multiplier" => 1.0,
        "num_local_experts" => 2,
        "num_experts_per_tok" => 1,
        "shared_intermediate_size" => 24,
        "mamba_n_heads" => 2,
        "mamba_d_head" => 8,
        "mamba_d_conv" => 3,
        "layer_types" => ["mamba"],
        "rms_norm_eps" => 1e-5,
        "rope_theta" => 10_000.0,
        "tie_word_embeddings" => true,
      }
    when "jamba"
      {
        "model_type" => "jamba",
        "hidden_size" => 16,
        "intermediate_size" => 32,
        "num_hidden_layers" => 1,
        "num_attention_heads" => 2,
        "num_key_value_heads" => 1,
        "attn_layer_offset" => 0,
        "attn_layer_period" => 1,
        "expert_layer_offset" => 0,
        "expert_layer_period" => 1,
        "mamba_d_conv" => 3,
        "mamba_d_state" => 8,
        "mamba_expand" => 2,
        "num_experts" => 2,
        "num_experts_per_tok" => 1,
        "rms_norm_eps" => 1e-5,
        "max_position_embeddings" => 128,
        "rope_theta" => 10_000.0,
        "vocab_size" => 64,
        "tie_word_embeddings" => true,
      }
    when "kimi_linear"
      {
        "model_type" => "kimi_linear",
        "vocab_size" => 64,
        "hidden_dim" => 16,
        "ffn_hidden_size" => 32,
        "moe_intermediate_size" => 24,
        "num_layers" => 1,
        "num_heads" => 2,
        "num_kv_heads" => 1,
        "num_local_experts" => 2,
        "n_shared_experts" => 1,
        "top_k" => 1,
        "norm_topk_prob" => true,
        "max_position_embeddings" => 128,
        "rms_norm_eps" => 1e-5,
        "rope_theta" => 10_000.0,
        "first_k_dense_replace" => 0,
        "layer_group_size" => 1,
        "group_norm_size" => 1,
        "use_bias" => false,
        "use_qkv_bias" => false,
      }
    else
      {
        "model_type" => "longcat_flash",
        "vocab_size" => 64,
        "hidden_dim" => 16,
        "ffn_hidden_size" => 32,
        "moe_intermediate_size" => 24,
        "num_layers" => 1,
        "num_heads" => 2,
        "num_kv_heads" => 1,
        "num_local_experts" => 2,
        "num_shared_experts" => 1,
        "routed_scaling_factor" => 1.0,
        "kv_lora_rank" => 8,
        "q_lora_rank" => 8,
        "qk_rope_head_dim" => 4,
        "qk_nope_head_dim" => 4,
        "v_head_dim" => 8,
        "topk_method" => "noaux_tc",
        "score_function" => "sigmoid",
        "norm_topk_prob" => true,
        "n_group" => 1,
        "topk_group" => 1,
        "top_k" => 1,
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
