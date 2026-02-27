# frozen_string_literal: true

require_relative "../test_helper"

class Phase28IntegrationRegistryTest < Minitest::Test
  MODEL_TYPES = %w[
    longcat_flash_ngram
    nemotron_h
    plamo2
    qwen3_next
    rwkv7
  ].freeze

  def test_phase28_model_keys_resolve_with_tiny_configs
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
    when "longcat_flash_ngram"
      {
        "model_type" => "longcat_flash_ngram",
        "vocab_size" => 64,
        "hidden_size" => 16,
        "ffn_hidden_size" => 32,
        "expert_ffn_hidden_size" => 24,
        "num_layers" => 1,
        "num_attention_heads" => 2,
        "n_routed_experts" => 2,
        "zero_expert_num" => 1,
        "moe_topk" => 1,
        "kv_lora_rank" => 8,
        "q_lora_rank" => 8,
        "qk_rope_head_dim" => 4,
        "qk_nope_head_dim" => 4,
        "v_head_dim" => 8,
        "routed_scaling_factor" => 1.0,
        "max_position_embeddings" => 128,
        "rms_norm_eps" => 1e-5,
        "rope_theta" => 10_000.0,
        "attention_bias" => false,
        "norm_topk_prob" => true,
        "n_group" => 1,
        "topk_group" => 1,
        "first_k_dense_replace" => 0,
      }
    when "nemotron_h"
      {
        "model_type" => "nemotron_h",
        "vocab_size" => 64,
        "hidden_size" => 16,
        "intermediate_size" => 32,
        "num_hidden_layers" => 1,
        "max_position_embeddings" => 128,
        "num_attention_heads" => 2,
        "num_key_value_heads" => 1,
        "attention_bias" => false,
        "mamba_num_heads" => 2,
        "mamba_head_dim" => 8,
        "conv_kernel" => 3,
        "layer_norm_epsilon" => 1e-5,
        "hybrid_override_pattern" => ["M"],
      }
    when "plamo2"
      {
        "model_type" => "plamo2",
        "hidden_size" => 16,
        "num_hidden_layers" => 1,
        "rms_norm_eps" => 1e-5,
        "tie_word_embeddings" => true,
        "num_attention_heads" => 2,
        "num_key_value_heads" => 1,
        "hidden_size_per_head" => 8,
        "max_position_embeddings" => 128,
        "attention_window_size" => 16,
        "mamba_d_conv" => 3,
        "mamba_step" => 2,
        "mamba_enabled" => true,
        "intermediate_size" => 32,
        "vocab_size" => 64,
      }
    when "qwen3_next"
      {
        "model_type" => "qwen3_next",
        "vocab_size" => 64,
        "hidden_size" => 16,
        "intermediate_size" => 32,
        "moe_intermediate_size" => 24,
        "num_hidden_layers" => 1,
        "num_attention_heads" => 2,
        "num_key_value_heads" => 1,
        "num_experts" => 2,
        "num_experts_per_tok" => 1,
        "shared_expert_intermediate_size" => 16,
        "decoder_sparse_step" => 1,
        "mlp_only_layers" => [0],
        "linear_num_value_heads" => 1,
        "linear_num_key_heads" => 1,
        "linear_key_head_dim" => 8,
        "linear_value_head_dim" => 8,
        "linear_conv_kernel_dim" => 3,
        "max_position_embeddings" => 128,
        "rms_norm_eps" => 1e-5,
        "rope_theta" => 10_000.0,
        "partial_rotary_factor" => 0.5,
        "attention_bias" => false,
      }
    else
      {
        "model_type" => "rwkv7",
        "vocab_size" => 64,
        "hidden_size" => 16,
        "intermediate_size" => 32,
        "norm_eps" => 1e-5,
        "head_dim" => 8,
        "num_hidden_layers" => 1,
        "a_low_rank_dim" => 4,
        "v_low_rank_dim" => 4,
        "gate_low_rank_dim" => 4,
        "decay_low_rank_dim" => 4,
      }
    end
  end
end
