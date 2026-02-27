# frozen_string_literal: true

require_relative "../test_helper"

class RegistryKeysPhi3SmallDots1Llama4Ministral3HunyuanGptOssMimoV2FlashLfm2MoeTest < Minitest::Test
  MODEL_TYPES = %w[
    phi3small
    dots1
    llama4
    ministral3
    hunyuan
    gpt_oss
    mimo_v2_flash
    lfm2_moe
  ].freeze

  def test_phase24_model_keys_resolve_with_tiny_configs
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
    when "phi3small"
      {
        "model_type" => "phi3small",
        "hidden_size" => 16,
        "dense_attention_every_n_layers" => 1,
        "ff_intermediate_size" => 32,
        "gegelu_limit" => 16.0,
        "num_hidden_layers" => 1,
        "num_attention_heads" => 2,
        "layer_norm_epsilon" => 1e-5,
        "vocab_size" => 64,
        "num_key_value_heads" => 1,
      }
    when "dots1"
      {
        "model_type" => "dots1",
        "hidden_size" => 16,
        "num_hidden_layers" => 1,
        "intermediate_size" => 32,
        "num_attention_heads" => 2,
        "num_key_value_heads" => 1,
        "rms_norm_eps" => 1e-5,
        "vocab_size" => 64,
        "max_position_embeddings" => 128,
        "first_k_dense_replace" => 0,
        "moe_intermediate_size" => 24,
        "n_routed_experts" => 2,
        "n_shared_experts" => 1,
        "norm_topk_prob" => false,
        "num_experts_per_tok" => 1,
        "rope_theta" => 10_000.0,
        "routed_scaling_factor" => 1.0,
      }
    when "llama4"
      {
        "model_type" => "llama4",
        "text_config" => {
          "model_type" => "llama4_text",
          "hidden_size" => 16,
          "num_attention_heads" => 2,
          "num_key_value_heads" => 1,
          "num_hidden_layers" => 2,
          "vocab_size" => 64,
          "intermediate_size" => 24,
          "intermediate_size_mlp" => 24,
          "num_local_experts" => 2,
          "num_experts_per_tok" => 1,
          "interleave_moe_layer_step" => 1,
          "attention_chunk_size" => 4,
          "max_position_embeddings" => 128,
          "rope_theta" => 10_000.0,
          "head_dim" => 8,
          "rms_norm_eps" => 1e-5,
        },
      }
    when "ministral3"
      {
        "model_type" => "ministral3",
        "hidden_size" => 16,
        "num_hidden_layers" => 2,
        "intermediate_size" => 32,
        "num_attention_heads" => 2,
        "num_key_value_heads" => 1,
        "head_dim" => 8,
        "max_position_embeddings" => 128,
        "rms_norm_eps" => 1e-5,
        "vocab_size" => 64,
        "layer_types" => ["sliding_attention", "full_attention"],
        "sliding_window" => 8,
        "rope_parameters" => {
          "rope_theta" => 10_000.0,
          "llama_4_scaling_beta" => 0.0,
          "original_max_position_embeddings" => 128,
        },
      }
    when "hunyuan"
      {
        "model_type" => "hunyuan",
        "vocab_size" => 64,
        "hidden_size" => 16,
        "num_hidden_layers" => 1,
        "intermediate_size" => 32,
        "num_attention_heads" => 2,
        "num_key_value_heads" => 1,
        "attention_bias" => false,
        "moe_topk" => 1,
        "num_experts" => 2,
        "num_shared_expert" => 1,
        "use_mixed_mlp_moe" => true,
        "use_qk_norm" => true,
        "rms_norm_eps" => 1e-5,
        "rope_theta" => 10_000.0,
        "use_cla" => false,
      }
    when "gpt_oss"
      {
        "model_type" => "gpt_oss",
        "num_hidden_layers" => 2,
        "num_local_experts" => 2,
        "num_experts_per_tok" => 1,
        "vocab_size" => 64,
        "hidden_size" => 16,
        "intermediate_size" => 24,
        "head_dim" => 8,
        "num_attention_heads" => 2,
        "num_key_value_heads" => 1,
        "sliding_window" => 8,
        "layer_types" => ["sliding_attention", "full_attention"],
      }
    when "mimo_v2_flash"
      {
        "model_type" => "mimo_v2_flash",
        "num_experts_per_tok" => 1,
        "hybrid_layer_pattern" => [0],
        "moe_layer_freq" => [0],
        "add_swa_attention_sink_bias" => false,
        "add_full_attention_sink_bias" => false,
        "sliding_window_size" => 8,
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
        "topk_method" => "noaux_tc",
        "scoring_func" => "sigmoid",
        "norm_topk_prob" => false,
        "n_group" => 1,
        "topk_group" => 1,
        "max_position_embeddings" => 128,
        "layernorm_epsilon" => 1e-5,
        "rope_theta" => 10_000.0,
        "swa_rope_theta" => 10_000.0,
        "swa_num_attention_heads" => 2,
        "swa_num_key_value_heads" => 1,
        "head_dim" => 8,
        "v_head_dim" => 8,
        "swa_head_dim" => 8,
        "swa_v_head_dim" => 8,
        "partial_rotary_factor" => 1.0,
      }
    else
      {
        "model_type" => "lfm2_moe",
        "vocab_size" => 64,
        "hidden_size" => 16,
        "intermediate_size" => 32,
        "moe_intermediate_size" => 24,
        "num_hidden_layers" => 1,
        "num_experts" => 2,
        "num_experts_per_tok" => 1,
        "norm_topk_prob" => false,
        "num_attention_heads" => 2,
        "num_key_value_heads" => 1,
        "max_position_embeddings" => 128,
        "use_expert_bias" => false,
        "num_dense_layers" => 1,
        "norm_eps" => 1e-5,
        "conv_bias" => false,
        "conv_L_cache" => 3,
        "full_attn_idxs" => [0],
        "rope_theta" => 10_000.0,
      }
    end
  end
end
