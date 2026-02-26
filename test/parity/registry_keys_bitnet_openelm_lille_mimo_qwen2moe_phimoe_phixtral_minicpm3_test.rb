# frozen_string_literal: true

require_relative "../test_helper"

class Phase21IntegrationRegistryTest < Minitest::Test
  MODEL_TYPES = %w[
    bitnet
    openelm
    lille-130m
    mimo
    qwen2_moe
    phimoe
    phixtral
    minicpm3
  ].freeze

  def test_phase21_model_keys_resolve_with_tiny_configs
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
    when "openelm"
      {
        "model_type" => "openelm",
        "head_dim" => 8,
        "num_transformer_layers" => 1,
        "model_dim" => 16,
        "vocab_size" => 64,
        "ffn_dim_divisor" => 8,
        "num_query_heads" => [2],
        "num_kv_heads" => [1],
        "ffn_multipliers" => [2.0],
      }
    when "lille-130m"
      {
        "model_type" => "lille-130m",
        "block_size" => 64,
        "layer_norm_eps" => 1e-5,
        "n_embd" => 16,
        "n_head" => 1,
        "n_kv_heads" => 1,
        "n_layer" => 1,
        "rope_theta" => 10_000.0,
        "vocab_size" => 64,
      }
    when "qwen2_moe"
      {
        "model_type" => "qwen2_moe",
        "hidden_size" => 16,
        "num_hidden_layers" => 1,
        "num_attention_heads" => 1,
        "num_key_value_heads" => 1,
        "intermediate_size" => 32,
        "vocab_size" => 64,
        "num_experts_per_tok" => 1,
        "num_experts" => 2,
        "moe_intermediate_size" => 8,
        "shared_expert_intermediate_size" => 16,
      }
    when "phimoe"
      {
        "model_type" => "phimoe",
        "vocab_size" => 64,
        "hidden_size" => 16,
        "intermediate_size" => 32,
        "num_hidden_layers" => 1,
        "num_attention_heads" => 1,
        "num_key_value_heads" => 1,
        "max_position_embeddings" => 128,
        "original_max_position_embeddings" => 64,
        "num_local_experts" => 2,
        "num_experts_per_tok" => 1,
        "rope_scaling" => {
          "short_factor" => 1.0,
          "long_factor" => 1.0,
        },
      }
    when "phixtral"
      {
        "model_type" => "phixtral",
        "num_vocab" => 64,
        "model_dim" => 16,
        "num_heads" => 1,
        "num_layers" => 1,
        "rotary_dim" => 8,
        "num_experts_per_tok" => 1,
        "num_local_experts" => 2,
      }
    when "minicpm3"
      {
        "model_type" => "minicpm3",
        "hidden_size" => 16,
        "dim_model_base" => 8,
        "num_hidden_layers" => 1,
        "intermediate_size" => 32,
        "num_attention_heads" => 1,
        "num_key_value_heads" => 1,
        "vocab_size" => 64,
        "q_lora_rank" => 8,
        "qk_nope_head_dim" => 8,
        "qk_rope_head_dim" => 8,
        "kv_lora_rank" => 8,
        "scale_depth" => 1.0,
        "scale_emb" => 1.0,
        "max_position_embeddings" => 128,
      }
    else
      {
        "model_type" => model_type,
        "hidden_size" => 16,
        "num_hidden_layers" => 1,
        "num_attention_heads" => 1,
        "num_key_value_heads" => 1,
        "intermediate_size" => 32,
        "vocab_size" => 64,
      }
    end
  end
end
