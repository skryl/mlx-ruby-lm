# frozen_string_literal: true

require_relative "../test_helper"

class Gemma3Ernie45MoeQwen3MoeGranitemoeOlmoeTest < Minitest::Test
  MODEL_TYPES = %w[
    gemma3_text
    gemma3
    gemma3n
    ernie4_5_moe
    qwen3_moe
    qwen3_vl_moe
    granitemoe
    olmoe
  ].freeze

  def test_phase20_model_keys_resolve_with_tiny_configs
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
    when "gemma3"
      {
        "model_type" => "gemma3",
        "text_config" => {
          "hidden_size" => 16,
          "num_hidden_layers" => 1,
          "intermediate_size" => 32,
          "num_attention_heads" => 2,
          "num_key_value_heads" => 1,
          "head_dim" => 8,
          "vocab_size" => 64,
        },
      }
    when "gemma3n"
      {
        "model_type" => "gemma3n",
        "text_config" => {
          "hidden_size" => 16,
          "num_hidden_layers" => 1,
          "num_attention_heads" => 2,
          "num_key_value_heads" => 1,
          "intermediate_size" => 32,
          "vocab_size" => 64,
          "head_dim" => 8,
        },
      }
    when "ernie4_5_moe"
      {
        "model_type" => "ernie4_5_moe",
        "hidden_size" => 16,
        "intermediate_size" => 32,
        "max_position_embeddings" => 64,
        "num_attention_heads" => 1,
        "num_key_value_heads" => 1,
        "num_hidden_layers" => 1,
        "rms_norm_eps" => 1e-5,
        "vocab_size" => 64,
        "rope_theta" => 10_000.0,
        "use_bias" => false,
        "tie_word_embeddings" => false,
        "moe_num_experts" => 2,
      }
    when "qwen3_vl_moe"
      {
        "model_type" => "qwen3_vl_moe",
        "text_config" => {
          "model_type" => "qwen3_moe",
          "hidden_size" => 16,
          "num_hidden_layers" => 1,
          "num_attention_heads" => 1,
          "num_key_value_heads" => 1,
          "intermediate_size" => 32,
          "vocab_size" => 64,
          "head_dim" => 16,
          "num_experts" => 2,
          "num_experts_per_tok" => 1,
        },
      }
    when "granitemoe"
      {
        "model_type" => "granitemoe",
        "hidden_size" => 16,
        "num_hidden_layers" => 1,
        "intermediate_size" => 32,
        "num_attention_heads" => 1,
        "num_key_value_heads" => 1,
        "rms_norm_eps" => 1e-5,
        "vocab_size" => 64,
        "logits_scaling" => 1.0,
        "attention_multiplier" => 1.0,
        "embedding_multiplier" => 1.0,
        "residual_multiplier" => 1.0,
        "max_position_embeddings" => 64,
        "attention_bias" => false,
        "mlp_bias" => false,
        "rope_theta" => 10_000.0,
        "num_local_experts" => 2,
        "num_experts_per_tok" => 1,
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
