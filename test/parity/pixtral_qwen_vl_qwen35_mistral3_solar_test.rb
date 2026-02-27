# frozen_string_literal: true

require_relative "../test_helper"

class PixtralQwenVlQwen35Mistral3SolarTest < Minitest::Test
  MODEL_TYPES = %w[
    pixtral
    qwen2_vl
    qwen3_vl
    smollm3
    qwen3_5
    qwen3_5_moe
    mistral3
    solar_open
  ].freeze

  def test_phase19_model_keys_resolve_with_tiny_configs
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
    when "pixtral"
      {
        "model_type" => "pixtral",
        "text_config" => llama_text_config,
      }
    when "qwen2_vl"
      {
        "model_type" => "qwen2_vl",
        "text_config" => qwen2_text_config,
      }
    when "qwen3_vl"
      {
        "model_type" => "qwen3_vl",
        "text_config" => qwen3_text_config("qwen3"),
      }
    when "qwen3_5"
      {
        "model_type" => "qwen3_5",
        "text_config" => qwen3_text_config("qwen3_5"),
      }
    when "qwen3_5_moe"
      {
        "model_type" => "qwen3_5_moe",
        "text_config" => qwen3_text_config("qwen3_5_moe").merge(
          "num_experts" => 2,
          "num_experts_per_tok" => 1
        ),
      }
    when "mistral3"
      {
        "model_type" => "mistral3",
        "text_config" => llama_text_config,
      }
    when "solar_open"
      {
        "model_type" => "solar_open",
        "vocab_size" => 64,
        "hidden_size" => 16,
        "intermediate_size" => 32,
        "moe_intermediate_size" => 8,
        "num_hidden_layers" => 1,
        "num_attention_heads" => 1,
        "num_key_value_heads" => 1,
        "head_dim" => 16,
        "n_shared_experts" => 1,
        "n_routed_experts" => 2,
        "routed_scaling_factor" => 1.0,
        "num_experts_per_tok" => 1,
        "first_k_dense_replace" => 0,
        "norm_topk_prob" => false,
        "max_position_embeddings" => 64,
        "rms_norm_eps" => 1e-5,
        "rope_theta" => 10_000.0,
        "tie_word_embeddings" => false,
        "partial_rotary_factor" => 1.0,
      }
    else
      {
        "model_type" => "smollm3",
        "hidden_size" => 16,
        "num_hidden_layers" => 1,
        "num_attention_heads" => 1,
        "num_key_value_heads" => 1,
        "intermediate_size" => 32,
        "vocab_size" => 64,
      }
    end
  end

  def llama_text_config
    {
      "model_type" => "llama",
      "hidden_size" => 16,
      "num_hidden_layers" => 1,
      "num_attention_heads" => 1,
      "num_key_value_heads" => 1,
      "intermediate_size" => 32,
      "vocab_size" => 64,
      "tie_word_embeddings" => false,
    }
  end

  def qwen2_text_config
    {
      "model_type" => "qwen2",
      "hidden_size" => 16,
      "num_hidden_layers" => 1,
      "num_attention_heads" => 1,
      "num_key_value_heads" => 1,
      "intermediate_size" => 32,
      "vocab_size" => 64,
      "tie_word_embeddings" => true,
    }
  end

  def qwen3_text_config(model_type)
    {
      "model_type" => model_type,
      "hidden_size" => 16,
      "num_hidden_layers" => 1,
      "num_attention_heads" => 1,
      "num_key_value_heads" => 1,
      "intermediate_size" => 32,
      "vocab_size" => 64,
      "head_dim" => 16,
      "tie_word_embeddings" => false,
    }
  end
end
