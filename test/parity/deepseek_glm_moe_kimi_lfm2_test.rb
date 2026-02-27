# frozen_string_literal: true

require_relative "../test_helper"

class DeepseekGlmMoeKimiLfm2Test < Minitest::Test
  MODEL_TYPES = %w[
    deepseek_v2
    deepseek_v3
    deepseek_v32
    glm_moe_dsa
    kimi_k25
    kimi_vl
    lfm2
    lfm2-vl
  ].freeze

  def test_phase22_model_keys_resolve_with_tiny_configs
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
    when "deepseek_v2", "deepseek_v3", "deepseek_v32"
      {
        "model_type" => model_type,
        "hidden_size" => 16,
        "num_hidden_layers" => 1,
        "num_attention_heads" => 1,
        "num_key_value_heads" => 1,
        "intermediate_size" => 32,
        "vocab_size" => 64,
      }
    when "glm_moe_dsa"
      {
        "model_type" => "glm_moe_dsa",
        "hidden_size" => 16,
        "num_hidden_layers" => 1,
        "num_attention_heads" => 1,
        "num_key_value_heads" => 1,
        "intermediate_size" => 32,
        "vocab_size" => 64,
        "rope_parameters" => {
          "rope_theta" => 10_000.0,
          "type" => "linear",
          "factor" => 1.0,
        },
      }
    when "kimi_k25", "kimi_vl"
      {
        "model_type" => model_type,
        "text_config" => {
          "model_type" => "deepseek",
          "hidden_size" => 16,
          "num_hidden_layers" => 1,
          "num_attention_heads" => 1,
          "num_key_value_heads" => 1,
          "intermediate_size" => 32,
          "vocab_size" => 64,
        },
      }
    when "lfm2"
      {
        "model_type" => "lfm2",
        "hidden_size" => 16,
        "num_hidden_layers" => 1,
        "num_attention_heads" => 1,
        "num_key_value_heads" => 1,
        "block_ff_dim" => 32,
        "vocab_size" => 64,
      }
    else
      {
        "model_type" => "lfm2-vl",
        "text_config" => {
          "model_type" => "lfm2",
          "hidden_size" => 16,
          "num_hidden_layers" => 1,
          "num_attention_heads" => 1,
          "num_key_value_heads" => 1,
          "block_ff_dim" => 32,
          "vocab_size" => 64,
        },
      }
    end
  end
end
