# frozen_string_literal: true

require_relative "../test_helper"

class RegistryKeysOlmo3Gpt2BigcodeNemotronApertusYoutuErnieBaichuanTest < Minitest::Test
  MODEL_TYPES = %w[olmo3 gpt2 gpt_bigcode nemotron apertus youtu_llm ernie4_5 baichuan_m1].freeze

  def test_phase18_model_keys_resolve_with_tiny_configs
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
