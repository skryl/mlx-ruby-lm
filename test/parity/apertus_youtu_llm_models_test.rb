$LOAD_PATH.unshift File.expand_path("../../lib", __dir__)
$LOAD_PATH.unshift File.expand_path("../../mlx-ruby/lib", __dir__)

require "mlx"
require "minitest/autorun"

require_relative "../../lib/mlx_lm/model_args"
require_relative "../../lib/mlx_lm/models"
require_relative "../../lib/mlx_lm/models/cache"
require_relative "../../lib/mlx_lm/models/activations"
require_relative "../../lib/mlx_lm/models/rope_utils"
require_relative "../../lib/mlx_lm/models/apertus"
require_relative "../../lib/mlx_lm/models/youtu_llm"

class ApertusYoutuLlmModelsTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_apertus_construct_forward_shape_and_registry_resolution
    args = MlxLm::Models::Apertus::ModelArgs.from_dict({
      "model_type" => "apertus",
      "hidden_size" => 32,
      "num_hidden_layers" => 2,
      "intermediate_size" => 64,
      "mlp_bias" => false,
      "num_attention_heads" => 4,
      "attention_bias" => false,
      "rms_norm_eps" => 1e-6,
      "vocab_size" => 96,
      "num_key_value_heads" => 2,
      "max_position_embeddings" => 256,
      "rope_theta" => 10_000.0,
      "post_norm" => false,
      "qk_norm" => true,
      "tie_word_embeddings" => false,
      "rope_traditional" => false,
    })

    model = MlxLm::Models::Apertus::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    tokens = @mx.array([[1, 2, 3, 4]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)

    assert_equal [1, 4, 96], output.shape
    assert MlxLm::Models::REGISTRY.key?("apertus"), "apertus should be registered"

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "apertus" })
    assert_equal MlxLm::Models::Apertus::Model, model_class
    assert_equal MlxLm::Models::Apertus::ModelArgs, args_class
  end
end

class ApertusYoutuLlmModelsYoutuLlmConstructForwardShapeAndRegistryResolutionTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_youtu_llm_construct_forward_shape_and_registry_resolution
    args = MlxLm::Models::YoutuLLM::ModelArgs.from_dict({
      "model_type" => "youtu_llm",
      "vocab_size" => 128,
      "hidden_size" => 64,
      "intermediate_size" => 128,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 4,
      "kv_lora_rank" => 16,
      "q_lora_rank" => 24,
      "qk_rope_head_dim" => 8,
      "v_head_dim" => 16,
      "qk_nope_head_dim" => 8,
      "max_position_embeddings" => 256,
      "rms_norm_eps" => 1e-6,
      "rope_theta" => 10_000.0,
      "rope_traditional" => true,
      "attention_bias" => false,
      "mlp_bias" => false,
      "tie_word_embeddings" => true,
    })

    model = MlxLm::Models::YoutuLLM::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)

    assert_equal [1, 3, 128], output.shape
    assert MlxLm::Models::REGISTRY.key?("youtu_llm"), "youtu_llm should be registered"

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "youtu_llm" })
    assert_equal MlxLm::Models::YoutuLLM::Model, model_class
    assert_equal MlxLm::Models::YoutuLLM::ModelArgs, args_class
  end
end
