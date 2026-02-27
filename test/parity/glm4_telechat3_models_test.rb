$LOAD_PATH.unshift File.expand_path("../../lib", __dir__)
$LOAD_PATH.unshift File.expand_path("../../mlx-ruby/lib", __dir__)

require "mlx"
require "minitest/autorun"

require_relative "../../lib/mlx_lm/model_args"
require_relative "../../lib/mlx_lm/models"
require_relative "../../lib/mlx_lm/models/activations"
require_relative "../../lib/mlx_lm/models/rope_utils"
require_relative "../../lib/mlx_lm/models/glm4"
require_relative "../../lib/mlx_lm/models/telechat3"

class Glm4Telechat3ModelsM4Test < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_glm4_construct_forward_shape_and_registration_resolution
    args = MlxLm::Models::GLM4::ModelArgs.from_dict({
      "model_type" => "glm4",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "intermediate_size" => 128,
      "num_attention_heads" => 4,
      "attention_bias" => false,
      "head_dim" => 16,
      "rms_norm_eps" => 1e-5,
      "vocab_size" => 128,
      "num_key_value_heads" => 2,
      "partial_rotary_factor" => 0.5,
      "rope_theta" => 10_000.0,
      "rope_traditional" => true,
    })

    model = MlxLm::Models::GLM4::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    input = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(input)
    @mx.eval(output)

    assert_equal [1, 3, 128], output.shape
    assert MlxLm::Models::REGISTRY.key?("glm4"), "glm4 should be registered"

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "glm4" })
    assert_equal MlxLm::Models::GLM4::Model, model_class
    assert_equal MlxLm::Models::GLM4::ModelArgs, args_class
  end
end

class Glm4Telechat3ModelsTelechat3Test < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_telechat3_construct_forward_shape_and_registration_resolution
    args = MlxLm::Models::Telechat3::ModelArgs.from_dict({
      "model_type" => "telechat3",
      "hidden_size" => 64,
      "intermediate_size" => 128,
      "max_position_embeddings" => 256,
      "num_attention_heads" => 4,
      "num_hidden_layers" => 2,
      "num_key_value_heads" => 2,
      "rms_norm_eps" => 1e-6,
      "vocab_size" => 96,
      "rope_theta" => 10_000.0,
      "mlp_bias" => false,
      "attention_bias" => false,
      "head_dim" => 16,
      "tie_word_embeddings" => false,
    })

    model = MlxLm::Models::Telechat3::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    input = @mx.array([[1, 2, 3, 4]], dtype: @mx.int32)
    output = model.call(input)
    @mx.eval(output)

    assert_equal [1, 4, 96], output.shape
    assert MlxLm::Models::REGISTRY.key?("telechat3"), "telechat3 should be registered"

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "telechat3" })
    assert_equal MlxLm::Models::Telechat3::Model, model_class
    assert_equal MlxLm::Models::Telechat3::ModelArgs, args_class
  end
end
