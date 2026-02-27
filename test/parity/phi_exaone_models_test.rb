$LOAD_PATH.unshift File.expand_path("../../lib", __dir__)
$LOAD_PATH.unshift File.expand_path("../../mlx-ruby/lib", __dir__)

require "mlx"
require "minitest/autorun"

require_relative "../../lib/mlx_lm/model_args"
require_relative "../../lib/mlx_lm/models"
require_relative "../../lib/mlx_lm/models/activations"
require_relative "../../lib/mlx_lm/models/rope_utils"
require_relative "../../lib/mlx_lm/models/phi"
require_relative "../../lib/mlx_lm/models/exaone"

class Phase16DenseLaneBRegistryTest < Minitest::Test
  def test_models_registered
    assert MlxLm::Models::REGISTRY.key?("phi"), "phi should be registered"
    assert MlxLm::Models::REGISTRY.key?("exaone"), "exaone should be registered"
  end

  def test_get_classes_resolves_lane_b_models
    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "phi" })
    assert_equal MlxLm::Models::Phi::Model, model_class
    assert_equal MlxLm::Models::Phi::ModelArgs, args_class

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "exaone" })
    assert_equal MlxLm::Models::Exaone::Model, model_class
    assert_equal MlxLm::Models::Exaone::ModelArgs, args_class
  end
end

class Phase16DenseLaneBPhiTest < Minitest::Test
  def setup
    @mx = MLX::Core
    @args = MlxLm::Models::Phi::ModelArgs.from_dict({
      "model_type" => "phi",
      "hidden_size" => 32,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 64,
      "vocab_size" => 100,
      "layer_norm_eps" => 1e-5,
      "partial_rotary_factor" => 0.5,
      "rope_theta" => 10_000.0,
    })
  end

  def test_phi_model_instantiates
    model = MlxLm::Models::Phi::Model.new(@args)
    assert_instance_of MlxLm::Models::Phi::Model, model
    assert_equal 2, model.layers.length
  end

  def test_phi_forward_shape
    model = MlxLm::Models::Phi::Model.new(@args)
    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    assert_equal [1, 3, 100], output.shape
  end
end

class Phase16DenseLaneBExaoneTest < Minitest::Test
  def setup
    @mx = MLX::Core
    @args = MlxLm::Models::Exaone::ModelArgs.from_dict({
      "model_type" => "exaone",
      "hidden_size" => 32,
      "num_layers" => 2,
      "intermediate_size" => 64,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "vocab_size" => 100,
      "rope_theta" => 10_000.0,
      "layer_norm_epsilon" => 1e-5,
      "tie_word_embeddings" => true,
      "attention_bias" => false,
      "mlp_bias" => false,
    })
  end

  def test_exaone_model_instantiates
    model = MlxLm::Models::Exaone::Model.new(@args)
    assert_instance_of MlxLm::Models::Exaone::Model, model
    assert_equal 2, model.layers.length
  end

  def test_exaone_forward_shape
    model = MlxLm::Models::Exaone::Model.new(@args)
    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    assert_equal [1, 3, 100], output.shape
  end
end
