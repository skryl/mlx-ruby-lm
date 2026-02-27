$LOAD_PATH.unshift File.expand_path("../../lib", __dir__)
$LOAD_PATH.unshift File.expand_path("../../mlx-ruby/lib", __dir__)

require "mlx"
require "minitest/autorun"

require_relative "../../lib/mlx_lm/model_args"
require_relative "../../lib/mlx_lm/models"
require_relative "../../lib/mlx_lm/models/activations"
require_relative "../../lib/mlx_lm/models/cache"
require_relative "../../lib/mlx_lm/models/ssm"
require_relative "../../lib/mlx_lm/models/mamba"
require_relative "../../lib/mlx_lm/models/mamba2"

class Phase23DenseLaneADMambaTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_mamba_tiny_construct_forward_shape_and_sanitize_conv_transpose
    args = MlxLm::Models::Mamba::ModelArgs.from_dict({
      "model_type" => "mamba",
      "vocab_size" => 67,
      "hidden_size" => 32,
      "intermediate_size" => 16,
      "state_size" => 8,
      "num_hidden_layers" => 2,
      "conv_kernel" => 3,
      "use_bias" => true,
      "use_conv_bias" => true,
      "time_step_rank" => "auto",
      "tie_word_embeddings" => true,
    })

    model = MlxLm::Models::Mamba::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3, 4]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 4, 67], output.shape

    conv_weight = @mx.array((0...6).to_a, dtype: @mx.float32).reshape([2, 1, 3])
    weights = {
      "backbone.layers.0.mixer.conv1d.weight" => conv_weight,
      "backbone.layers.0.mixer.in_proj.weight" => @mx.zeros([32, 32]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)
    sanitized_conv = sanitized["backbone.layers.0.mixer.conv1d.weight"]
    @mx.eval(sanitized_conv)

    assert_equal [2, 3, 1], sanitized_conv.shape
    assert_equal @mx.swapaxes(conv_weight, 1, 2).to_a, sanitized_conv.to_a
    assert sanitized.key?("backbone.layers.0.mixer.in_proj.weight")
  end
end

class Phase23DenseLaneADMamba2Test < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_mamba2_tiny_construct_forward_shape_and_sanitize_conv_transpose
    args = MlxLm::Models::Mamba2::ModelArgs.from_dict({
      "model_type" => "mamba2",
      "num_heads" => 4,
      "head_dim" => 4,
      "vocab_size" => 71,
      "hidden_size" => 32,
      "intermediate_size" => 16,
      "state_size" => 8,
      "num_hidden_layers" => 2,
      "layer_norm_epsilon" => 1e-5,
      "conv_kernel" => 3,
      "n_groups" => 2,
      "use_bias" => true,
      "use_conv_bias" => true,
      "tie_word_embeddings" => true,
      "time_step_limit" => [0.001, 10.0],
      "time_step_rank" => "auto",
    })

    model = MlxLm::Models::Mamba2::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 3, 71], output.shape

    conv_weight = @mx.array((0...12).to_a, dtype: @mx.float32).reshape([3, 1, 4])
    weights = {
      "backbone.layers.0.mixer.conv1d.weight" => conv_weight,
      "backbone.layers.0.mixer.in_proj.weight" => @mx.zeros([16, 32]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)
    sanitized_conv = sanitized["backbone.layers.0.mixer.conv1d.weight"]
    @mx.eval(sanitized_conv)

    assert_equal [3, 4, 1], sanitized_conv.shape
    assert_equal @mx.swapaxes(conv_weight, 1, 2).to_a, sanitized_conv.to_a
    assert sanitized.key?("backbone.layers.0.mixer.in_proj.weight")
  end
end

class Phase23DenseLaneADRegistryTest < Minitest::Test
  def test_models_registered_and_resolve
    assert MlxLm::Models::REGISTRY.key?("mamba"), "mamba should be registered"
    assert MlxLm::Models::REGISTRY.key?("mamba2"), "mamba2 should be registered"

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "mamba" })
    assert_equal MlxLm::Models::Mamba::Model, model_class
    assert_equal MlxLm::Models::Mamba::ModelArgs, args_class

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "mamba2" })
    assert_equal MlxLm::Models::Mamba2::Model, model_class
    assert_equal MlxLm::Models::Mamba2::ModelArgs, args_class
  end

  def test_falcon_mamba_alias_remaps_to_mamba_and_enables_bcdt_rms
    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "falcon_mamba" })
    assert_equal MlxLm::Models::Mamba::Model, model_class
    assert_equal MlxLm::Models::Mamba::ModelArgs, args_class

    args = args_class.from_dict({
      "model_type" => "falcon_mamba",
      "vocab_size" => 64,
      "hidden_size" => 16,
      "intermediate_size" => 16,
      "state_size" => 8,
      "num_hidden_layers" => 1,
      "conv_kernel" => 3,
      "use_bias" => true,
      "use_conv_bias" => true,
      "time_step_rank" => "auto",
    })

    assert_equal true, args.use_bcdt_rms
  end
end
