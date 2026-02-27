$LOAD_PATH.unshift File.expand_path("../../lib", __dir__)
$LOAD_PATH.unshift File.expand_path("../../mlx-ruby/lib", __dir__)

require "mlx"
require "minitest/autorun"

require_relative "../../lib/mlx_lm/model_args"
require_relative "../../lib/mlx_lm/models"
require_relative "../../lib/mlx_lm/models/cache"
require_relative "../../lib/mlx_lm/models/activations"
require_relative "../../lib/mlx_lm/models/rope_utils"
require_relative "../../lib/mlx_lm/models/ernie4_5"
require_relative "../../lib/mlx_lm/models/baichuan_m1"

class Ernie45BaichuanM1ModelsErnie45Test < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_ernie45_construct_forward_shape_and_head_dim_default
    args = MlxLm::Models::Ernie45::ModelArgs.from_dict({
      "model_type" => "ernie4_5",
      "hidden_size" => 48,
      "intermediate_size" => 96,
      "max_position_embeddings" => 256,
      "num_attention_heads" => 3,
      "num_key_value_heads" => 3,
      "num_hidden_layers" => 2,
      "rms_norm_eps" => 1e-5,
      "vocab_size" => 97,
      "rope_theta" => 10_000.0,
      "use_bias" => false,
      "tie_word_embeddings" => true,
    })

    model = MlxLm::Models::Ernie45::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    input = @mx.array([[1, 2, 3, 4]], dtype: @mx.int32)
    output = model.call(input)
    @mx.eval(output)

    assert_equal [1, 4, 97], output.shape
    assert_equal 16, model.layers.first.self_attn.instance_variable_get(:@head_dim)
  end
end

class Ernie45BaichuanM1ModelsM1Test < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_baichuan_m1_construct_forward_shape_and_sanitize_normalizes_lm_head
    args = MlxLm::Models::BaichuanM1::ModelArgs.from_dict({
      "model_type" => "baichuan_m1",
      "vocab_size" => 96,
      "hidden_size" => 32,
      "intermediate_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 2,
      "rope_theta" => 10_000.0,
      "sliding_window" => 4,
      "sliding_window_layers" => [0],
      "conv_window" => 2,
      "rms_norm_eps" => 1e-5,
      "tie_word_embeddings" => false,
    })

    model = MlxLm::Models::BaichuanM1::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    input = @mx.array([[1, 2, 3, 4]], dtype: @mx.int32)
    output = model.call(input)
    @mx.eval(output)

    assert_equal [1, 4, 96], output.shape

    weights = {
      "lm_head.weight" => @mx.array([[3.0, 4.0], [0.0, 5.0]], dtype: @mx.float32),
    }
    sanitized = model.sanitize(weights)
    norms = @mx.norm(sanitized["lm_head.weight"], nil, -1)
    @mx.eval(norms)

    assert_in_delta 1.0, norms.to_a[0], 1e-5
    assert_in_delta 1.0, norms.to_a[1], 1e-5
  end
end

class Ernie45BaichuanM1ModelsTest < Minitest::Test
  def test_models_registered
    assert MlxLm::Models::REGISTRY.key?("ernie4_5"), "ernie4_5 should be registered"
    assert MlxLm::Models::REGISTRY.key?("baichuan_m1"), "baichuan_m1 should be registered"

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "ernie4_5" })
    assert_equal MlxLm::Models::Ernie45::Model, model_class
    assert_equal MlxLm::Models::Ernie45::ModelArgs, args_class

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "baichuan_m1" })
    assert_equal MlxLm::Models::BaichuanM1::Model, model_class
    assert_equal MlxLm::Models::BaichuanM1::ModelArgs, args_class
  end
end
