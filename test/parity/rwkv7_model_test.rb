$LOAD_PATH.unshift File.expand_path("../../lib", __dir__)
$LOAD_PATH.unshift File.expand_path("../../mlx-ruby/lib", __dir__)

require "mlx"
require "minitest/autorun"

require_relative "../../lib/mlx_lm/model_args"
require_relative "../../lib/mlx_lm/models"
require_relative "../../lib/mlx_lm/models/cache"
require_relative "../../lib/mlx_lm/models/recurrent_gemma"
require_relative "../../lib/mlx_lm/models/rwkv7"

class Phase45DenseLaneAURwkv7Test < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_rwkv7_construct_forward_shape_sanitize_and_cache
    args = MlxLm::Models::Rwkv7::ModelArgs.from_dict({
      "model_type" => "rwkv7",
      "vocab_size" => 67,
      "hidden_size" => 32,
      "intermediate_size" => 64,
      "norm_eps" => 1e-5,
      "head_dim" => 8,
      "num_hidden_layers" => 2,
      "a_low_rank_dim" => 8,
      "v_low_rank_dim" => 8,
      "gate_low_rank_dim" => 8,
      "decay_low_rank_dim" => 8,
      "tie_word_embeddings" => false,
    })

    model = MlxLm::Models::Rwkv7::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 3, 67], output.shape
    assert_equal 2, model.layers.length

    conv_weight = @mx.array((0...24).to_a, dtype: @mx.float32).reshape([4, 1, 6])
    weights = {
      "blocks.0.time_mix.conv_1d.weight" => conv_weight,
      "blocks.0.time_mix.linear_x.weight" => @mx.zeros([32, 32]).astype(@mx.float32),
    }
    sanitized = model.sanitize(weights)
    sanitized_conv = sanitized["model.layers.0.temporal_block.conv_1d.weight"]
    @mx.eval(sanitized_conv)
    assert_equal [4, 6, 1], sanitized_conv.shape
    assert sanitized.key?("model.layers.0.temporal_block.linear_x.weight")

    cache = model.make_cache
    assert_equal 2, cache.length
    assert_instance_of MlxLm::ArraysCache, cache[0]
    assert_instance_of MlxLm::ArraysCache, cache[1]
  end
end

class Phase45DenseLaneAURegistryTest < Minitest::Test
  def test_rwkv7_registered_and_resolve
    assert MlxLm::Models::REGISTRY.key?("rwkv7"), "rwkv7 should be registered"

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "rwkv7" })
    assert_equal MlxLm::Models::Rwkv7::Model, model_class
    assert_equal MlxLm::Models::Rwkv7::ModelArgs, args_class
  end
end
