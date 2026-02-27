$LOAD_PATH.unshift File.expand_path("../../lib", __dir__)
$LOAD_PATH.unshift File.expand_path("../../mlx-ruby/lib", __dir__)

require "mlx"
require "minitest/autorun"

require_relative "../../lib/mlx_lm/model_args"
require_relative "../../lib/mlx_lm/models"
require_relative "../../lib/mlx_lm/models/cache"
require_relative "../../lib/mlx_lm/models/activations"
require_relative "../../lib/mlx_lm/models/rope_utils"
require_relative "../../lib/mlx_lm/models/olmo3"
require_relative "../../lib/mlx_lm/models/gpt2"

class Olmo3Gpt2ModelsTest < Minitest::Test
  def test_registry_entries_for_lane_i_models
    assert MlxLm::Models::REGISTRY.key?("olmo3"), "olmo3 should be registered"
    assert MlxLm::Models::REGISTRY.key?("gpt2"), "gpt2 should be registered"

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "olmo3" })
    assert_equal MlxLm::Models::OLMo3::Model, model_class
    assert_equal MlxLm::Models::OLMo3::ModelArgs, args_class

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "gpt2" })
    assert_equal MlxLm::Models::GPT2::Model, model_class
    assert_equal MlxLm::Models::GPT2::ModelArgs, args_class
  end
end

class Olmo3Gpt2ModelsMo3Test < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_olmo3_construct_forward_shape_and_cache_types
    args = MlxLm::Models::OLMo3::ModelArgs.from_dict({
      "model_type" => "olmo3",
      "hidden_size" => 48,
      "num_hidden_layers" => 4,
      "intermediate_size" => 96,
      "num_attention_heads" => 4,
      "rms_norm_eps" => 1e-5,
      "vocab_size" => 128,
      "max_position_embeddings" => 256,
      "sliding_window" => 4,
      "rope_theta" => 10_000.0,
      "tie_word_embeddings" => false,
    })

    model = MlxLm::Models::OLMo3::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    tokens = @mx.array([[1, 2, 3, 4]], @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 4, 128], output.shape

    caches = model.make_cache
    assert_equal 4, caches.length
    assert_instance_of MlxLm::RotatingKVCache, caches[0]
    assert_instance_of MlxLm::RotatingKVCache, caches[1]
    assert_instance_of MlxLm::RotatingKVCache, caches[2]
    assert_instance_of MlxLm::KVCache, caches[3]
  end
end

class Olmo3Gpt2ModelsT2Test < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_gpt2_construct_and_forward_shape
    args = MlxLm::Models::GPT2::ModelArgs.from_dict({
      "model_type" => "gpt2",
      "n_ctx" => 16,
      "n_embd" => 32,
      "n_head" => 4,
      "n_layer" => 2,
      "n_positions" => 16,
      "layer_norm_epsilon" => 1e-5,
      "vocab_size" => 96,
    })

    model = MlxLm::Models::GPT2::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    tokens = @mx.array([[1, 2, 3]], @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 3, 96], output.shape
  end
end
