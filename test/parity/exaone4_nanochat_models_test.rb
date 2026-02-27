require_relative "../test_helper"
require_relative "../../lib/mlx_lm/models/exaone4"
require_relative "../../lib/mlx_lm/models/nanochat"

class Exaone4NanochatModelsExaone4Test < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_exaone4_construct_forward_shape_and_cache_pattern
    args = MlxLm::Models::Exaone4::ModelArgs.from_dict({
      "model_type" => "exaone4",
      "hidden_size" => 64,
      "num_hidden_layers" => 4,
      "intermediate_size" => 128,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 2,
      "rms_norm_eps" => 1e-5,
      "vocab_size" => 96,
      "max_position_embeddings" => 256,
      "rope_theta" => 10_000.0,
      "head_dim" => 16,
      "tie_word_embeddings" => false,
      "sliding_window" => 4,
      "sliding_window_pattern" => "LLGL",
    })

    model = MlxLm::Models::Exaone4::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    tokens = @mx.array([[1, 2, 3, 4]], @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)

    assert_equal [1, 4, 96], output.shape

    caches = model.make_cache
    assert_equal 4, caches.length
    assert_instance_of MlxLm::RotatingKVCache, caches[0]
    assert_instance_of MlxLm::RotatingKVCache, caches[1]
    assert_instance_of MlxLm::KVCache, caches[2]
    assert_instance_of MlxLm::RotatingKVCache, caches[3]
  end
end

class Exaone4NanochatModelsTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_nanochat_construct_forward_shape_and_softcap
    args = MlxLm::Models::Nanochat::ModelArgs.from_dict({
      "model_type" => "nanochat",
      "hidden_size" => 40,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 5,
      "num_key_value_heads" => 5,
      "vocab_size" => 64,
      "intermediate_size" => 80,
      "rope_theta" => 10_000.0,
    })

    model = MlxLm::Models::Nanochat::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    tokens = @mx.array([[1, 2, 3]], @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)

    assert_equal [1, 3, 64], output.shape

    values = output.to_a.flatten
    assert values.all? { |v| v <= 15.0 + 1e-5 && v >= -15.0 - 1e-5 }, "nanochat logits should be softcapped to [-15, 15]"
  end
end

class Exaone4NanochatModelsModelsRegisteredTest < Minitest::Test
  def test_models_registered
    assert MlxLm::Models::REGISTRY.key?("exaone4"), "exaone4 should be registered"
    assert MlxLm::Models::REGISTRY.key?("nanochat"), "nanochat should be registered"

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "exaone4" })
    assert_equal MlxLm::Models::Exaone4::Model, model_class
    assert_equal MlxLm::Models::Exaone4::ModelArgs, args_class

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "nanochat" })
    assert_equal MlxLm::Models::Nanochat::Model, model_class
    assert_equal MlxLm::Models::Nanochat::ModelArgs, args_class
  end
end
