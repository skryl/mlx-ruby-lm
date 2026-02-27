require_relative "../test_helper"

class OlmoSeedOssModelsTest < Minitest::Test
  include ParityTestHelpers

  def setup
    @mx = MLX::Core
  end

  def test_olmo_construct_and_forward_shape
    args = MlxLm::Models::OLMo::ModelArgs.from_dict({
      "model_type" => "olmo",
      "d_model" => 64,
      "n_layers" => 2,
      "mlp_hidden_size" => 128,
      "n_heads" => 2,
      "vocab_size" => 128,
      "embedding_size" => 128,
      "weight_tying" => false,
    })

    model = MlxLm::Models::OLMo::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    input = @mx.array([[1, 2, 3]], @mx.int32)
    output = model.call(input)
    @mx.eval(output)

    assert_equal [1, 3, 128], output.shape
  end

  def test_seed_oss_construct_and_forward_shape
    args = MlxLm::Models::SeedOSS::ModelArgs.from_dict({
      "model_type" => "seed_oss",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "intermediate_size" => 128,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "head_dim" => 32,
      "rms_norm_eps" => 1e-6,
      "vocab_size" => 128,
      "tie_word_embeddings" => true,
    })

    model = MlxLm::Models::SeedOSS::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    input = @mx.array([[1, 2, 3]], @mx.int32)
    output = model.call(input)
    @mx.eval(output)

    assert_equal [1, 3, 128], output.shape
  end

  def test_olmo_registered
    assert MlxLm::Models::REGISTRY.key?("olmo"), "olmo should be registered"
    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "olmo" })
    assert_equal MlxLm::Models::OLMo::Model, model_class
    assert_equal MlxLm::Models::OLMo::ModelArgs, args_class
  end

  def test_seed_oss_registered
    assert MlxLm::Models::REGISTRY.key?("seed_oss"), "seed_oss should be registered"
    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "seed_oss" })
    assert_equal MlxLm::Models::SeedOSS::Model, model_class
    assert_equal MlxLm::Models::SeedOSS::ModelArgs, args_class
  end
end
