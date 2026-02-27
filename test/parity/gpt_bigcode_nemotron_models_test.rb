$LOAD_PATH.unshift File.expand_path("../../lib", __dir__)
$LOAD_PATH.unshift File.expand_path("../../mlx-ruby/lib", __dir__)

require "mlx"
require "minitest/autorun"

require_relative "../../lib/mlx_lm/model_args"
require_relative "../../lib/mlx_lm/models"
require_relative "../../lib/mlx_lm/models/gpt_bigcode"
require_relative "../../lib/mlx_lm/models/nemotron"

class GptBigcodeNemotronModelsTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_gpt_bigcode_construct_forward_shape_and_multi_query_kv_heads
    args = MlxLm::Models::GPTBigCode::ModelArgs.from_dict({
      "model_type" => "gpt_bigcode",
      "n_embd" => 64,
      "n_layer" => 2,
      "n_inner" => 128,
      "n_head" => 4,
      "n_positions" => 64,
      "layer_norm_epsilon" => 1e-5,
      "vocab_size" => 96,
      "multi_query" => true,
      "tie_word_embeddings" => true,
    })

    model = MlxLm::Models::GPTBigCode::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    input = @mx.array([[1, 2, 3, 4]], dtype: @mx.int32)
    output = model.call(input)
    @mx.eval(output)

    assert_equal [1, 4, 96], output.shape
    assert_equal 1, model.layers.first.attn.instance_variable_get(:@n_kv_heads)
  end
end

class GptBigcodeNemotronModelsNemotronConstructForwardShapeAndLinearRopeScaleTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_nemotron_construct_forward_shape_and_linear_rope_scale
    args = MlxLm::Models::Nemotron::ModelArgs.from_dict({
      "model_type" => "nemotron",
      "hidden_size" => 64,
      "hidden_act" => "relu2",
      "num_hidden_layers" => 2,
      "intermediate_size" => 128,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 2,
      "norm_eps" => 1e-5,
      "vocab_size" => 80,
      "partial_rotary_factor" => 0.5,
      "rope_theta" => 10_000.0,
      "rope_scaling" => { "type" => "linear", "factor" => 2.0 },
      "tie_word_embeddings" => false,
    })

    model = MlxLm::Models::Nemotron::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    input = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(input)
    @mx.eval(output)

    assert_equal [1, 3, 80], output.shape
    assert_in_delta 0.5, model.layers.first.self_attn.rope.instance_variable_get(:@scale), 1e-12
  end
end

class GptBigcodeNemotronModelsModelsRegisteredTest < Minitest::Test
  def test_models_registered
    assert MlxLm::Models::REGISTRY.key?("gpt_bigcode"), "gpt_bigcode should be registered"
    assert MlxLm::Models::REGISTRY.key?("nemotron"), "nemotron should be registered"

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "gpt_bigcode" })
    assert_equal MlxLm::Models::GPTBigCode::Model, model_class
    assert_equal MlxLm::Models::GPTBigCode::ModelArgs, args_class

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "nemotron" })
    assert_equal MlxLm::Models::Nemotron::Model, model_class
    assert_equal MlxLm::Models::Nemotron::ModelArgs, args_class
  end
end
