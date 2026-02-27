$LOAD_PATH.unshift File.expand_path("../../lib", __dir__)
$LOAD_PATH.unshift File.expand_path("../../mlx-ruby/lib", __dir__)

require "mlx"
require "minitest/autorun"

require_relative "../../lib/mlx_lm/model_args"
require_relative "../../lib/mlx_lm/models"
require_relative "../../lib/mlx_lm/models/activations"
require_relative "../../lib/mlx_lm/models/llama4_text"
require_relative "../../lib/mlx_lm/models/plamo"

class Phase23DenseLaneACLlama4TextTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_llama4_text_construct_forward_shape_and_tied_output_path
    args_hash = {
      "model_type" => "llama4_text",
      "hidden_size" => 32,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 2,
      "num_hidden_layers" => 2,
      "vocab_size" => 97,
      "intermediate_size" => 64,
      "intermediate_size_mlp" => 64,
      "rms_norm_eps" => 1e-5,
      "rope_theta" => 10_000.0,
      "head_dim" => 8,
      "tie_word_embeddings" => true,
      "no_rope_layers" => [0, 1],
      "use_qk_norm" => true,
    }

    args = MlxLm::Models::Llama4Text::ModelArgs.from_dict(args_hash)
    model = MlxLm::Models::Llama4Text::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3, 4]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)

    assert_equal [1, 4, 97], output.shape
    assert_equal 2, model.layers.length
    assert_nil model.output

    untied_args = MlxLm::Models::Llama4Text::ModelArgs.from_dict(
      args_hash.merge("tie_word_embeddings" => false)
    )
    untied_model = MlxLm::Models::Llama4Text::Model.new(untied_args)
    assert_instance_of MLX::NN::Linear, untied_model.output
  end
end

class Phase23DenseLaneACPlamoTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_plamo_construct_forward_shape
    args = MlxLm::Models::Plamo::ModelArgs.from_dict({
      "model_type" => "plamo",
      "hidden_size" => 32,
      "num_hidden_layers" => 2,
      "intermediate_size" => 64,
      "num_attention_heads" => 4,
      "rms_norm_eps" => 1e-5,
      "vocab_size" => 101,
      "n_shared_head" => 2,
      "rope_theta" => 10_000.0,
      "rope_traditional" => false,
    })

    model = MlxLm::Models::Plamo::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)

    assert_equal [1, 3, 101], output.shape
    assert_equal 2, model.layers.length
  end
end

class Phase23DenseLaneACRegistryTest < Minitest::Test
  def test_models_registered_and_resolve
    assert MlxLm::Models::REGISTRY.key?("llama4_text"), "llama4_text should be registered"
    assert MlxLm::Models::REGISTRY.key?("plamo"), "plamo should be registered"

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "llama4_text" })
    assert_equal MlxLm::Models::Llama4Text::Model, model_class
    assert_equal MlxLm::Models::Llama4Text::ModelArgs, args_class

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "plamo" })
    assert_equal MlxLm::Models::Plamo::Model, model_class
    assert_equal MlxLm::Models::Plamo::ModelArgs, args_class
  end
end
