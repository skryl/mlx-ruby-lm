$LOAD_PATH.unshift File.expand_path("../../lib", __dir__)
$LOAD_PATH.unshift File.expand_path("../../mlx-ruby/lib", __dir__)

require "mlx"
require "minitest/autorun"

require_relative "../../lib/mlx_lm/model_args"
require_relative "../../lib/mlx_lm/models"
require_relative "../../lib/mlx_lm/models/activations"
require_relative "../../lib/mlx_lm/models/glm"
require_relative "../../lib/mlx_lm/models/helium"

class GlmHeliumModelsTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_glm_instantiation_forward_shape_and_sanitize
    args = MlxLm::Models::GLM::ModelArgs.from_dict({
      "model_type" => "glm",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "intermediate_size" => 128,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 2,
      "rms_norm_eps" => 1e-5,
      "vocab_size" => 128,
      "head_dim" => 16,
      "attention_bias" => false,
      "tie_word_embeddings" => true,
    })
    model = MlxLm::Models::GLM::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    input = @mx.array([[1, 2, 3, 4]], dtype: @mx.int32)
    output = model.call(input)
    @mx.eval(output)

    assert_equal [1, 4, 128], output.shape
    assert MlxLm::Models::REGISTRY.key?("glm")

    sanitized = model.sanitize({
      "lm_head.weight" => 1,
      "model.layers.0.self_attn.rotary_emb.inv_freq" => 2,
      "model.embed_tokens.weight" => 3,
    })
    refute sanitized.key?("lm_head.weight")
    refute sanitized.key?("model.layers.0.self_attn.rotary_emb.inv_freq")
    assert_equal 3, sanitized["model.embed_tokens.weight"]
  end
end

class GlmHeliumModelsHeliumInstantiationForwardShapeAndMlpBiasTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_helium_instantiation_forward_shape_and_mlp_bias
    args = MlxLm::Models::Helium::ModelArgs.from_dict({
      "model_type" => "helium",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "intermediate_size" => 128,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 2,
      "rms_norm_eps" => 1e-5,
      "vocab_size" => 96,
      "head_dim" => 8,
      "attention_bias" => false,
      "mlp_bias" => true,
      "tie_word_embeddings" => false,
      "rope_theta" => 1000.0,
    })
    model = MlxLm::Models::Helium::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    input = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(input)
    @mx.eval(output)

    assert_equal [1, 3, 96], output.shape
    assert MlxLm::Models::REGISTRY.key?("helium")

    mlp = model.layers[0].mlp
    assert mlp.gate_proj.state.key?("bias")
    assert mlp.up_proj.state.key?("bias")
    assert mlp.down_proj.state.key?("bias")
  end
end
