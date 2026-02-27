$LOAD_PATH.unshift File.expand_path("../../lib", __dir__)
$LOAD_PATH.unshift File.expand_path("../../mlx-ruby/lib", __dir__)

require "mlx"
require "minitest/autorun"

require_relative "../../lib/mlx_lm/model_args"
require_relative "../../lib/mlx_lm/models"
require_relative "../../lib/mlx_lm/models/activations"
require_relative "../../lib/mlx_lm/models/rope_utils"
require_relative "../../lib/mlx_lm/models/granite"
require_relative "../../lib/mlx_lm/models/minicpm"

class GraniteMinicpmModelsTest < Minitest::Test
  def test_registration_entries_for_lane_g_models
    assert MlxLm::Models::REGISTRY.key?("granite"), "granite should be registered"
    assert MlxLm::Models::REGISTRY.key?("minicpm"), "minicpm should be registered"

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "granite" })
    assert_equal MlxLm::Models::Granite::Model, model_class
    assert_equal MlxLm::Models::Granite::ModelArgs, args_class

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "minicpm" })
    assert_equal MlxLm::Models::MiniCPM::Model, model_class
    assert_equal MlxLm::Models::MiniCPM::ModelArgs, args_class
  end
end

class GraniteMinicpmModelsGraniteConstructForwardShapeAndScalingFieldsTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_granite_construct_forward_shape_and_scaling_fields
    args = MlxLm::Models::Granite::ModelArgs.from_dict({
      "model_type" => "granite",
      "hidden_size" => 32,
      "num_hidden_layers" => 2,
      "intermediate_size" => 64,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "rms_norm_eps" => 1e-5,
      "vocab_size" => 128,
      "logits_scaling" => 2.0,
      "attention_multiplier" => 0.25,
      "embedding_multiplier" => 1.5,
      "residual_multiplier" => 0.75,
      "max_position_embeddings" => 256,
      "attention_bias" => false,
      "mlp_bias" => false,
      "rope_theta" => 10_000.0,
      "tie_word_embeddings" => true,
    })
    model = MlxLm::Models::Granite::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    input = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(input)
    @mx.eval(output)

    assert_equal [1, 3, 128], output.shape
    assert_in_delta 2.0, model.instance_variable_get(:@logits_scaling), 1e-12
    assert_in_delta 1.5, model.model.instance_variable_get(:@embedding_multiplier), 1e-12
    assert_in_delta 0.75, model.layers.first.instance_variable_get(:@residual_multiplier), 1e-12
    assert_in_delta 0.25, model.layers.first.self_attn.instance_variable_get(:@scale), 1e-12
  end
end

class GraniteMinicpmModelsMinicpmConstructForwardShapeAndDepthScalingTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_minicpm_construct_forward_shape_and_depth_scaling
    args = MlxLm::Models::MiniCPM::ModelArgs.from_dict({
      "model_type" => "minicpm",
      "hidden_size" => 32,
      "dim_model_base" => 16,
      "num_hidden_layers" => 2,
      "intermediate_size" => 64,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "rms_norm_eps" => 1e-5,
      "vocab_size" => 96,
      "scale_depth" => 1.4,
      "scale_emb" => 8.0,
      "max_position_embeddings" => 256,
      "rope_theta" => 10_000.0,
      "tie_word_embeddings" => false,
    })
    model = MlxLm::Models::MiniCPM::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    input = @mx.array([[1, 2, 3, 4]], dtype: @mx.int32)
    output = model.call(input)
    @mx.eval(output)

    assert_equal [1, 4, 96], output.shape
    expected_residual_scale = args.scale_depth / Math.sqrt(args.num_hidden_layers)
    assert_in_delta expected_residual_scale, model.layers.first.instance_variable_get(:@residual_scale), 1e-12
  end

  def test_minicpm_sanitize_adds_missing_lm_head_weight
    args = MlxLm::Models::MiniCPM::ModelArgs.from_dict({
      "model_type" => "minicpm",
      "hidden_size" => 32,
      "dim_model_base" => 16,
      "num_hidden_layers" => 1,
      "intermediate_size" => 64,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "rms_norm_eps" => 1e-5,
      "vocab_size" => 96,
      "scale_depth" => 1.0,
      "scale_emb" => 1.0,
      "tie_word_embeddings" => false,
    })
    model = MlxLm::Models::MiniCPM::Model.new(args)
    embed_weight = @mx.zeros([96, 32]).astype(@mx.float32)
    weights = { "model.embed_tokens.weight" => embed_weight }

    sanitized = model.sanitize(weights)

    assert sanitized.key?("lm_head.weight")
    assert_same embed_weight, sanitized["lm_head.weight"]
  end
end
