$LOAD_PATH.unshift File.expand_path("../../lib", __dir__)
$LOAD_PATH.unshift File.expand_path("../../mlx-ruby/lib", __dir__)

require "mlx"
require "minitest/autorun"

require_relative "../../lib/mlx_lm/model_args"
require_relative "../../lib/mlx_lm/models"
require_relative "../../lib/mlx_lm/models/activations"
require_relative "../../lib/mlx_lm/models/rope_utils"
require_relative "../../lib/mlx_lm/models/qwen3"
require_relative "../../lib/mlx_lm/models/lfm2"
require_relative "../../lib/mlx_lm/models/lfm2_vl"

class Lfm2Lfm2VlModelsLfm2Test < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_lfm2_construct_forward_shape_and_sanitize_conv_transpose
    args = MlxLm::Models::Lfm2::ModelArgs.from_dict({
      "model_type" => "lfm2",
      "vocab_size" => 97,
      "hidden_size" => 32,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 2,
      "max_position_embeddings" => 128,
      "norm_eps" => 1e-5,
      "conv_bias" => false,
      "conv_L_cache" => 3,
      "block_dim" => 32,
      "block_ff_dim" => 64,
      "block_multiple_of" => 8,
      "block_ffn_dim_multiplier" => 1.5,
      "block_auto_adjust_ff_dim" => true,
      "layer_types" => ["full_attention", "conv"],
      "rope_parameters" => { "rope_theta" => 10_000.0 },
    })

    model = MlxLm::Models::Lfm2::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3, 4]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 4, 97], output.shape

    conv_weight = @mx.array((0...24).to_a, dtype: @mx.float32).reshape([2, 3, 4])
    weights = {
      "model.layers.0.conv.weight" => conv_weight,
      "model.layers.0.self_attn.q_proj.weight" => @mx.zeros([32, 32]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)
    sanitized_conv = sanitized["model.layers.0.conv.weight"]
    @mx.eval(sanitized_conv)

    assert_equal [2, 4, 3], sanitized_conv.shape
    assert_equal @mx.swapaxes(conv_weight, 1, 2).to_a, sanitized_conv.to_a
    assert sanitized.key?("model.layers.0.self_attn.q_proj.weight")
  end
end

class Lfm2Lfm2VlModelsLfm2VlTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_lfm2_vl_construct_forward_shape_and_sanitize_multimodal_key_removal
    args = MlxLm::Models::Lfm2VL::ModelArgs.from_dict({
      "model_type" => "lfm2-vl",
      "text_config" => {
        "model_type" => "lfm2",
        "vocab_size" => 101,
        "hidden_size" => 32,
        "num_hidden_layers" => 2,
        "num_attention_heads" => 4,
        "num_key_value_heads" => 2,
        "max_position_embeddings" => 128,
        "norm_eps" => 1e-5,
        "conv_bias" => false,
        "conv_L_cache" => 3,
        "block_dim" => 32,
        "block_ff_dim" => 64,
        "block_multiple_of" => 8,
        "block_ffn_dim_multiplier" => 1.5,
        "block_auto_adjust_ff_dim" => true,
        "layer_types" => ["full_attention", "conv"],
        "rope_theta" => 10_000.0,
      },
    })

    model = MlxLm::Models::Lfm2VL::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 3, 101], output.shape

    weights = {
      "vision_tower.patch_embed.weight" => @mx.zeros([1]).astype(@mx.float32),
      "multi_modal_projector.linear.weight" => @mx.zeros([1]).astype(@mx.float32),
      "language_model.model.embed_tokens.weight" => @mx.zeros([101, 32]).astype(@mx.float32),
      "language_model.model.layers.0.self_attn.q_proj.weight" => @mx.zeros([32, 32]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)
    refute sanitized.key?("vision_tower.patch_embed.weight")
    refute sanitized.key?("multi_modal_projector.linear.weight")
    assert sanitized.key?("language_model.model.embed_tokens.weight")
    assert sanitized.key?("language_model.model.layers.0.self_attn.q_proj.weight")
  end
end

class Lfm2Lfm2VlModelsTest < Minitest::Test
  def test_models_registered_and_resolve
    assert MlxLm::Models::REGISTRY.key?("lfm2"), "lfm2 should be registered"
    assert MlxLm::Models::REGISTRY.key?("lfm2-vl"), "lfm2-vl should be registered"

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "lfm2" })
    assert_equal MlxLm::Models::Lfm2::Model, model_class
    assert_equal MlxLm::Models::Lfm2::ModelArgs, args_class

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "lfm2-vl" })
    assert_equal MlxLm::Models::Lfm2VL::Model, model_class
    assert_equal MlxLm::Models::Lfm2VL::ModelArgs, args_class
  end
end
