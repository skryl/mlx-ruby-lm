$LOAD_PATH.unshift File.expand_path("../../lib", __dir__)
$LOAD_PATH.unshift File.expand_path("../../mlx-ruby/lib", __dir__)

require "mlx"
require "minitest/autorun"

require_relative "../../lib/mlx_lm/model_args"
require_relative "../../lib/mlx_lm/models"
require_relative "../../lib/mlx_lm/models/cache"
require_relative "../../lib/mlx_lm/models/activations"
require_relative "../../lib/mlx_lm/models/rope_utils"
require_relative "../../lib/mlx_lm/models/qwen3"
require_relative "../../lib/mlx_lm/models/qwen3_5"
require_relative "../../lib/mlx_lm/models/qwen3_5_moe"

class Qwen35Qwen35MoeModelsQwen35Test < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_qwen3_5_construct_forward_shape_and_sanitize_key_remap
    args = MlxLm::Models::Qwen35::ModelArgs.from_dict({
      "model_type" => "qwen3_5",
      "text_config" => {
        "model_type" => "qwen3_5",
        "hidden_size" => 32,
        "num_hidden_layers" => 2,
        "intermediate_size" => 64,
        "num_attention_heads" => 4,
        "num_key_value_heads" => 2,
        "rms_norm_eps" => 1e-5,
        "vocab_size" => 113,
        "head_dim" => 8,
        "max_position_embeddings" => 256,
        "rope_theta" => 10_000.0,
        "tie_word_embeddings" => false,
      },
    })

    model = MlxLm::Models::Qwen35::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)

    assert_equal [1, 3, 113], output.shape

    weights = {
      "model.language_model.embed_tokens.weight" => @mx.zeros([113, 32]).astype(@mx.float32),
      "model.language_model.layers.0.self_attn.q_proj.weight" => @mx.zeros([32, 32]).astype(@mx.float32),
      "language_model.model.norm.weight" => @mx.zeros([32]).astype(@mx.float32),
      "lm_head.weight" => @mx.zeros([113, 32]).astype(@mx.float32),
      "model.visual.encoder.weight" => @mx.zeros([2, 2]).astype(@mx.float32),
    }
    sanitized = model.sanitize(weights)

    refute sanitized.key?("model.visual.encoder.weight")
    assert sanitized.key?("language_model.model.embed_tokens.weight")
    assert sanitized.key?("language_model.model.layers.0.self_attn.q_proj.weight")
    assert sanitized.key?("language_model.model.norm.weight")
    assert sanitized.key?("language_model.lm_head.weight")
    assert sanitized.keys.all? { |k| k.start_with?("language_model.") }
  end
end

class Qwen35Qwen35MoeModelsQwen35MoeTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_qwen3_5_moe_construct_forward_shape_and_sanitize_moe_key_remap
    args = MlxLm::Models::Qwen35Moe::ModelArgs.from_dict({
      "model_type" => "qwen3_5_moe",
      "text_config" => {
        "model_type" => "qwen3_5_moe",
        "hidden_size" => 32,
        "num_hidden_layers" => 2,
        "intermediate_size" => 64,
        "num_attention_heads" => 4,
        "num_key_value_heads" => 2,
        "rms_norm_eps" => 1e-5,
        "vocab_size" => 109,
        "head_dim" => 8,
        "max_position_embeddings" => 256,
        "rope_theta" => 10_000.0,
        "tie_word_embeddings" => false,
        "num_experts" => 2,
        "num_experts_per_tok" => 1,
      },
    })

    model = MlxLm::Models::Qwen35Moe::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    tokens = @mx.array([[1, 2, 3, 4]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)

    assert_equal [1, 4, 109], output.shape

    gate_up = @mx.array((0...24).to_a, dtype: @mx.float32).reshape([6, 4])
    down_proj = @mx.zeros([3, 4]).astype(@mx.float32)
    weights = {
      "model.language_model.layers.0.mlp.experts.gate_up_proj" => gate_up,
      "model.language_model.layers.0.mlp.experts.down_proj" => down_proj,
      "model.visual.patch_embed.weight" => @mx.zeros([2, 2]).astype(@mx.float32),
    }
    sanitized = model.sanitize(weights)

    refute sanitized.key?("model.visual.patch_embed.weight")
    refute sanitized.key?("language_model.model.layers.0.mlp.experts.gate_up_proj")
    refute sanitized.key?("language_model.model.layers.0.mlp.experts.down_proj")
    assert sanitized.key?("language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight")
    assert sanitized.key?("language_model.model.layers.0.mlp.switch_mlp.up_proj.weight")
    assert sanitized.key?("language_model.model.layers.0.mlp.switch_mlp.down_proj.weight")

    assert_equal [3, 4], sanitized["language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"].shape
    assert_equal [3, 4], sanitized["language_model.model.layers.0.mlp.switch_mlp.up_proj.weight"].shape
    assert_equal [3, 4], sanitized["language_model.model.layers.0.mlp.switch_mlp.down_proj.weight"].shape
  end
end

class Qwen35Qwen35MoeModelsTest < Minitest::Test
  def test_models_registered_and_resolve
    assert MlxLm::Models::REGISTRY.key?("qwen3_5"), "qwen3_5 should be registered"
    assert MlxLm::Models::REGISTRY.key?("qwen3_5_moe"), "qwen3_5_moe should be registered"

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "qwen3_5" })
    assert_equal MlxLm::Models::Qwen35::Model, model_class
    assert_equal MlxLm::Models::Qwen35::ModelArgs, args_class

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "qwen3_5_moe" })
    assert_equal MlxLm::Models::Qwen35Moe::Model, model_class
    assert_equal MlxLm::Models::Qwen35Moe::ModelArgs, args_class
  end
end
