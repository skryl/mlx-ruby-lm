$LOAD_PATH.unshift File.expand_path("../../lib", __dir__)
$LOAD_PATH.unshift File.expand_path("../../mlx-ruby/lib", __dir__)

require "mlx"
require "minitest/autorun"

require_relative "../../lib/mlx_lm/model_args"
require_relative "../../lib/mlx_lm/models"
require_relative "../../lib/mlx_lm/models/cache"
require_relative "../../lib/mlx_lm/models/activations"
require_relative "../../lib/mlx_lm/models/rope_utils"
require_relative "../../lib/mlx_lm/models/switch_layers"
require_relative "../../lib/mlx_lm/models/qwen3"
require_relative "../../lib/mlx_lm/models/qwen3_moe"
require_relative "../../lib/mlx_lm/models/qwen3_vl_moe"

class Qwen3MoeQwen3VlMoeModelsQwen3MoeTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_qwen3_moe_construct_forward_shape_and_sanitize_stacks_experts
    args = MlxLm::Models::Qwen3Moe::ModelArgs.from_dict({
      "model_type" => "qwen3_moe",
      "hidden_size" => 32,
      "num_hidden_layers" => 2,
      "intermediate_size" => 64,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 2,
      "rms_norm_eps" => 1e-5,
      "vocab_size" => 111,
      "head_dim" => 8,
      "rope_theta" => 10_000.0,
      "max_position_embeddings" => 256,
      "tie_word_embeddings" => true,
      "num_experts" => 2,
      "num_experts_per_tok" => 1,
      "decoder_sparse_step" => 1,
      "mlp_only_layers" => [],
      "moe_intermediate_size" => 16,
      "norm_topk_prob" => true,
    })

    model = MlxLm::Models::Qwen3Moe::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3, 4]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 4, 111], output.shape

    weights = {
      "lm_head.weight" => @mx.zeros([111, 32]).astype(@mx.float32),
      "model.layers.0.mlp.experts.0.up_proj.weight" => @mx.array([[1.0, 2.0], [3.0, 4.0]], dtype: @mx.float32),
      "model.layers.0.mlp.experts.1.up_proj.weight" => @mx.array([[5.0, 6.0], [7.0, 8.0]], dtype: @mx.float32),
      "model.layers.0.mlp.experts.0.down_proj.weight" => @mx.array([[9.0, 10.0], [11.0, 12.0]], dtype: @mx.float32),
      "model.layers.0.mlp.experts.1.down_proj.weight" => @mx.array([[13.0, 14.0], [15.0, 16.0]], dtype: @mx.float32),
      "model.layers.0.mlp.experts.0.gate_proj.weight" => @mx.array([[17.0, 18.0], [19.0, 20.0]], dtype: @mx.float32),
      "model.layers.0.mlp.experts.1.gate_proj.weight" => @mx.array([[21.0, 22.0], [23.0, 24.0]], dtype: @mx.float32),
      "model.layers.1.self_attn.q_proj.weight" => @mx.zeros([32, 32]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)
    up_stacked = sanitized["model.layers.0.mlp.switch_mlp.up_proj.weight"]
    @mx.eval(up_stacked)

    refute sanitized.key?("lm_head.weight")
    refute sanitized.key?("model.layers.0.mlp.experts.0.up_proj.weight")
    refute sanitized.key?("model.layers.0.mlp.experts.1.up_proj.weight")
    refute sanitized.key?("model.layers.0.mlp.experts.0.down_proj.weight")
    refute sanitized.key?("model.layers.0.mlp.experts.1.down_proj.weight")
    refute sanitized.key?("model.layers.0.mlp.experts.0.gate_proj.weight")
    refute sanitized.key?("model.layers.0.mlp.experts.1.gate_proj.weight")
    assert sanitized.key?("model.layers.0.mlp.switch_mlp.up_proj.weight")
    assert sanitized.key?("model.layers.0.mlp.switch_mlp.down_proj.weight")
    assert sanitized.key?("model.layers.0.mlp.switch_mlp.gate_proj.weight")
    assert sanitized.key?("model.layers.1.self_attn.q_proj.weight")

    assert_equal [2, 2, 2], up_stacked.shape
    assert_equal [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], up_stacked.to_a
  end
end

class Qwen3MoeQwen3VlMoeModelsQwen3VLMoeTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_qwen3_vl_moe_construct_forward_shape_and_sanitize_remaps_gate_up
    args = MlxLm::Models::Qwen3VLMoe::ModelArgs.from_dict({
      "model_type" => "qwen3_vl_moe",
      "text_config" => {
        "model_type" => "qwen3_moe",
        "hidden_size" => 24,
        "num_hidden_layers" => 2,
        "intermediate_size" => 48,
        "num_attention_heads" => 4,
        "num_key_value_heads" => 2,
        "rms_norm_eps" => 1e-5,
        "vocab_size" => 101,
        "head_dim" => 6,
        "rope_theta" => 10_000.0,
        "max_position_embeddings" => 256,
        "tie_word_embeddings" => false,
        "num_experts" => 2,
        "num_experts_per_tok" => 1,
        "decoder_sparse_step" => 1,
        "mlp_only_layers" => [],
        "moe_intermediate_size" => 12,
        "norm_topk_prob" => true,
      },
    })

    model = MlxLm::Models::Qwen3VLMoe::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 3, 101], output.shape

    gate_up = @mx.array((0...24).to_a, dtype: @mx.float32).reshape([2, 3, 4])
    down_proj = @mx.array((0...12).to_a, dtype: @mx.float32).reshape([2, 3, 2])
    weights = {
      "visual.encoder.weight" => @mx.zeros([2, 2]).astype(@mx.float32),
      "language_model.model.layers.0.mlp.experts.gate_up_proj" => gate_up,
      "language_model.model.layers.0.mlp.experts.down_proj" => down_proj,
      "language_model.lm_head.weight" => @mx.zeros([101, 24]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)
    gate_stacked = sanitized["language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"]
    up_stacked = sanitized["language_model.model.layers.0.mlp.switch_mlp.up_proj.weight"]
    down_stacked = sanitized["language_model.model.layers.0.mlp.switch_mlp.down_proj.weight"]
    @mx.eval(gate_stacked, up_stacked, down_stacked)

    refute sanitized.key?("visual.encoder.weight")
    refute sanitized.key?("language_model.model.layers.0.mlp.experts.gate_up_proj")
    refute sanitized.key?("language_model.model.layers.0.mlp.experts.down_proj")
    assert sanitized.key?("language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight")
    assert sanitized.key?("language_model.model.layers.0.mlp.switch_mlp.up_proj.weight")
    assert sanitized.key?("language_model.model.layers.0.mlp.switch_mlp.down_proj.weight")
    assert sanitized.key?("language_model.lm_head.weight")

    assert_equal [2, 2, 3], gate_stacked.shape
    assert_equal [2, 2, 3], up_stacked.shape
    assert_equal [2, 2, 3], down_stacked.shape
  end
end

class Qwen3MoeQwen3VlMoeModelsTest < Minitest::Test
  def test_models_registered_and_resolve
    assert MlxLm::Models::REGISTRY.key?("qwen3_moe"), "qwen3_moe should be registered"
    assert MlxLm::Models::REGISTRY.key?("qwen3_vl_moe"), "qwen3_vl_moe should be registered"

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "qwen3_moe" })
    assert_equal MlxLm::Models::Qwen3Moe::Model, model_class
    assert_equal MlxLm::Models::Qwen3Moe::ModelArgs, args_class

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "qwen3_vl_moe" })
    assert_equal MlxLm::Models::Qwen3VLMoe::Model, model_class
    assert_equal MlxLm::Models::Qwen3VLMoe::ModelArgs, args_class
  end
end
