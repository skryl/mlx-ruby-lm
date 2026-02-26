$LOAD_PATH.unshift File.expand_path("../../lib", __dir__)
$LOAD_PATH.unshift File.expand_path("../../mlx-ruby/lib", __dir__)

require "mlx"
require "minitest/autorun"

require_relative "../../lib/mlx_lm/model_args"
require_relative "../../lib/mlx_lm/models"
require_relative "../../lib/mlx_lm/models/activations"
require_relative "../../lib/mlx_lm/models/rope_utils"
require_relative "../../lib/mlx_lm/models/switch_layers"
require_relative "../../lib/mlx_lm/models/qwen2"
require_relative "../../lib/mlx_lm/models/qwen2_moe"
require_relative "../../lib/mlx_lm/models/phimoe"

class Phase21DenseLaneWQwen2MoeTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_qwen2_moe_construct_forward_shape_and_sanitize_stacks_experts
    args = MlxLm::Models::Qwen2Moe::ModelArgs.from_dict({
      "model_type" => "qwen2_moe",
      "hidden_size" => 32,
      "num_hidden_layers" => 2,
      "intermediate_size" => 64,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 2,
      "rms_norm_eps" => 1e-5,
      "vocab_size" => 109,
      "rope_theta" => 10_000.0,
      "max_position_embeddings" => 256,
      "tie_word_embeddings" => false,
      "num_experts_per_tok" => 1,
      "num_experts" => 2,
      "moe_intermediate_size" => 16,
      "shared_expert_intermediate_size" => 24,
    })

    model = MlxLm::Models::Qwen2Moe::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3, 4]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 4, 109], output.shape

    weights = {
      "model.layers.0.mlp.experts.0.up_proj.weight" => @mx.array([[1.0, 2.0], [3.0, 4.0]], dtype: @mx.float32),
      "model.layers.0.mlp.experts.1.up_proj.weight" => @mx.array([[5.0, 6.0], [7.0, 8.0]], dtype: @mx.float32),
      "model.layers.0.mlp.experts.0.down_proj.weight" => @mx.array([[9.0, 10.0], [11.0, 12.0]], dtype: @mx.float32),
      "model.layers.0.mlp.experts.1.down_proj.weight" => @mx.array([[13.0, 14.0], [15.0, 16.0]], dtype: @mx.float32),
      "model.layers.0.mlp.experts.0.gate_proj.weight" => @mx.array([[17.0, 18.0], [19.0, 20.0]], dtype: @mx.float32),
      "model.layers.0.mlp.experts.1.gate_proj.weight" => @mx.array([[21.0, 22.0], [23.0, 24.0]], dtype: @mx.float32),
      "model.layers.0.mlp.experts.0.up_proj.scales" => @mx.array([0.5, 1.0], dtype: @mx.float32),
      "model.layers.0.mlp.experts.1.up_proj.scales" => @mx.array([1.5, 2.0], dtype: @mx.float32),
      "model.layers.1.self_attn.q_proj.weight" => @mx.zeros([32, 32]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)
    up_stacked = sanitized["model.layers.0.mlp.switch_mlp.up_proj.weight"]
    scales_stacked = sanitized["model.layers.0.mlp.switch_mlp.up_proj.scales"]
    @mx.eval(up_stacked, scales_stacked)

    refute sanitized.key?("model.layers.0.mlp.experts.0.up_proj.weight")
    refute sanitized.key?("model.layers.0.mlp.experts.1.up_proj.weight")
    refute sanitized.key?("model.layers.0.mlp.experts.0.down_proj.weight")
    refute sanitized.key?("model.layers.0.mlp.experts.1.down_proj.weight")
    refute sanitized.key?("model.layers.0.mlp.experts.0.gate_proj.weight")
    refute sanitized.key?("model.layers.0.mlp.experts.1.gate_proj.weight")
    refute sanitized.key?("model.layers.0.mlp.experts.0.up_proj.scales")
    refute sanitized.key?("model.layers.0.mlp.experts.1.up_proj.scales")
    assert sanitized.key?("model.layers.0.mlp.switch_mlp.up_proj.weight")
    assert sanitized.key?("model.layers.0.mlp.switch_mlp.down_proj.weight")
    assert sanitized.key?("model.layers.0.mlp.switch_mlp.gate_proj.weight")
    assert sanitized.key?("model.layers.0.mlp.switch_mlp.up_proj.scales")
    assert sanitized.key?("model.layers.1.self_attn.q_proj.weight")

    assert_equal [2, 2, 2], up_stacked.shape
    assert_equal [2, 2], scales_stacked.shape
    assert_equal [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], up_stacked.to_a
    assert_equal [[0.5, 1.0], [1.5, 2.0]], scales_stacked.to_a
  end
end

class Phase21DenseLaneWPhiMoeTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_phimoe_construct_forward_shape_and_sanitize_stacks_experts
    args = MlxLm::Models::PhiMoe::ModelArgs.from_dict({
      "model_type" => "phimoe",
      "vocab_size" => 113,
      "hidden_size" => 32,
      "intermediate_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 2,
      "max_position_embeddings" => 256,
      "original_max_position_embeddings" => 64,
      "rms_norm_eps" => 1e-5,
      "num_local_experts" => 2,
      "num_experts_per_tok" => 1,
      "rope_theta" => 10_000.0,
      "rope_scaling" => {
        "short_factor" => [1.0, 1.0, 1.0, 1.0],
        "long_factor" => [1.0, 1.0, 1.0, 1.0],
        "short_mscale" => 1.0,
        "long_mscale" => 1.0,
      },
    })

    model = MlxLm::Models::PhiMoe::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 3, 113], output.shape

    weights = {
      "model.layers.0.block_sparse_moe.experts.0.w1.weight" => @mx.array([[1.0, 2.0], [3.0, 4.0]], dtype: @mx.float32),
      "model.layers.0.block_sparse_moe.experts.1.w1.weight" => @mx.array([[5.0, 6.0], [7.0, 8.0]], dtype: @mx.float32),
      "model.layers.0.block_sparse_moe.experts.0.w2.weight" => @mx.array([[9.0, 10.0], [11.0, 12.0]], dtype: @mx.float32),
      "model.layers.0.block_sparse_moe.experts.1.w2.weight" => @mx.array([[13.0, 14.0], [15.0, 16.0]], dtype: @mx.float32),
      "model.layers.0.block_sparse_moe.experts.0.w3.weight" => @mx.array([[17.0, 18.0], [19.0, 20.0]], dtype: @mx.float32),
      "model.layers.0.block_sparse_moe.experts.1.w3.weight" => @mx.array([[21.0, 22.0], [23.0, 24.0]], dtype: @mx.float32),
      "model.layers.0.block_sparse_moe.experts.0.w1.scales" => @mx.array([0.25, 0.5], dtype: @mx.float32),
      "model.layers.0.block_sparse_moe.experts.1.w1.scales" => @mx.array([0.75, 1.0], dtype: @mx.float32),
      "model.layers.1.self_attn.q_proj.weight" => @mx.zeros([32, 32]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)
    gate_stacked = sanitized["model.layers.0.block_sparse_moe.switch_mlp.gate_proj.weight"]
    scales_stacked = sanitized["model.layers.0.block_sparse_moe.switch_mlp.gate_proj.scales"]
    @mx.eval(gate_stacked, scales_stacked)

    refute sanitized.key?("model.layers.0.block_sparse_moe.experts.0.w1.weight")
    refute sanitized.key?("model.layers.0.block_sparse_moe.experts.1.w1.weight")
    refute sanitized.key?("model.layers.0.block_sparse_moe.experts.0.w2.weight")
    refute sanitized.key?("model.layers.0.block_sparse_moe.experts.1.w2.weight")
    refute sanitized.key?("model.layers.0.block_sparse_moe.experts.0.w3.weight")
    refute sanitized.key?("model.layers.0.block_sparse_moe.experts.1.w3.weight")
    refute sanitized.key?("model.layers.0.block_sparse_moe.experts.0.w1.scales")
    refute sanitized.key?("model.layers.0.block_sparse_moe.experts.1.w1.scales")
    assert sanitized.key?("model.layers.0.block_sparse_moe.switch_mlp.gate_proj.weight")
    assert sanitized.key?("model.layers.0.block_sparse_moe.switch_mlp.down_proj.weight")
    assert sanitized.key?("model.layers.0.block_sparse_moe.switch_mlp.up_proj.weight")
    assert sanitized.key?("model.layers.0.block_sparse_moe.switch_mlp.gate_proj.scales")
    assert sanitized.key?("model.layers.1.self_attn.q_proj.weight")

    assert_equal [2, 2, 2], gate_stacked.shape
    assert_equal [2, 2], scales_stacked.shape
    assert_equal [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], gate_stacked.to_a
    assert_equal [[0.25, 0.5], [0.75, 1.0]], scales_stacked.to_a
  end
end

class Phase21DenseLaneWRegistryTest < Minitest::Test
  def test_models_registered_and_resolve
    assert MlxLm::Models::REGISTRY.key?("qwen2_moe"), "qwen2_moe should be registered"
    assert MlxLm::Models::REGISTRY.key?("phimoe"), "phimoe should be registered"

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "qwen2_moe" })
    assert_equal MlxLm::Models::Qwen2Moe::Model, model_class
    assert_equal MlxLm::Models::Qwen2Moe::ModelArgs, args_class

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "phimoe" })
    assert_equal MlxLm::Models::PhiMoe::Model, model_class
    assert_equal MlxLm::Models::PhiMoe::ModelArgs, args_class
  end
end
