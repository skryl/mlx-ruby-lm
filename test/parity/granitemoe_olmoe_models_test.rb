$LOAD_PATH.unshift File.expand_path("../../lib", __dir__)
$LOAD_PATH.unshift File.expand_path("../../mlx-ruby/lib", __dir__)

require "mlx"
require "minitest/autorun"

require_relative "../../lib/mlx_lm/model_args"
require_relative "../../lib/mlx_lm/models"
require_relative "../../lib/mlx_lm/models/activations"
require_relative "../../lib/mlx_lm/models/rope_utils"
require_relative "../../lib/mlx_lm/models/granite"
require_relative "../../lib/mlx_lm/models/granitemoe"
require_relative "../../lib/mlx_lm/models/olmo2"
require_relative "../../lib/mlx_lm/models/olmoe"

class Phase20DenseLaneTGraniteMoeTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_granitemoe_construct_forward_shape_and_sanitize_moe_linear_split
    args = MlxLm::Models::GraniteMoe::ModelArgs.from_dict({
      "model_type" => "granitemoe",
      "hidden_size" => 32,
      "num_hidden_layers" => 2,
      "intermediate_size" => 64,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 4,
      "rms_norm_eps" => 1e-5,
      "vocab_size" => 97,
      "logits_scaling" => 2.0,
      "attention_multiplier" => 0.25,
      "embedding_multiplier" => 1.25,
      "residual_multiplier" => 0.75,
      "max_position_embeddings" => 256,
      "attention_bias" => false,
      "mlp_bias" => false,
      "rope_theta" => 10_000.0,
      "num_local_experts" => 2,
      "num_experts_per_tok" => 1,
      "tie_word_embeddings" => true,
    })

    model = MlxLm::Models::GraniteMoe::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    tokens = @mx.array([[1, 2, 3, 4]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 4, 97], output.shape

    weights = {
      "model.layers.0.block_sparse_moe.input_linear.weight" => @mx.array((0...24).to_a, dtype: @mx.float32).reshape([4, 6]),
      "model.layers.0.block_sparse_moe.output_linear.weight" => @mx.zeros([3, 4]).astype(@mx.float32),
      "lm_head.weight" => @mx.zeros([97, 32]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)

    refute sanitized.key?("model.layers.0.block_sparse_moe.input_linear.weight")
    refute sanitized.key?("model.layers.0.block_sparse_moe.output_linear.weight")
    refute sanitized.key?("lm_head.weight")
    assert sanitized.key?("model.layers.0.block_sparse_moe.switch_mlp.gate_proj.weight")
    assert sanitized.key?("model.layers.0.block_sparse_moe.switch_mlp.up_proj.weight")
    assert sanitized.key?("model.layers.0.block_sparse_moe.switch_mlp.down_proj.weight")
    assert_equal [4, 3], sanitized["model.layers.0.block_sparse_moe.switch_mlp.gate_proj.weight"].shape
    assert_equal [4, 3], sanitized["model.layers.0.block_sparse_moe.switch_mlp.up_proj.weight"].shape
    assert_equal [3, 4], sanitized["model.layers.0.block_sparse_moe.switch_mlp.down_proj.weight"].shape
  end
end

class Phase20DenseLaneTOLMoETest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_olmoe_construct_forward_shape_and_sanitize_stacks_expert_weights
    args = MlxLm::Models::OLMoE::ModelArgs.from_dict({
      "model_type" => "olmoe",
      "hidden_size" => 32,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 2,
      "intermediate_size" => 64,
      "vocab_size" => 103,
      "rms_norm_eps" => 1e-5,
      "head_dim" => 8,
      "max_position_embeddings" => 256,
      "rope_theta" => 10_000.0,
      "num_experts" => 2,
      "num_experts_per_tok" => 1,
      "norm_topk_prob" => false,
      "tie_word_embeddings" => true,
    })

    model = MlxLm::Models::OLMoE::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 3, 103], output.shape

    weights = {
      "model.layers.0.mlp.experts.0.gate_proj.weight" => @mx.array([[1.0, 2.0], [3.0, 4.0]], dtype: @mx.float32),
      "model.layers.0.mlp.experts.1.gate_proj.weight" => @mx.array([[5.0, 6.0], [7.0, 8.0]], dtype: @mx.float32),
      "model.layers.0.mlp.experts.0.up_proj.weight" => @mx.array([[2.0, 2.0], [2.0, 2.0]], dtype: @mx.float32),
      "model.layers.0.mlp.experts.1.up_proj.weight" => @mx.array([[4.0, 4.0], [4.0, 4.0]], dtype: @mx.float32),
      "model.layers.0.mlp.experts.0.down_proj.weight" => @mx.array([[9.0, 9.0], [9.0, 9.0]], dtype: @mx.float32),
      "model.layers.0.mlp.experts.1.down_proj.weight" => @mx.array([[10.0, 10.0], [10.0, 10.0]], dtype: @mx.float32),
      "model.layers.0.self_attn.rotary_emb.inv_freq" => @mx.zeros([8]).astype(@mx.float32),
      "lm_head.weight" => @mx.zeros([103, 32]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)
    stacked_gate = sanitized["model.layers.0.mlp.switch_mlp.gate_proj.weight"]
    @mx.eval(stacked_gate)

    refute sanitized.key?("model.layers.0.mlp.experts.0.gate_proj.weight")
    refute sanitized.key?("model.layers.0.mlp.experts.1.gate_proj.weight")
    refute sanitized.key?("model.layers.0.self_attn.rotary_emb.inv_freq")
    refute sanitized.key?("lm_head.weight")
    assert sanitized.key?("model.layers.0.mlp.switch_mlp.gate_proj.weight")
    assert sanitized.key?("model.layers.0.mlp.switch_mlp.up_proj.weight")
    assert sanitized.key?("model.layers.0.mlp.switch_mlp.down_proj.weight")
    assert_equal [2, 2, 2], stacked_gate.shape
  end
end

class Phase20DenseLaneTRegistryTest < Minitest::Test
  def test_models_registered_and_resolve
    assert MlxLm::Models::REGISTRY.key?("granitemoe"), "granitemoe should be registered"
    assert MlxLm::Models::REGISTRY.key?("olmoe"), "olmoe should be registered"

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "granitemoe" })
    assert_equal MlxLm::Models::GraniteMoe::Model, model_class
    assert_equal MlxLm::Models::GraniteMoe::ModelArgs, args_class

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "olmoe" })
    assert_equal MlxLm::Models::OLMoE::Model, model_class
    assert_equal MlxLm::Models::OLMoE::ModelArgs, args_class
  end
end
