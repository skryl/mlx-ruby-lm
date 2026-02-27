$LOAD_PATH.unshift File.expand_path("../../lib", __dir__)
$LOAD_PATH.unshift File.expand_path("../../mlx-ruby/lib", __dir__)

require "mlx"
require "minitest/autorun"

require_relative "../../lib/mlx_lm/model_args"
require_relative "../../lib/mlx_lm/models"
require_relative "../../lib/mlx_lm/models/switch_layers"
require_relative "../../lib/mlx_lm/models/deepseek"
require_relative "../../lib/mlx_lm/models/deepseek_v2"
require_relative "../../lib/mlx_lm/models/deepseek_v3"

class Phase22DenseLaneYDeepseekV2Test < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_deepseek_v2_construct_forward_shape_and_sanitize_stacks_experts
    args = MlxLm::Models::DeepseekV2::ModelArgs.from_dict({
      "model_type" => "deepseek_v2",
      "hidden_size" => 32,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 4,
      "intermediate_size" => 64,
      "moe_intermediate_size" => 16,
      "vocab_size" => 97,
      "n_routed_experts" => 2,
      "num_experts_per_tok" => 1,
      "n_shared_experts" => 1,
      "moe_layer_freq" => 1,
      "first_k_dense_replace" => 0,
      "max_position_embeddings" => 256,
      "rope_theta" => 10_000.0,
    })

    model = MlxLm::Models::DeepseekV2::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3, 4]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 4, 97], output.shape

    weights = {
      "model.layers.0.mlp.experts.0.gate_proj.weight" => @mx.array([[1.0, 2.0], [3.0, 4.0]], dtype: @mx.float32),
      "model.layers.0.mlp.experts.1.gate_proj.weight" => @mx.array([[5.0, 6.0], [7.0, 8.0]], dtype: @mx.float32),
      "model.layers.0.self_attn.rotary_emb.inv_freq" => @mx.zeros([8]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)
    stacked = sanitized["model.layers.0.mlp.switch_mlp.gate_proj.weight"]
    @mx.eval(stacked)

    refute sanitized.key?("model.layers.0.mlp.experts.0.gate_proj.weight")
    refute sanitized.key?("model.layers.0.mlp.experts.1.gate_proj.weight")
    assert sanitized.key?("model.layers.0.self_attn.rotary_emb.inv_freq")
    assert_equal [2, 2, 2], stacked.shape
  end
end

class Phase22DenseLaneYDeepseekV3Test < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_deepseek_v3_construct_forward_shape_and_sanitize_prunes_extra_keys
    args = MlxLm::Models::DeepseekV3::ModelArgs.from_dict({
      "model_type" => "deepseek_v3",
      "hidden_size" => 32,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 4,
      "intermediate_size" => 64,
      "moe_intermediate_size" => 16,
      "vocab_size" => 103,
      "n_routed_experts" => 2,
      "num_experts_per_tok" => 1,
      "n_shared_experts" => 1,
      "moe_layer_freq" => 1,
      "first_k_dense_replace" => 0,
      "max_position_embeddings" => 256,
      "rope_theta" => 10_000.0,
    })

    model = MlxLm::Models::DeepseekV3::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 3, 103], output.shape

    weights = {
      "model.layers.0.mlp.experts.0.up_proj.weight" => @mx.array([[1.0, 2.0], [3.0, 4.0]], dtype: @mx.float32),
      "model.layers.0.mlp.experts.1.up_proj.weight" => @mx.array([[5.0, 6.0], [7.0, 8.0]], dtype: @mx.float32),
      "model.layers.0.self_attn.rotary_emb.inv_freq" => @mx.zeros([8]).astype(@mx.float32),
      "model.layers.61.mlp.down_proj.weight" => @mx.zeros([1]).astype(@mx.float32),
      "model.layers.1.self_attn.q_proj.weight" => @mx.zeros([32, 32]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)
    stacked = sanitized["model.layers.0.mlp.switch_mlp.up_proj.weight"]
    @mx.eval(stacked)

    refute sanitized.key?("model.layers.0.mlp.experts.0.up_proj.weight")
    refute sanitized.key?("model.layers.0.mlp.experts.1.up_proj.weight")
    refute sanitized.key?("model.layers.0.self_attn.rotary_emb.inv_freq")
    refute sanitized.key?("model.layers.61.mlp.down_proj.weight")
    assert sanitized.key?("model.layers.1.self_attn.q_proj.weight")
    assert_equal [2, 2, 2], stacked.shape
  end
end

class Phase22DenseLaneYRegistryTest < Minitest::Test
  def test_models_registered_and_resolve
    assert MlxLm::Models::REGISTRY.key?("deepseek_v2"), "deepseek_v2 should be registered"
    assert MlxLm::Models::REGISTRY.key?("deepseek_v3"), "deepseek_v3 should be registered"

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "deepseek_v2" })
    assert_equal MlxLm::Models::DeepseekV2::Model, model_class
    assert_equal MlxLm::Models::DeepseekV2::ModelArgs, args_class

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "deepseek_v3" })
    assert_equal MlxLm::Models::DeepseekV3::Model, model_class
    assert_equal MlxLm::Models::DeepseekV3::ModelArgs, args_class
  end
end
