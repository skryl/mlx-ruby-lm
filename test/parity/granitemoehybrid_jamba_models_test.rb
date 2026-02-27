$LOAD_PATH.unshift File.expand_path("../../lib", __dir__)
$LOAD_PATH.unshift File.expand_path("../../mlx-ruby/lib", __dir__)

require "mlx"
require "minitest/autorun"

require_relative "../../lib/mlx_lm/model_args"
require_relative "../../lib/mlx_lm/models"
require_relative "../../lib/mlx_lm/models/cache"
require_relative "../../lib/mlx_lm/models/recurrent_gemma"
require_relative "../../lib/mlx_lm/models/falcon_h1"
require_relative "../../lib/mlx_lm/models/granitemoehybrid"
require_relative "../../lib/mlx_lm/models/jamba"

class Phase27HybridLaneAQGraniteMoeHybridTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_granitemoehybrid_construct_forward_shape_sanitize_mapping_and_cache
    args = MlxLm::Models::GraniteMoeHybrid::ModelArgs.from_dict({
      "model_type" => "granitemoehybrid",
      "vocab_size" => 73,
      "hidden_size" => 32,
      "intermediate_size" => 64,
      "num_hidden_layers" => 2,
      "max_position_embeddings" => 128,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 2,
      "attention_bias" => false,
      "embedding_multiplier" => 1.0,
      "attention_multiplier" => 1.0,
      "logits_scaling" => 1.0,
      "residual_multiplier" => 1.0,
      "num_local_experts" => 2,
      "num_experts_per_tok" => 1,
      "shared_intermediate_size" => 32,
      "mamba_n_heads" => 4,
      "mamba_d_head" => 8,
      "mamba_d_conv" => 3,
      "layer_types" => ["mamba", "attention"],
      "rms_norm_eps" => 1e-5,
      "rope_theta" => 10_000.0,
      "tie_word_embeddings" => true,
    })
    assert_instance_of MlxLm::Models::GraniteMoeHybrid::ModelArgs, args

    model = MlxLm::Models::GraniteMoeHybrid::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 3, 73], output.shape

    input_linear = @mx.array((0...24).to_a, dtype: @mx.float32).reshape([4, 6])
    output_linear = @mx.array((0...12).to_a, dtype: @mx.float32).reshape([3, 4])
    expected_gate, _expected_up = @mx.split(input_linear, [3], 1)
    weights = {
      "model.layers.0.block_sparse_moe.input_linear.weight" => input_linear,
      "model.layers.0.block_sparse_moe.output_linear.weight" => output_linear,
      "model.layers.0.mamba.in_proj.weight" => @mx.zeros([32, 32]).astype(@mx.float32),
      "model.layers.0.temporal_block.linear_out.weight" => @mx.ones([32, 32]).astype(@mx.float32),
      "model.norm.weight" => @mx.ones([32]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)
    mapped_gate = sanitized["model.layers.0.mlp_block.switch_mlp.gate_proj.weight"]
    mapped_in_proj = sanitized["model.layers.0.temporal_block.linear_x.weight"]
    pass_through = sanitized["model.layers.0.temporal_block.linear_out.weight"]
    mapped_final_norm = sanitized["model.final_norm.weight"]
    @mx.eval(mapped_gate, mapped_in_proj, pass_through, mapped_final_norm)

    refute sanitized.key?("model.layers.0.block_sparse_moe.input_linear.weight")
    refute sanitized.key?("model.layers.0.block_sparse_moe.output_linear.weight")
    refute sanitized.key?("model.layers.0.mamba.in_proj.weight")
    refute sanitized.key?("model.norm.weight")
    assert sanitized.key?("model.layers.0.mlp_block.switch_mlp.gate_proj.weight")
    assert sanitized.key?("model.layers.0.mlp_block.switch_mlp.up_proj.weight")
    assert sanitized.key?("model.layers.0.mlp_block.switch_mlp.down_proj.weight")
    assert sanitized.key?("model.layers.0.temporal_block.linear_x.weight")
    assert sanitized.key?("model.layers.0.temporal_block.linear_out.weight")
    assert sanitized.key?("model.final_norm.weight")
    assert_equal [4, 3], mapped_gate.shape
    assert_equal expected_gate.to_a, mapped_gate.to_a

    cache = model.make_cache
    assert_equal 2, cache.length
  end
end

class Phase27HybridLaneAQJambaTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_jamba_construct_forward_shape_sanitize_mapping_and_cache
    args = MlxLm::Models::Jamba::ModelArgs.from_dict({
      "model_type" => "jamba",
      "hidden_size" => 32,
      "intermediate_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 2,
      "attn_layer_offset" => 1,
      "attn_layer_period" => 2,
      "expert_layer_offset" => 1,
      "expert_layer_period" => 2,
      "mamba_d_conv" => 3,
      "mamba_d_state" => 16,
      "mamba_expand" => 2,
      "num_experts" => 2,
      "num_experts_per_tok" => 1,
      "rms_norm_eps" => 1e-5,
      "max_position_embeddings" => 128,
      "rope_theta" => 10_000.0,
      "vocab_size" => 79,
      "tie_word_embeddings" => true,
    })
    assert_instance_of MlxLm::Models::Jamba::ModelArgs, args

    model = MlxLm::Models::Jamba::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 3, 79], output.shape

    expert0 = @mx.array([[1.0, 2.0], [3.0, 4.0]], dtype: @mx.float32)
    expert1 = @mx.array([[5.0, 6.0], [7.0, 8.0]], dtype: @mx.float32)
    weights = {
      "model.layers.0.feed_forward.experts.0.up_proj.weight" => expert0,
      "model.layers.0.feed_forward.experts.1.up_proj.weight" => expert1,
      "model.layers.0.mamba.in_proj.weight" => @mx.zeros([32, 32]).astype(@mx.float32),
      "model.layers.0.temporal_block.linear_out.weight" => @mx.ones([32, 32]).astype(@mx.float32),
      "model.final_layernorm.weight" => @mx.ones([32]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)
    stacked_experts = sanitized["model.layers.0.mlp_block.switch_mlp.up_proj.weight"]
    mapped_in_proj = sanitized["model.layers.0.temporal_block.linear_x.weight"]
    pass_through = sanitized["model.layers.0.temporal_block.linear_out.weight"]
    mapped_final_norm = sanitized["model.final_norm.weight"]
    @mx.eval(stacked_experts, mapped_in_proj, pass_through, mapped_final_norm)

    refute sanitized.key?("model.layers.0.feed_forward.experts.0.up_proj.weight")
    refute sanitized.key?("model.layers.0.feed_forward.experts.1.up_proj.weight")
    refute sanitized.key?("model.layers.0.mamba.in_proj.weight")
    refute sanitized.key?("model.final_layernorm.weight")
    assert sanitized.key?("model.layers.0.mlp_block.switch_mlp.up_proj.weight")
    assert sanitized.key?("model.layers.0.temporal_block.linear_x.weight")
    assert sanitized.key?("model.layers.0.temporal_block.linear_out.weight")
    assert sanitized.key?("model.final_norm.weight")
    assert_equal [2, 2, 2], stacked_experts.shape
    assert_equal expert0.to_a, stacked_experts[0].to_a
    assert_equal expert1.to_a, stacked_experts[1].to_a

    cache = model.make_cache
    assert_equal 2, cache.length
  end
end

class Phase27HybridLaneAQRegistryTest < Minitest::Test
  def test_models_registered_and_resolve
    assert MlxLm::Models::REGISTRY.key?("granitemoehybrid"), "granitemoehybrid should be registered"
    assert MlxLm::Models::REGISTRY.key?("jamba"), "jamba should be registered"

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "granitemoehybrid" })
    assert_equal MlxLm::Models::GraniteMoeHybrid::Model, model_class
    assert_equal MlxLm::Models::GraniteMoeHybrid::ModelArgs, args_class

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "jamba" })
    assert_equal MlxLm::Models::Jamba::Model, model_class
    assert_equal MlxLm::Models::Jamba::ModelArgs, args_class
  end
end
