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
require_relative "../../lib/mlx_lm/models/recurrent_gemma"
require_relative "../../lib/mlx_lm/models/step3p5"

class RecurrentGemmaStep3P5ModelsTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_recurrent_gemma_construct_forward_shape_sanitize_and_make_cache
    args = MlxLm::Models::RecurrentGemma::ModelArgs.from_dict({
      "model_type" => "recurrent_gemma",
      "hidden_size" => 32,
      "attention_bias" => false,
      "conv1d_width" => 3,
      "intermediate_size" => 64,
      "logits_soft_cap" => 1.5,
      "num_attention_heads" => 4,
      "num_hidden_layers" => 3,
      "num_key_value_heads" => 2,
      "rms_norm_eps" => 1e-5,
      "rope_theta" => 10_000.0,
      "attention_window_size" => 4,
      "vocab_size" => 97,
      "block_types" => ["recurrent", "attention"],
    })

    model = MlxLm::Models::RecurrentGemma::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3, 4]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 4, 97], output.shape

    conv_weight = @mx.array((0...24).to_a, dtype: @mx.float32).reshape([4, 1, 6])
    weights = {
      "model.layers.0.temporal_block.conv_1d.weight" => conv_weight,
      "model.layers.0.temporal_block.linear_x.weight" => @mx.zeros([32, 32]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)
    sanitized_conv = sanitized["model.layers.0.temporal_block.conv_1d.weight"]
    @mx.eval(sanitized_conv)

    assert_equal [4, 6, 1], sanitized_conv.shape
    assert_equal @mx.swapaxes(conv_weight, 1, 2).to_a, sanitized_conv.to_a
    assert sanitized.key?("model.layers.0.temporal_block.linear_x.weight")
    assert_nil model.lm_head

    tied_output = model.call(tokens)
    @mx.eval(tied_output)
    assert_equal [1, 4, 97], tied_output.shape

    cache = model.make_cache
    assert_equal 3, cache.length
    assert_instance_of MlxLm::ArraysCache, cache[0]
    assert_instance_of MlxLm::RotatingKVCache, cache[1]
    assert_instance_of MlxLm::ArraysCache, cache[2]
    assert_equal 2, cache[0].cache.length
    assert_equal 2, cache[2].cache.length
  end
end

class RecurrentGemmaStep3P5ModelsStep3p5Test < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_step3p5_construct_forward_shape_sanitize_and_make_cache
    args = MlxLm::Models::Step3p5::ModelArgs.from_dict({
      "model_type" => "step3p5",
      "hidden_size" => 32,
      "num_hidden_layers" => 3,
      "vocab_size" => 103,
      "num_attention_heads" => 4,
      "num_attention_groups" => 2,
      "head_dim" => 8,
      "intermediate_size" => 64,
      "rms_norm_eps" => 1e-5,
      "rope_theta" => [10_000.0, 10_000.0, 10_000.0],
      "sliding_window" => 4,
      "layer_types" => ["full_attention", "sliding_attention", "full_attention"],
      "partial_rotary_factors" => [0.5, 1.0, 0.5],
      "attention_other_setting" => {
        "num_attention_heads" => 4,
        "num_attention_groups" => 2,
      },
      "use_head_wise_attn_gate" => true,
      "moe_num_experts" => 2,
      "moe_top_k" => 1,
      "moe_intermediate_size" => 48,
      "share_expert_dim" => 48,
      "moe_layers_enum" => "1,2",
    })

    model = MlxLm::Models::Step3p5::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 3, 103], output.shape

    norm_weight = @mx.array([1.0, 2.0], dtype: @mx.float32)
    weights = {
      "model.layers.1.moe.gate_proj.weight" => @mx.array((0...48).to_a, dtype: @mx.float32).reshape([2, 6, 4]),
      "model.layers.1.moe.up_proj.weight" => @mx.array((48...96).to_a, dtype: @mx.float32).reshape([2, 6, 4]),
      "model.layers.1.moe.down_proj.weight" => @mx.array((0...48).to_a, dtype: @mx.float32).reshape([2, 4, 6]),
      "model.layers.1.moe.gate.weight" => @mx.zeros([2, 32]).astype(@mx.float32),
      "model.layers.1.moe.router_bias" => @mx.zeros([2]).astype(@mx.float32),
      "model.layers.1.share_expert.gate_proj.weight" => @mx.zeros([48, 32]).astype(@mx.float32),
      "model.layers.0.input_layernorm.weight" => norm_weight,
      "model.layers.5.mlp.up_proj.weight" => @mx.zeros([32, 32]).astype(@mx.float32),
      "model.mtp.layers.0.weight" => @mx.zeros([2, 2]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)
    remapped_gate = sanitized["model.layers.1.mlp.switch_mlp.gate_proj.weight"]
    remapped_up = sanitized["model.layers.1.mlp.switch_mlp.up_proj.weight"]
    remapped_down = sanitized["model.layers.1.mlp.switch_mlp.down_proj.weight"]
    remapped_router = sanitized["model.layers.1.mlp.gate.gate.weight"]
    remapped_router_bias = sanitized["model.layers.1.mlp.gate.router_bias"]
    remapped_shared = sanitized["model.layers.1.mlp.share_expert.gate_proj.weight"]
    sanitized_norm = sanitized["model.layers.0.input_layernorm.weight"]
    @mx.eval(
      remapped_gate,
      remapped_up,
      remapped_down,
      remapped_router,
      remapped_router_bias,
      remapped_shared,
      sanitized_norm
    )

    refute sanitized.key?("model.layers.1.moe.gate_proj.weight")
    refute sanitized.key?("model.layers.1.moe.up_proj.weight")
    refute sanitized.key?("model.layers.1.moe.down_proj.weight")
    refute sanitized.key?("model.layers.1.moe.gate.weight")
    refute sanitized.key?("model.layers.1.moe.router_bias")
    refute sanitized.key?("model.layers.1.share_expert.gate_proj.weight")
    refute sanitized.key?("model.layers.5.mlp.up_proj.weight")
    refute sanitized.key?("model.mtp.layers.0.weight")

    assert sanitized.key?("model.layers.1.mlp.switch_mlp.gate_proj.weight")
    assert sanitized.key?("model.layers.1.mlp.switch_mlp.up_proj.weight")
    assert sanitized.key?("model.layers.1.mlp.switch_mlp.down_proj.weight")
    assert sanitized.key?("model.layers.1.mlp.gate.gate.weight")
    assert sanitized.key?("model.layers.1.mlp.gate.router_bias")
    assert sanitized.key?("model.layers.1.mlp.share_expert.gate_proj.weight")

    assert_equal [2, 6, 4], remapped_gate.shape
    assert_equal [2, 6, 4], remapped_up.shape
    assert_equal [2, 4, 6], remapped_down.shape
    assert_equal [2, 32], remapped_router.shape
    assert_equal [2], remapped_router_bias.shape
    assert_equal [48, 32], remapped_shared.shape
    assert_equal [2.0, 3.0], sanitized_norm.to_a

    cache = model.make_cache
    assert_equal 3, cache.length
    cache.each { |entry| assert_instance_of MlxLm::KVCache, entry }
  end
end

class RecurrentGemmaStep3P5ModelsModelsRegisteredAndResolveTest < Minitest::Test
  def test_models_registered_and_resolve
    assert MlxLm::Models::REGISTRY.key?("recurrent_gemma"), "recurrent_gemma should be registered"
    assert MlxLm::Models::REGISTRY.key?("step3p5"), "step3p5 should be registered"

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "recurrent_gemma" })
    assert_equal MlxLm::Models::RecurrentGemma::Model, model_class
    assert_equal MlxLm::Models::RecurrentGemma::ModelArgs, args_class

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "step3p5" })
    assert_equal MlxLm::Models::Step3p5::Model, model_class
    assert_equal MlxLm::Models::Step3p5::ModelArgs, args_class
  end
end
