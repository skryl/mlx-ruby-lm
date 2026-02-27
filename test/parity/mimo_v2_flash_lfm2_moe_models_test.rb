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
require_relative "../../lib/mlx_lm/models/mimo_v2_flash"
require_relative "../../lib/mlx_lm/models/lfm2_moe"

class Phase24DenseLaneAJMimoV2FlashTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_mimo_v2_flash_construct_forward_shape_and_sanitize_stacks_experts_and_cleans_mtp
    args = MlxLm::Models::MimoV2Flash::ModelArgs.from_dict({
      "model_type" => "mimo_v2_flash",
      "num_experts_per_tok" => 1,
      "hybrid_layer_pattern" => [0, 1],
      "moe_layer_freq" => [0, 1],
      "add_swa_attention_sink_bias" => false,
      "add_full_attention_sink_bias" => false,
      "sliding_window_size" => 2,
      "vocab_size" => 113,
      "hidden_size" => 32,
      "intermediate_size" => 64,
      "moe_intermediate_size" => 48,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 2,
      "n_shared_experts" => 1,
      "n_routed_experts" => 2,
      "routed_scaling_factor" => 1.0,
      "topk_method" => "noaux_tc",
      "scoring_func" => "sigmoid",
      "norm_topk_prob" => true,
      "n_group" => 1,
      "topk_group" => 1,
      "max_position_embeddings" => 128,
      "layernorm_epsilon" => 1e-5,
      "rope_theta" => 10_000.0,
      "swa_rope_theta" => 20_000.0,
      "swa_num_attention_heads" => 4,
      "swa_num_key_value_heads" => 2,
      "head_dim" => 8,
      "v_head_dim" => 8,
      "swa_head_dim" => 8,
      "swa_v_head_dim" => 8,
      "partial_rotary_factor" => 1.0,
    })

    model = MlxLm::Models::MimoV2Flash::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3, 4]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 4, 113], output.shape

    weights = {
      "model.layers.1.mlp.experts.0.gate_proj.weight" => @mx.array((0...24).to_a, dtype: @mx.float32).reshape([6, 4]),
      "model.layers.1.mlp.experts.1.gate_proj.weight" => @mx.array((24...48).to_a, dtype: @mx.float32).reshape([6, 4]),
      "model.layers.1.mlp.experts.0.up_proj.weight" => @mx.array((48...72).to_a, dtype: @mx.float32).reshape([6, 4]),
      "model.layers.1.mlp.experts.1.up_proj.weight" => @mx.array((72...96).to_a, dtype: @mx.float32).reshape([6, 4]),
      "model.layers.1.mlp.experts.0.down_proj.weight" => @mx.array((0...24).to_a, dtype: @mx.float32).reshape([4, 6]),
      "model.layers.1.mlp.experts.1.down_proj.weight" => @mx.array((24...48).to_a, dtype: @mx.float32).reshape([4, 6]),
      "model.mtp.layers.0.weight" => @mx.zeros([2, 2]).astype(@mx.float32),
      "model.layers.0.self_attn.q_proj.weight" => @mx.zeros([32, 32]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)
    stacked_gate = sanitized["model.layers.1.mlp.switch_mlp.gate_proj.weight"]
    stacked_up = sanitized["model.layers.1.mlp.switch_mlp.up_proj.weight"]
    stacked_down = sanitized["model.layers.1.mlp.switch_mlp.down_proj.weight"]
    @mx.eval(stacked_gate, stacked_up, stacked_down)

    refute sanitized.key?("model.layers.1.mlp.experts.0.gate_proj.weight")
    refute sanitized.key?("model.layers.1.mlp.experts.1.gate_proj.weight")
    refute sanitized.key?("model.layers.1.mlp.experts.0.up_proj.weight")
    refute sanitized.key?("model.layers.1.mlp.experts.1.up_proj.weight")
    refute sanitized.key?("model.layers.1.mlp.experts.0.down_proj.weight")
    refute sanitized.key?("model.layers.1.mlp.experts.1.down_proj.weight")
    refute sanitized.key?("model.mtp.layers.0.weight")

    assert sanitized.key?("model.layers.1.mlp.switch_mlp.gate_proj.weight")
    assert sanitized.key?("model.layers.1.mlp.switch_mlp.up_proj.weight")
    assert sanitized.key?("model.layers.1.mlp.switch_mlp.down_proj.weight")
    assert sanitized.key?("model.layers.0.self_attn.q_proj.weight")

    assert_equal [2, 6, 4], stacked_gate.shape
    assert_equal [2, 6, 4], stacked_up.shape
    assert_equal [2, 4, 6], stacked_down.shape
  end
end

class Phase24DenseLaneAJLfm2MoeTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_lfm2_moe_construct_forward_shape_and_sanitize_transposes_conv_and_stacks_experts
    args = MlxLm::Models::Lfm2Moe::ModelArgs.from_dict({
      "model_type" => "lfm2_moe",
      "vocab_size" => 101,
      "hidden_size" => 32,
      "intermediate_size" => 64,
      "moe_intermediate_size" => 48,
      "num_hidden_layers" => 3,
      "num_experts" => 2,
      "num_experts_per_tok" => 1,
      "norm_topk_prob" => true,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 2,
      "max_position_embeddings" => 128,
      "use_expert_bias" => true,
      "num_dense_layers" => 1,
      "norm_eps" => 1e-5,
      "conv_bias" => false,
      "conv_L_cache" => 3,
      "layer_types" => ["full_attention", "conv", "full_attention"],
      "rope_parameters" => { "rope_theta" => 10_000.0 },
    })

    model = MlxLm::Models::Lfm2Moe::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 3, 101], output.shape

    conv_weight = @mx.array((0...24).to_a, dtype: @mx.float32).reshape([2, 3, 4])
    weights = {
      "model.layers.1.conv.weight" => conv_weight,
      "model.layers.1.feed_forward.experts.0.w1.weight" => @mx.array((0...24).to_a, dtype: @mx.float32).reshape([6, 4]),
      "model.layers.1.feed_forward.experts.1.w1.weight" => @mx.array((24...48).to_a, dtype: @mx.float32).reshape([6, 4]),
      "model.layers.1.feed_forward.experts.0.w2.weight" => @mx.array((0...24).to_a, dtype: @mx.float32).reshape([4, 6]),
      "model.layers.1.feed_forward.experts.1.w2.weight" => @mx.array((24...48).to_a, dtype: @mx.float32).reshape([4, 6]),
      "model.layers.1.feed_forward.experts.0.w3.weight" => @mx.array((48...72).to_a, dtype: @mx.float32).reshape([6, 4]),
      "model.layers.1.feed_forward.experts.1.w3.weight" => @mx.array((72...96).to_a, dtype: @mx.float32).reshape([6, 4]),
      "model.layers.0.self_attn.q_proj.weight" => @mx.zeros([32, 32]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)
    sanitized_conv = sanitized["model.layers.1.conv.weight"]
    stacked_gate = sanitized["model.layers.1.feed_forward.switch_mlp.gate_proj.weight"]
    stacked_down = sanitized["model.layers.1.feed_forward.switch_mlp.down_proj.weight"]
    stacked_up = sanitized["model.layers.1.feed_forward.switch_mlp.up_proj.weight"]
    @mx.eval(sanitized_conv, stacked_gate, stacked_down, stacked_up)

    assert_equal [2, 4, 3], sanitized_conv.shape
    assert_equal @mx.swapaxes(conv_weight, 1, 2).to_a, sanitized_conv.to_a

    refute sanitized.key?("model.layers.1.feed_forward.experts.0.gate_proj.weight")
    refute sanitized.key?("model.layers.1.feed_forward.experts.1.gate_proj.weight")
    refute sanitized.key?("model.layers.1.feed_forward.experts.0.down_proj.weight")
    refute sanitized.key?("model.layers.1.feed_forward.experts.1.down_proj.weight")
    refute sanitized.key?("model.layers.1.feed_forward.experts.0.up_proj.weight")
    refute sanitized.key?("model.layers.1.feed_forward.experts.1.up_proj.weight")

    assert sanitized.key?("model.layers.1.feed_forward.switch_mlp.gate_proj.weight")
    assert sanitized.key?("model.layers.1.feed_forward.switch_mlp.down_proj.weight")
    assert sanitized.key?("model.layers.1.feed_forward.switch_mlp.up_proj.weight")
    assert sanitized.key?("model.layers.0.self_attn.q_proj.weight")

    assert_equal [2, 6, 4], stacked_gate.shape
    assert_equal [2, 4, 6], stacked_down.shape
    assert_equal [2, 6, 4], stacked_up.shape
  end

  def test_lfm2_moe_make_cache_returns_attention_and_conv_cache_types
    args = MlxLm::Models::Lfm2Moe::ModelArgs.from_dict({
      "model_type" => "lfm2_moe",
      "vocab_size" => 64,
      "hidden_size" => 16,
      "intermediate_size" => 32,
      "moe_intermediate_size" => 24,
      "num_hidden_layers" => 3,
      "num_experts" => 2,
      "num_experts_per_tok" => 1,
      "norm_topk_prob" => false,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 1,
      "max_position_embeddings" => 64,
      "use_expert_bias" => false,
      "num_dense_layers" => 1,
      "norm_eps" => 1e-5,
      "conv_bias" => false,
      "conv_L_cache" => 3,
      "full_attn_idxs" => [0, 2],
      "rope_theta" => 10_000.0,
    })

    model = MlxLm::Models::Lfm2Moe::Model.new(args)
    cache = model.make_cache

    assert_equal 3, cache.length
    assert_instance_of MlxLm::KVCache, cache[0]
    assert_instance_of MlxLm::ArraysCache, cache[1]
    assert_instance_of MlxLm::KVCache, cache[2]
  end
end

class Phase24DenseLaneAJRegistryTest < Minitest::Test
  def test_models_registered_and_resolve
    assert MlxLm::Models::REGISTRY.key?("mimo_v2_flash"), "mimo_v2_flash should be registered"
    assert MlxLm::Models::REGISTRY.key?("lfm2_moe"), "lfm2_moe should be registered"

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "mimo_v2_flash" })
    assert_equal MlxLm::Models::MimoV2Flash::Model, model_class
    assert_equal MlxLm::Models::MimoV2Flash::ModelArgs, args_class

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "lfm2_moe" })
    assert_equal MlxLm::Models::Lfm2Moe::Model, model_class
    assert_equal MlxLm::Models::Lfm2Moe::ModelArgs, args_class
  end
end
