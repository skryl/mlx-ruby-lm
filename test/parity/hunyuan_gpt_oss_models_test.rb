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
require_relative "../../lib/mlx_lm/models/hunyuan"
require_relative "../../lib/mlx_lm/models/gpt_oss"

class HunyuanGptOssModelsTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_hunyuan_construct_forward_shape_and_sanitize_stacks_experts
    args = MlxLm::Models::Hunyuan::ModelArgs.from_dict({
      "model_type" => "hunyuan",
      "vocab_size" => 103,
      "hidden_size" => 32,
      "num_hidden_layers" => 2,
      "intermediate_size" => 64,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 2,
      "attention_bias" => false,
      "moe_topk" => 1,
      "num_experts" => 2,
      "num_shared_expert" => 1,
      "use_mixed_mlp_moe" => true,
      "use_qk_norm" => true,
      "rms_norm_eps" => 1e-5,
      "rope_theta" => 10_000.0,
      "use_cla" => true,
      "cla_share_factor" => 2,
      "rope_scaling" => {
        "alpha" => 1.0,
        "factor" => 1.0,
        "type" => "dynamic",
      },
      "tie_word_embeddings" => true,
    })

    model = MlxLm::Models::Hunyuan::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3, 4]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 4, 103], output.shape
    assert_equal 2, model.layers.length

    weights = {
      "model.layers.0.mlp.experts.0.up_proj.weight" => @mx.array((0...12).to_a, dtype: @mx.float32).reshape([4, 3]),
      "model.layers.0.mlp.experts.1.up_proj.weight" => @mx.array((12...24).to_a, dtype: @mx.float32).reshape([4, 3]),
      "model.layers.0.mlp.experts.0.down_proj.weight" => @mx.array((0...12).to_a, dtype: @mx.float32).reshape([3, 4]),
      "model.layers.0.mlp.experts.1.down_proj.weight" => @mx.array((12...24).to_a, dtype: @mx.float32).reshape([3, 4]),
      "model.layers.0.mlp.experts.0.gate_proj.weight" => @mx.array((0...12).to_a, dtype: @mx.float32).reshape([4, 3]),
      "model.layers.0.mlp.experts.1.gate_proj.weight" => @mx.array((12...24).to_a, dtype: @mx.float32).reshape([4, 3]),
      "model.layers.1.self_attn.q_proj.weight" => @mx.zeros([32, 32]).astype(@mx.float32),
      "lm_head.weight" => @mx.zeros([103, 32]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)
    up_stacked = sanitized["model.layers.0.mlp.switch_mlp.up_proj.weight"]
    down_stacked = sanitized["model.layers.0.mlp.switch_mlp.down_proj.weight"]
    gate_stacked = sanitized["model.layers.0.mlp.switch_mlp.gate_proj.weight"]
    @mx.eval(up_stacked, down_stacked, gate_stacked)

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
    assert_equal [2, 4, 3], up_stacked.shape
    assert_equal [2, 3, 4], down_stacked.shape
    assert_equal [2, 4, 3], gate_stacked.shape
  end
end

class HunyuanGptOssModelsGptOssConstructForwardShapeSanitizeCleanupAndCacheMixTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_gpt_oss_construct_forward_shape_sanitize_cleanup_and_cache_mix
    args = MlxLm::Models::GptOss::ModelArgs.from_dict({
      "model_type" => "gpt_oss",
      "num_hidden_layers" => 4,
      "num_local_experts" => 2,
      "num_experts_per_tok" => 1,
      "vocab_size" => 109,
      "rms_norm_eps" => 1e-5,
      "hidden_size" => 32,
      "intermediate_size" => 16,
      "head_dim" => 8,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 2,
      "sliding_window" => 4,
      "rope_theta" => 10_000,
      "layer_types" => [
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
      ],
    })

    model = MlxLm::Models::GptOss::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 3, 109], output.shape
    assert_equal 4, model.layers.length

    weights = {
      "model.layers.0.mlp.experts.gate_up_proj.weight" => @mx.array((0...48).to_a, dtype: @mx.float32).reshape([2, 6, 4]),
      "model.layers.0.mlp.experts.gate_up_proj_bias" => @mx.array((0...12).to_a, dtype: @mx.float32).reshape([2, 6]),
      "model.layers.0.mlp.experts.down_proj.weight" => @mx.array((0...24).to_a, dtype: @mx.float32).reshape([2, 4, 3]),
      "model.layers.0.mlp.experts.gate_up_proj_scales" => @mx.array((0...24).to_a, dtype: @mx.float32).reshape([2, 6, 2]),
      "model.layers.0.mlp.experts.down_proj_scales" => @mx.array((0...16).to_a, dtype: @mx.float32).reshape([2, 4, 2]),
      "model.layers.3.self_attn.q_proj.weight" => @mx.zeros([32, 32]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)

    gate_weight = sanitized["model.layers.0.mlp.experts.gate_proj.weight"]
    up_weight = sanitized["model.layers.0.mlp.experts.up_proj.weight"]
    gate_bias = sanitized["model.layers.0.mlp.experts.gate_proj.bias"]
    up_bias = sanitized["model.layers.0.mlp.experts.up_proj.bias"]
    gate_scales = sanitized["model.layers.0.mlp.experts.gate_proj.scales"]
    up_scales = sanitized["model.layers.0.mlp.experts.up_proj.scales"]
    down_weight = sanitized["model.layers.0.mlp.experts.down_proj.weight"]
    down_scales = sanitized["model.layers.0.mlp.experts.down_proj.scales"]
    @mx.eval(
      gate_weight,
      up_weight,
      gate_bias,
      up_bias,
      gate_scales,
      up_scales,
      down_weight,
      down_scales
    )

    refute sanitized.key?("model.layers.0.mlp.experts.gate_up_proj.weight")
    refute sanitized.key?("model.layers.0.mlp.experts.gate_up_proj_bias")
    refute sanitized.key?("model.layers.0.mlp.experts.gate_up_proj_scales")
    refute sanitized.key?("model.layers.0.mlp.experts.down_proj_scales")
    assert sanitized.key?("model.layers.0.mlp.experts.gate_proj.weight")
    assert sanitized.key?("model.layers.0.mlp.experts.up_proj.weight")
    assert sanitized.key?("model.layers.0.mlp.experts.gate_proj.bias")
    assert sanitized.key?("model.layers.0.mlp.experts.up_proj.bias")
    assert sanitized.key?("model.layers.0.mlp.experts.gate_proj.scales")
    assert sanitized.key?("model.layers.0.mlp.experts.up_proj.scales")
    assert sanitized.key?("model.layers.0.mlp.experts.down_proj.weight")
    assert sanitized.key?("model.layers.0.mlp.experts.down_proj.scales")
    assert sanitized.key?("model.layers.3.self_attn.q_proj.weight")
    assert_equal [2, 3, 4], gate_weight.shape
    assert_equal [2, 3, 4], up_weight.shape
    assert_equal [2, 3], gate_bias.shape
    assert_equal [2, 3], up_bias.shape
    assert_equal [2, 3, 2], gate_scales.shape
    assert_equal [2, 3, 2], up_scales.shape

    caches = model.make_cache
    assert_equal 4, caches.length
    assert_instance_of MlxLm::RotatingKVCache, caches[0]
    assert_instance_of MlxLm::KVCache, caches[1]
    assert_instance_of MlxLm::RotatingKVCache, caches[2]
    assert_instance_of MlxLm::KVCache, caches[3]
  end
end

class HunyuanGptOssModelsModelsRegisteredAndResolveTest < Minitest::Test
  def test_models_registered_and_resolve
    assert MlxLm::Models::REGISTRY.key?("hunyuan"), "hunyuan should be registered"
    assert MlxLm::Models::REGISTRY.key?("gpt_oss"), "gpt_oss should be registered"

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "hunyuan" })
    assert_equal MlxLm::Models::Hunyuan::Model, model_class
    assert_equal MlxLm::Models::Hunyuan::ModelArgs, args_class

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "gpt_oss" })
    assert_equal MlxLm::Models::GptOss::Model, model_class
    assert_equal MlxLm::Models::GptOss::ModelArgs, args_class
  end
end
