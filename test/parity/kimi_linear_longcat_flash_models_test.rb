require_relative "../test_helper"
require_relative "../../lib/mlx_lm/models/kimi_linear"
require_relative "../../lib/mlx_lm/models/longcat_flash"

class Phase44DenseLaneARKimiLinearTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_kimi_linear_wrapper_from_dict_forward_shape_and_sanitize_mapping_and_stacking
    args = MlxLm::Models::KimiLinear::ModelArgs.from_dict({
      "model_type" => "kimi_linear",
      "vocab_size" => 73,
      "hidden_dim" => 32,
      "ffn_hidden_size" => 64,
      "moe_intermediate_size" => 24,
      "num_layers" => 2,
      "num_heads" => 4,
      "num_kv_heads" => 2,
      "num_local_experts" => 2,
      "n_shared_experts" => 1,
      "top_k" => 1,
      "norm_topk_prob" => true,
      "max_position_embeddings" => 128,
      "rms_norm_eps" => 1e-5,
      "rope_theta" => 10_000.0,
      "first_k_dense_replace" => 0,
      "layer_group_size" => 1,
      "group_norm_size" => 1,
      "use_bias" => false,
      "use_qkv_bias" => false,
      "tie_word_embeddings" => false,
      "score_func" => "softmax",
      "n_group" => 1,
      "topk_group" => 1,
      "moe_router_enable_expert_bias" => true,
      "moe_router_enable_shared_expert" => true,
    })

    assert_equal 32, args.hidden_size
    assert_equal 2, args.num_hidden_layers
    assert_equal 4, args.num_attention_heads
    assert_equal 2, args.num_experts

    model = MlxLm::Models::KimiLinear::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 3, 73], output.shape
    assert_equal 2, model.layers.length
    assert_nil model.make_cache

    gate_weight = @mx.array([[1.0, 2.0], [3.0, 4.0]], dtype: @mx.float32)
    weights = {
      "model.layers.0.mlp.experts.0.gate_proj.weight" => @mx.array([[1.0, 2.0], [3.0, 4.0]], dtype: @mx.float32),
      "model.layers.0.mlp.experts.1.gate_proj.weight" => @mx.array([[5.0, 6.0], [7.0, 8.0]], dtype: @mx.float32),
      "model.layers.0.mlp.router.weight" => gate_weight,
      "model.layers.0.mlp.router.bias" => @mx.array([0.1, -0.2], dtype: @mx.float32),
      "model.embed_tokens.weight" => @mx.zeros([73, 32]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)
    stacked = sanitized["model.layers.0.mlp.switch_mlp.gate_proj.weight"]
    remapped_gate = sanitized["model.layers.0.mlp.gate.gate_proj.weight"]
    remapped_bias = sanitized["model.layers.0.mlp.gate.gate_proj.bias"]
    remapped_embed = sanitized["model.word_embeddings.weight"]
    @mx.eval(stacked, remapped_gate, remapped_bias, remapped_embed)

    refute sanitized.key?("model.layers.0.mlp.experts.0.gate_proj.weight")
    refute sanitized.key?("model.layers.0.mlp.experts.1.gate_proj.weight")
    refute sanitized.key?("model.layers.0.mlp.router.weight")
    refute sanitized.key?("model.layers.0.mlp.router.bias")
    refute sanitized.key?("model.embed_tokens.weight")
    assert sanitized.key?("model.layers.0.mlp.switch_mlp.gate_proj.weight")
    assert sanitized.key?("model.layers.0.mlp.gate.gate_proj.weight")
    assert sanitized.key?("model.layers.0.mlp.gate.gate_proj.bias")
    assert sanitized.key?("model.word_embeddings.weight")
    assert_equal [2, 2, 2], stacked.shape
    assert_equal gate_weight.to_a, remapped_gate.to_a
  end
end

class Phase44DenseLaneARLongcatFlashTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_longcat_flash_wrapper_from_dict_forward_shape_and_sanitize_mapping_and_stacking
    args = MlxLm::Models::LongcatFlash::ModelArgs.from_dict({
      "model_type" => "longcat_flash",
      "vocab_size" => 71,
      "hidden_dim" => 32,
      "ffn_hidden_size" => 64,
      "moe_intermediate_size" => 16,
      "num_layers" => 2,
      "num_heads" => 4,
      "num_kv_heads" => 4,
      "num_local_experts" => 2,
      "num_shared_experts" => 1,
      "routed_scaling_factor" => 1.0,
      "kv_lora_rank" => 8,
      "q_lora_rank" => 8,
      "qk_rope_head_dim" => 8,
      "qk_nope_head_dim" => 8,
      "v_head_dim" => 8,
      "topk_method" => "noaux_tc",
      "score_function" => "sigmoid",
      "norm_topk_prob" => true,
      "n_group" => 1,
      "topk_group" => 1,
      "top_k" => 1,
      "moe_layer_freq" => 1,
      "first_k_dense_replace" => 0,
      "max_position_embeddings" => 128,
      "rms_norm_eps" => 1e-5,
      "rope_theta" => 10_000.0,
      "attention_bias" => false,
      "partial_rotary_factor" => 1.0,
      "tie_word_embeddings" => false,
    })

    assert_equal 32, args.hidden_size
    assert_equal 2, args.num_hidden_layers
    assert_equal 4, args.num_attention_heads
    assert_equal 2, args.n_routed_experts

    model = MlxLm::Models::LongcatFlash::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 3, 71], output.shape
    assert_equal 2, model.layers.length
    assert_nil model.make_cache

    embed_q_weight = @mx.array([[1.0, 2.0], [3.0, 4.0]], dtype: @mx.float32)
    weights = {
      "model.layers.0.attention.embed_q.weight" => embed_q_weight,
      "model.layers.0.mlp.experts.0.up_proj.weight" => @mx.array([[1.0, 2.0], [3.0, 4.0]], dtype: @mx.float32),
      "model.layers.0.mlp.experts.1.up_proj.weight" => @mx.array([[5.0, 6.0], [7.0, 8.0]], dtype: @mx.float32),
      "model.layers.2.mlp.up_proj.weight" => @mx.zeros([2, 2]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)
    mapped_q_proj = sanitized["model.layers.0.self_attn.q_proj.weight"]
    stacked = sanitized["model.layers.0.mlp.switch_mlp.up_proj.weight"]
    @mx.eval(mapped_q_proj, stacked)

    refute sanitized.key?("model.layers.0.attention.embed_q.weight")
    refute sanitized.key?("model.layers.0.mlp.experts.0.up_proj.weight")
    refute sanitized.key?("model.layers.0.mlp.experts.1.up_proj.weight")
    refute sanitized.key?("model.layers.2.mlp.up_proj.weight")
    assert sanitized.key?("model.layers.0.self_attn.q_proj.weight")
    assert sanitized.key?("model.layers.0.mlp.switch_mlp.up_proj.weight")
    assert_equal embed_q_weight.to_a, mapped_q_proj.to_a
    assert_equal [2, 2, 2], stacked.shape
  end
end

class Phase44DenseLaneARRegistryTest < Minitest::Test
  def test_models_registered_and_resolve
    assert MlxLm::Models::REGISTRY.key?("kimi_linear"), "kimi_linear should be registered"
    assert MlxLm::Models::REGISTRY.key?("longcat_flash"), "longcat_flash should be registered"

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "kimi_linear" })
    assert_equal MlxLm::Models::KimiLinear::Model, model_class
    assert_equal MlxLm::Models::KimiLinear::ModelArgs, args_class

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "longcat_flash" })
    assert_equal MlxLm::Models::LongcatFlash::Model, model_class
    assert_equal MlxLm::Models::LongcatFlash::ModelArgs, args_class
  end
end
