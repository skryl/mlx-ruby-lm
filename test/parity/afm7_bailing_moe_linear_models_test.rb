require_relative "../test_helper"
require_relative "../../lib/mlx_lm/models/afmoe"
require_relative "../../lib/mlx_lm/models/bailing_moe"
require_relative "../../lib/mlx_lm/models/afm7"
require_relative "../../lib/mlx_lm/models/bailing_moe_linear"

class Phase27DenseLaneAOAfm7Test < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_afm7_construct_forward_shape_sanitize_passthrough_and_predicates
    args = MlxLm::Models::Afm7::ModelArgs.from_dict({
      "model_type" => "afm7",
      "vocab_size" => 67,
      "hidden_dim" => 32,
      "num_layers" => 3,
      "num_kv_reuse_layers" => 1,
      "num_heads" => 4,
      "num_kv_heads" => 2,
      "hidden_dim_scale_factor" => 2.0,
      "rms_norm_eps" => 1e-5,
      "rope_theta" => 10_000.0,
      "max_position_embeddings" => 128,
      "tie_word_embeddings" => true,
    })

    model = MlxLm::Models::Afm7::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 3, 67], output.shape
    assert_equal 3, model.layers.length

    weights = {
      "model.layers.0.self_attn.rotary_emb.inv_freq" => @mx.zeros([4]).astype(@mx.float32),
      "model.layers.0.self_attn.q_proj.weight" => @mx.zeros([32, 32]).astype(@mx.float32),
      "model.embed_tokens.weight" => @mx.zeros([67, 32]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)
    refute sanitized.key?("model.layers.0.self_attn.rotary_emb.inv_freq")
    assert sanitized.key?("model.layers.0.self_attn.q_proj.weight")
    assert sanitized.key?("model.embed_tokens.weight")

    cast_predicate = model.cast_predicate
    quant_predicate = model.quant_predicate
    assert_equal false, cast_predicate.call("model.layers.0.mlp.expert_bias")
    assert_equal true, cast_predicate.call("model.layers.0.self_attn.q_proj.weight")
    assert_equal({group_size: 64, bits: 8}, quant_predicate.call("model.layers.2.mlp.router.gate", nil))
    assert_equal true, quant_predicate.call("model.layers.2.mlp.experts.gate_proj", nil)
  end
end

class Phase27DenseLaneAOBailingMoeLinearTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_bailing_moe_linear_construct_forward_shape_sanitize_stack_and_predicates
    args = MlxLm::Models::BailingMoeLinear::ModelArgs.from_dict({
      "model_type" => "bailing_moe_linear",
      "hidden_size" => 32,
      "intermediate_size" => 64,
      "max_position_embeddings" => 128,
      "moe_intermediate_size" => 24,
      "num_experts" => 2,
      "num_shared_experts" => 1,
      "norm_topk_prob" => true,
      "num_attention_heads" => 4,
      "num_experts_per_tok" => 2,
      "num_hidden_layers" => 3,
      "num_key_value_heads" => 2,
      "rms_norm_eps" => 1e-5,
      "rope_theta" => 10_000.0,
      "vocab_size" => 79,
      "first_k_dense_replace" => 1,
      "layer_group_size" => 2,
      "group_norm_size" => 1,
      "use_bias" => false,
      "use_qkv_bias" => false,
      "tie_word_embeddings" => false,
      "score_function" => "softmax",
      "n_group" => 1,
      "topk_group" => 1,
      "moe_router_enable_expert_bias" => true,
      "moe_router_enable_shared_expert" => true,
    })

    model = MlxLm::Models::BailingMoeLinear::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3, 4]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 4, 79], output.shape
    assert_equal 3, model.layers.length

    weights = {
      "model.layers.1.mlp.experts.0.gate_proj.weight" => @mx.array([[1.0, 2.0], [3.0, 4.0]], dtype: @mx.float32),
      "model.layers.1.mlp.experts.1.gate_proj.weight" => @mx.array([[5.0, 6.0], [7.0, 8.0]], dtype: @mx.float32),
      "model.layers.1.mlp.experts.0.down_proj.weight" => @mx.array([[0.0, 1.0], [1.0, 0.0]], dtype: @mx.float32),
      "model.layers.1.mlp.experts.1.down_proj.weight" => @mx.array([[2.0, 3.0], [4.0, 5.0]], dtype: @mx.float32),
      "model.layers.1.mlp.experts.0.up_proj.weight" => @mx.array([[1.0, 1.0], [1.0, 1.0]], dtype: @mx.float32),
      "model.layers.1.mlp.experts.1.up_proj.weight" => @mx.array([[2.0, 2.0], [2.0, 2.0]], dtype: @mx.float32),
      "model.layers.1.mlp.gate.weight" => @mx.array([[1.0, 0.0], [0.0, 1.0]], dtype: @mx.float32),
      "model.layers.1.mlp.gate.bias" => @mx.array([0.1, 0.2], dtype: @mx.float32),
      "model.word_embeddings.weight" => @mx.zeros([79, 32]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)
    gate = sanitized["model.layers.1.mlp.switch_mlp.gate_proj.weight"]
    down = sanitized["model.layers.1.mlp.switch_mlp.down_proj.weight"]
    up = sanitized["model.layers.1.mlp.switch_mlp.up_proj.weight"]
    @mx.eval(gate, down, up)

    assert_equal [2, 2, 2], gate.shape
    assert_equal [2, 2, 2], down.shape
    assert_equal [2, 2, 2], up.shape
    assert sanitized.key?("model.layers.1.mlp.gate.gate_proj.weight")
    assert sanitized.key?("model.layers.1.mlp.gate.gate_proj.bias")
    refute sanitized.key?("model.layers.1.mlp.experts.0.gate_proj.weight")
    refute sanitized.key?("model.layers.1.mlp.experts.1.gate_proj.weight")
    refute sanitized.key?("model.layers.1.mlp.gate.weight")
    refute sanitized.key?("model.layers.1.mlp.gate.bias")
    assert sanitized.key?("model.word_embeddings.weight")

    cast_predicate = model.cast_predicate
    quant_predicate = model.quant_predicate
    assert_equal false, cast_predicate.call("model.layers.1.mlp.gate.expert_bias")
    assert_equal true, cast_predicate.call("model.layers.1.mlp.gate.gate_proj.weight")
    assert_equal({group_size: 64, bits: 8}, quant_predicate.call("model.layers.1.mlp.gate.gate_proj", nil))
    assert_equal true, quant_predicate.call("model.layers.1.mlp.switch_mlp.gate_proj", nil)
  end
end

class Phase27DenseLaneAORegistryTest < Minitest::Test
  def test_models_registered_and_resolve
    assert MlxLm::Models::REGISTRY.key?("afm7"), "afm7 should be registered"
    assert MlxLm::Models::REGISTRY.key?("bailing_moe_linear"), "bailing_moe_linear should be registered"

    model_class, args_class = MlxLm::Models.get_classes({"model_type" => "afm7"})
    assert_equal MlxLm::Models::Afm7::Model, model_class
    assert_equal MlxLm::Models::Afm7::ModelArgs, args_class

    model_class, args_class = MlxLm::Models.get_classes({"model_type" => "bailing_moe_linear"})
    assert_equal MlxLm::Models::BailingMoeLinear::Model, model_class
    assert_equal MlxLm::Models::BailingMoeLinear::ModelArgs, args_class
  end
end
