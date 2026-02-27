require_relative "../test_helper"
require_relative "../../lib/mlx_lm/models/afmoe"
require_relative "../../lib/mlx_lm/models/bailing_moe"

class Phase26DenseLaneAKAfmoeTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_afmoe_construct_forward_shape_sanitize_and_make_cache
    args = MlxLm::Models::Afmoe::ModelArgs.from_dict({
      "model_type" => "afmoe",
      "vocab_size" => 73,
      "hidden_size" => 32,
      "intermediate_size" => 64,
      "moe_intermediate_size" => 24,
      "num_hidden_layers" => 4,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 2,
      "head_dim" => 8,
      "max_position_embeddings" => 128,
      "rms_norm_eps" => 1e-5,
      "rope_theta" => 10_000.0,
      "tie_word_embeddings" => false,
      "num_experts" => 2,
      "num_experts_per_tok" => 2,
      "num_shared_experts" => 1,
      "num_dense_layers" => 1,
      "route_norm" => true,
      "route_scale" => 1.0,
      "score_func" => "sigmoid",
      "n_group" => 1,
      "topk_group" => 1,
      "sliding_window" => 4,
      "mup_enabled" => false,
      "layer_types" => [
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
      ],
    })

    model = MlxLm::Models::Afmoe::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3, 4]], @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 4, 73], output.shape

    weights = {
      "model.layers.0.self_attn.rotary_emb.inv_freq" => @mx.zeros([4]).astype(@mx.float32),
      "model.layers.2.mlp.experts.0.gate_proj.weight" => @mx.array([[1.0, 2.0], [3.0, 4.0]], dtype: @mx.float32),
      "model.layers.2.mlp.experts.1.gate_proj.weight" => @mx.array([[5.0, 6.0], [7.0, 8.0]], dtype: @mx.float32),
      "model.layers.2.mlp.experts.0.down_proj.weight" => @mx.array([[0.0, 1.0], [1.0, 0.0]], dtype: @mx.float32),
      "model.layers.2.mlp.experts.1.down_proj.weight" => @mx.array([[2.0, 3.0], [4.0, 5.0]], dtype: @mx.float32),
      "model.layers.2.mlp.experts.0.up_proj.weight" => @mx.array([[1.0, 1.0], [1.0, 1.0]], dtype: @mx.float32),
      "model.layers.2.mlp.experts.1.up_proj.weight" => @mx.array([[2.0, 2.0], [2.0, 2.0]], dtype: @mx.float32),
      "model.embed_tokens.weight" => @mx.zeros([73, 32]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)
    gate = sanitized["model.layers.2.mlp.experts.gate_proj.weight"]
    down = sanitized["model.layers.2.mlp.experts.down_proj.weight"]
    up = sanitized["model.layers.2.mlp.experts.up_proj.weight"]
    @mx.eval(gate, down, up)

    assert_equal [2, 2, 2], gate.shape
    assert_equal [2, 2, 2], down.shape
    assert_equal [2, 2, 2], up.shape

    refute sanitized.key?("model.layers.0.self_attn.rotary_emb.inv_freq")
    refute sanitized.key?("model.layers.2.mlp.experts.0.gate_proj.weight")
    refute sanitized.key?("model.layers.2.mlp.experts.1.gate_proj.weight")
    assert sanitized.key?("model.embed_tokens.weight")

    cache = model.make_cache
    assert_equal 4, cache.length
    assert_instance_of MlxLm::KVCache, cache[0]
    assert_instance_of MlxLm::RotatingKVCache, cache[1]
    assert_instance_of MlxLm::RotatingKVCache, cache[2]
    assert_instance_of MlxLm::KVCache, cache[3]
  end
end

class Phase26DenseLaneAKBailingMoeTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_bailing_moe_construct_forward_shape_and_sanitize
    args = MlxLm::Models::BailingMoe::ModelArgs.from_dict({
      "model_type" => "bailing_moe",
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
      "use_bias" => false,
      "use_qkv_bias" => false,
      "tie_word_embeddings" => false,
      "score_function" => "softmax",
      "n_group" => 1,
      "topk_group" => 1,
      "moe_router_enable_expert_bias" => true,
      "moe_router_enable_shared_expert" => true,
    })

    model = MlxLm::Models::BailingMoe::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3]], @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 3, 79], output.shape

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
    refute sanitized.key?("model.layers.1.mlp.gate.weight")
    refute sanitized.key?("model.layers.1.mlp.gate.bias")
    refute sanitized.key?("model.layers.1.mlp.experts.0.gate_proj.weight")
    refute sanitized.key?("model.layers.1.mlp.experts.1.gate_proj.weight")
    assert sanitized.key?("model.word_embeddings.weight")
  end
end

class Phase26DenseLaneAKRegistryTest < Minitest::Test
  def test_models_registered_and_resolve
    assert MlxLm::Models::REGISTRY.key?("afmoe"), "afmoe should be registered"
    assert MlxLm::Models::REGISTRY.key?("bailing_moe"), "bailing_moe should be registered"

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "afmoe" })
    assert_equal MlxLm::Models::Afmoe::Model, model_class
    assert_equal MlxLm::Models::Afmoe::ModelArgs, args_class

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "bailing_moe" })
    assert_equal MlxLm::Models::BailingMoe::Model, model_class
    assert_equal MlxLm::Models::BailingMoe::ModelArgs, args_class
  end
end
