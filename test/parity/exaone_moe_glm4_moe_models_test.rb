require_relative "../test_helper"
require_relative "../../lib/mlx_lm/models/exaone_moe"
require_relative "../../lib/mlx_lm/models/glm4_moe"

class ExaoneMoeGlm4MoeModelsTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_exaone_moe_construct_forward_shape_sanitize_and_make_cache
    args = MlxLm::Models::ExaoneMoe::ModelArgs.from_dict({
      "model_type" => "exaone_moe",
      "vocab_size" => 97,
      "hidden_size" => 32,
      "intermediate_size" => 64,
      "moe_intermediate_size" => 16,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 4,
      "head_dim" => 8,
      "num_experts" => 2,
      "num_experts_per_tok" => 1,
      "num_shared_experts" => 1,
      "rms_norm_eps" => 1e-5,
      "max_position_embeddings" => 256,
      "sliding_window" => 2,
      "layer_types" => ["full_attention", "sliding_attention"],
      "is_moe_layer" => [true, false],
      "n_group" => 1,
      "topk_group" => 1,
      "routed_scaling_factor" => 1.0,
      "norm_topk_prob" => true,
      "rope_theta" => 10_000.0,
      "rope_parameters" => { "rope_theta" => 20_000.0 },
      "tie_word_embeddings" => true,
    })

    assert_equal 20_000.0, args.rope_theta

    model = MlxLm::Models::ExaoneMoe::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3, 4]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 4, 97], output.shape

    weights = {
      "mtp.head.weight" => @mx.zeros([2, 2]).astype(@mx.float32),
      "lm_head.weight" => @mx.zeros([97, 32]).astype(@mx.float32),
      "model.layers.0.mlp.e_score_correction_bias" => @mx.array([0.1, -0.1], dtype: @mx.float32),
      "model.layers.0.mlp.experts.0.gate_proj.weight" => @mx.array([[1.0, 2.0], [3.0, 4.0]], dtype: @mx.float32),
      "model.layers.0.mlp.experts.1.gate_proj.weight" => @mx.array([[5.0, 6.0], [7.0, 8.0]], dtype: @mx.float32),
      "model.layers.1.self_attn.q_proj.weight" => @mx.zeros([32, 32]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)
    stacked = sanitized["model.layers.0.mlp.switch_mlp.gate_proj.weight"]
    @mx.eval(stacked)

    refute sanitized.key?("mtp.head.weight")
    refute sanitized.key?("lm_head.weight")
    refute sanitized.key?("model.layers.0.mlp.e_score_correction_bias")
    refute sanitized.key?("model.layers.0.mlp.experts.0.gate_proj.weight")
    refute sanitized.key?("model.layers.0.mlp.experts.1.gate_proj.weight")
    assert sanitized.key?("model.layers.0.mlp.gate.e_score_correction_bias")
    assert sanitized.key?("model.layers.0.mlp.switch_mlp.gate_proj.weight")
    assert sanitized.key?("model.layers.1.self_attn.q_proj.weight")
    assert_equal [2, 2, 2], stacked.shape

    cache = model.make_cache
    assert_equal 2, cache.length
    assert_instance_of MlxLm::KVCache, cache[0]
    assert_instance_of MlxLm::RotatingKVCache, cache[1]
  end
end

class ExaoneMoeGlm4MoeModelsGlm4MoeTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_glm4_moe_construct_forward_shape_and_sanitize
    args = MlxLm::Models::Glm4Moe::ModelArgs.from_dict({
      "model_type" => "glm4_moe",
      "vocab_size" => 101,
      "hidden_size" => 32,
      "intermediate_size" => 64,
      "max_position_embeddings" => 256,
      "moe_intermediate_size" => 16,
      "norm_topk_prob" => true,
      "num_attention_heads" => 4,
      "n_group" => 1,
      "head_dim" => 8,
      "topk_group" => 1,
      "n_shared_experts" => 1,
      "n_routed_experts" => 2,
      "routed_scaling_factor" => 1.0,
      "num_experts_per_tok" => 1,
      "first_k_dense_replace" => 0,
      "num_hidden_layers" => 2,
      "num_key_value_heads" => 4,
      "rms_norm_eps" => 1e-5,
      "rope_theta" => 10_000.0,
      "use_qk_norm" => true,
      "tie_word_embeddings" => false,
      "attention_bias" => false,
      "partial_rotary_factor" => 0.5,
    })

    model = MlxLm::Models::Glm4Moe::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 3, 101], output.shape

    weights = {
      "model.layers.0.mlp.experts.0.up_proj.weight" => @mx.array([[1.0, 2.0], [3.0, 4.0]], dtype: @mx.float32),
      "model.layers.0.mlp.experts.1.up_proj.weight" => @mx.array([[5.0, 6.0], [7.0, 8.0]], dtype: @mx.float32),
      "model.layers.1.self_attn.q_proj.weight" => @mx.zeros([32, 32]).astype(@mx.float32),
      "model.layers.2.mlp.up_proj.weight" => @mx.zeros([2, 2]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)
    stacked = sanitized["model.layers.0.mlp.switch_mlp.up_proj.weight"]
    @mx.eval(stacked)

    refute sanitized.key?("model.layers.0.mlp.experts.0.up_proj.weight")
    refute sanitized.key?("model.layers.0.mlp.experts.1.up_proj.weight")
    refute sanitized.key?("model.layers.2.mlp.up_proj.weight")
    assert sanitized.key?("model.layers.0.mlp.switch_mlp.up_proj.weight")
    assert sanitized.key?("model.layers.1.self_attn.q_proj.weight")
    assert_equal [2, 2, 2], stacked.shape
  end
end

class ExaoneMoeGlm4MoeModelsModelsRegisteredAndResolveTest < Minitest::Test
  def test_models_registered_and_resolve
    assert MlxLm::Models::REGISTRY.key?("exaone_moe"), "exaone_moe should be registered"
    assert MlxLm::Models::REGISTRY.key?("glm4_moe"), "glm4_moe should be registered"

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "exaone_moe" })
    assert_equal MlxLm::Models::ExaoneMoe::Model, model_class
    assert_equal MlxLm::Models::ExaoneMoe::ModelArgs, args_class

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "glm4_moe" })
    assert_equal MlxLm::Models::Glm4Moe::Model, model_class
    assert_equal MlxLm::Models::Glm4Moe::ModelArgs, args_class
  end
end
