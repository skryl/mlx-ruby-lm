require_relative "../test_helper"
require_relative "../../lib/mlx_lm/models/falcon_h1"
require_relative "../../lib/mlx_lm/models/glm4_moe_lite"

class FalconH1Glm4MoeLiteModelsH1Test < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_falcon_h1_wrapper_construct_forward_shape_sanitize_mapping_and_cache
    args = MlxLm::Models::FalconH1::ModelArgs.from_dict({
      "model_type" => "falcon_h1",
      "hidden_size" => 32,
      "intermediate_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 2,
      "mamba_d_conv" => 3,
      "rms_norm_eps" => 1e-5,
      "rope_theta" => 10_000.0,
      "vocab_size" => 89,
      "max_position_embeddings" => 128,
      "attention_window_size" => 4,
      "tie_word_embeddings" => true,
    })

    model = MlxLm::Models::FalconH1::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 3, 89], output.shape

    conv_weight = @mx.array((0...24).to_a, dtype: @mx.float32).reshape([4, 1, 6])
    weights = {
      "model.layers.0.mamba.conv1d.weight" => conv_weight,
      "model.layers.0.feed_forward.gate_proj.weight" => @mx.zeros([64, 32]).astype(@mx.float32),
      "model.layers.0.temporal_block.linear_x.weight" => @mx.ones([32, 32]).astype(@mx.float32),
      "model.final_layernorm.weight" => @mx.ones([32]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)
    sanitized_conv = sanitized["model.layers.0.temporal_block.conv_1d.weight"]
    sanitized_mlp = sanitized["model.layers.0.mlp_block.gate_proj.weight"]
    sanitized_pass_through = sanitized["model.layers.0.temporal_block.linear_x.weight"]
    sanitized_final_norm = sanitized["model.final_norm.weight"]
    @mx.eval(sanitized_conv, sanitized_mlp, sanitized_pass_through, sanitized_final_norm)

    refute sanitized.key?("model.layers.0.mamba.conv1d.weight")
    refute sanitized.key?("model.layers.0.feed_forward.gate_proj.weight")
    refute sanitized.key?("model.final_layernorm.weight")
    assert sanitized.key?("model.layers.0.temporal_block.conv_1d.weight")
    assert sanitized.key?("model.layers.0.mlp_block.gate_proj.weight")
    assert sanitized.key?("model.layers.0.temporal_block.linear_x.weight")
    assert sanitized.key?("model.final_norm.weight")
    assert_equal [4, 6, 1], sanitized_conv.shape
    assert_equal @mx.swapaxes(conv_weight, 1, 2).to_a, sanitized_conv.to_a

    cache = model.make_cache
    assert_equal 2, cache.length
  end
end

class FalconH1Glm4MoeLiteModelsGlm4MoeLiteTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_glm4_moe_lite_wrapper_construct_forward_shape_and_sanitize_mapping_and_stacking
    args = MlxLm::Models::Glm4MoeLite::ModelArgs.from_dict({
      "model_type" => "glm4_moe_lite",
      "vocab_size" => 101,
      "hidden_size" => 32,
      "intermediate_size" => 64,
      "moe_intermediate_size" => 16,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 4,
      "n_shared_experts" => 1,
      "n_routed_experts" => 2,
      "routed_scaling_factor" => 1.0,
      "kv_lora_rank" => 8,
      "q_lora_rank" => 8,
      "qk_rope_head_dim" => 8,
      "qk_nope_head_dim" => 8,
      "v_head_dim" => 8,
      "topk_method" => "noaux_tc",
      "scoring_func" => "sigmoid",
      "norm_topk_prob" => true,
      "n_group" => 1,
      "topk_group" => 1,
      "num_experts_per_tok" => 1,
      "moe_layer_freq" => 1,
      "first_k_dense_replace" => 0,
      "max_position_embeddings" => 256,
      "rms_norm_eps" => 1e-5,
      "rope_theta" => 10_000.0,
      "rope_scaling" => nil,
      "attention_bias" => false,
      "partial_rotary_factor" => 1.0,
      "tie_word_embeddings" => false,
      "num_nextn_predict_layers" => 1,
    })

    model = MlxLm::Models::Glm4MoeLite::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 3, 101], output.shape

    embed_q_weight = @mx.array([[1.0, 2.0], [3.0, 4.0]], dtype: @mx.float32)
    weights = {
      "model.layers.0.self_attn.embed_q.weight" => embed_q_weight,
      "model.layers.0.mlp.experts.0.up_proj.weight" => @mx.array([[1.0, 2.0], [3.0, 4.0]], dtype: @mx.float32),
      "model.layers.0.mlp.experts.1.up_proj.weight" => @mx.array([[5.0, 6.0], [7.0, 8.0]], dtype: @mx.float32),
      "model.layers.1.self_attn.q_proj.weight" => @mx.zeros([32, 32]).astype(@mx.float32),
      "model.layers.2.mlp.up_proj.weight" => @mx.zeros([2, 2]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)
    stacked = sanitized["model.layers.0.mlp.switch_mlp.up_proj.weight"]
    mapped_q_proj = sanitized["model.layers.0.self_attn.q_proj.weight"]
    pass_through_q_proj = sanitized["model.layers.1.self_attn.q_proj.weight"]
    @mx.eval(stacked, mapped_q_proj, pass_through_q_proj)

    refute sanitized.key?("model.layers.0.self_attn.embed_q.weight")
    refute sanitized.key?("model.layers.0.mlp.experts.0.up_proj.weight")
    refute sanitized.key?("model.layers.0.mlp.experts.1.up_proj.weight")
    refute sanitized.key?("model.layers.2.mlp.up_proj.weight")
    assert sanitized.key?("model.layers.0.self_attn.q_proj.weight")
    assert sanitized.key?("model.layers.0.mlp.switch_mlp.up_proj.weight")
    assert sanitized.key?("model.layers.1.self_attn.q_proj.weight")
    assert_equal [2, 2, 2], stacked.shape
    assert_equal embed_q_weight.to_a, mapped_q_proj.to_a
  end
end

class FalconH1Glm4MoeLiteModelsTest < Minitest::Test
  def test_models_registered_and_resolve
    assert MlxLm::Models::REGISTRY.key?("falcon_h1"), "falcon_h1 should be registered"
    assert MlxLm::Models::REGISTRY.key?("glm4_moe_lite"), "glm4_moe_lite should be registered"

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "falcon_h1" })
    assert_equal MlxLm::Models::FalconH1::Model, model_class
    assert_equal MlxLm::Models::FalconH1::ModelArgs, args_class

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "glm4_moe_lite" })
    assert_equal MlxLm::Models::Glm4MoeLite::Model, model_class
    assert_equal MlxLm::Models::Glm4MoeLite::ModelArgs, args_class
  end
end
