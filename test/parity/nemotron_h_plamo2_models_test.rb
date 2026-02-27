# frozen_string_literal: true

require_relative "../test_helper"
require_relative "../../lib/mlx_lm/models/nemotron_h"
require_relative "../../lib/mlx_lm/models/plamo2"

class Phase28HybridLaneATNemotronHTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_nemotron_h_wrapper_construct_forward_shape_sanitize_mapping_and_cache
    args = MlxLm::Models::NemotronH::ModelArgs.from_dict({
      "model_type" => "nemotron_h",
      "vocab_size" => 83,
      "hidden_size" => 32,
      "intermediate_size" => 64,
      "num_hidden_layers" => 2,
      "max_position_embeddings" => 128,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 2,
      "attention_bias" => false,
      "mamba_num_heads" => 4,
      "mamba_head_dim" => 8,
      "conv_kernel" => 3,
      "layer_norm_epsilon" => 1e-5,
      "hybrid_override_pattern" => ["M", "*"],
      "tie_word_embeddings" => true,
    })

    model = MlxLm::Models::NemotronH::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 3, 83], output.shape

    expert0 = @mx.array([[1.0, 2.0], [3.0, 4.0]], dtype: @mx.float32)
    expert1 = @mx.array([[5.0, 6.0], [7.0, 8.0]], dtype: @mx.float32)
    conv_weight = @mx.array((0...24).to_a, dtype: @mx.float32).reshape([4, 1, 6])
    weights = {
      "backbone.layers.0.mixer.experts.0.up_proj.weight" => expert0,
      "backbone.layers.0.mixer.experts.1.up_proj.weight" => expert1,
      "backbone.layers.0.mixer.in_proj.weight" => @mx.zeros([32, 32]).astype(@mx.float32),
      "backbone.layers.0.mixer.conv1d.weight" => conv_weight,
      "backbone.layers.0.norm.weight" => @mx.ones([32]).astype(@mx.float32),
      "backbone.norm_f.weight" => @mx.ones([32]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)
    stacked_experts = sanitized["model.layers.0.mlp_block.switch_mlp.up_proj.weight"]
    mapped_in_proj = sanitized["model.layers.0.temporal_block.linear_x.weight"]
    mapped_conv = sanitized["model.layers.0.temporal_block.conv_1d.weight"]
    mapped_layer_norm = sanitized["model.layers.0.temporal_pre_norm.weight"]
    mapped_final_norm = sanitized["model.final_norm.weight"]
    @mx.eval(stacked_experts, mapped_in_proj, mapped_conv, mapped_layer_norm, mapped_final_norm)

    refute sanitized.key?("backbone.layers.0.mixer.experts.0.up_proj.weight")
    refute sanitized.key?("backbone.layers.0.mixer.experts.1.up_proj.weight")
    refute sanitized.key?("backbone.layers.0.mixer.in_proj.weight")
    refute sanitized.key?("backbone.layers.0.mixer.conv1d.weight")
    refute sanitized.key?("backbone.layers.0.norm.weight")
    refute sanitized.key?("backbone.norm_f.weight")
    assert sanitized.key?("model.layers.0.mlp_block.switch_mlp.up_proj.weight")
    assert sanitized.key?("model.layers.0.temporal_block.linear_x.weight")
    assert sanitized.key?("model.layers.0.temporal_block.conv_1d.weight")
    assert sanitized.key?("model.layers.0.temporal_pre_norm.weight")
    assert sanitized.key?("model.final_norm.weight")
    assert_equal [2, 2, 2], stacked_experts.shape
    assert_equal expert0.to_a, stacked_experts[0].to_a
    assert_equal expert1.to_a, stacked_experts[1].to_a
    assert_equal [4, 6, 1], mapped_conv.shape

    cache = model.make_cache
    assert_equal 2, cache.length
  end
end

class Phase28HybridLaneATPlamo2Test < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_plamo2_wrapper_construct_forward_shape_sanitize_mapping_and_cache
    args = MlxLm::Models::Plamo2::ModelArgs.from_dict({
      "model_type" => "plamo2",
      "hidden_size" => 32,
      "num_hidden_layers" => 3,
      "rms_norm_eps" => 1e-5,
      "tie_word_embeddings" => true,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 2,
      "hidden_size_per_head" => 8,
      "max_position_embeddings" => 128,
      "attention_window_size" => 16,
      "mamba_d_conv" => 3,
      "mamba_step" => 2,
      "mamba_enabled" => true,
      "intermediate_size" => 64,
      "vocab_size" => 71,
    })

    model = MlxLm::Models::Plamo2::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 3, 71], output.shape

    gate_up = @mx.array((0...16).to_a, dtype: @mx.float32).reshape([8, 2])
    expected_gate, expected_up = @mx.split(gate_up, [4], 0)
    conv_weight = @mx.array((0...24).to_a, dtype: @mx.float32).reshape([4, 1, 6])
    weights = {
      "model.layers.layers.0.mlp.gate_up_proj.weight" => gate_up,
      "model.layers.layers.0.mlp.down_proj.weight" => @mx.zeros([2, 4]).astype(@mx.float32),
      "model.layers.layers.0.mixer.in_proj.weight" => @mx.zeros([32, 32]).astype(@mx.float32),
      "model.layers.layers.0.mixer.conv1d.weight" => conv_weight,
      "model.layers.layers.0.pre_mixer_norm.weight" => @mx.ones([32]).astype(@mx.float32),
      "model.layers.layers.0.pre_mlp_norm.weight" => @mx.ones([32]).astype(@mx.float32),
      "model.norm.weight" => @mx.ones([32]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)
    mapped_gate = sanitized["model.layers.0.mlp_block.gate_proj.weight"]
    mapped_up = sanitized["model.layers.0.mlp_block.up_proj.weight"]
    mapped_down = sanitized["model.layers.0.mlp_block.down_proj.weight"]
    mapped_in_proj = sanitized["model.layers.0.temporal_block.linear_x.weight"]
    mapped_conv = sanitized["model.layers.0.temporal_block.conv_1d.weight"]
    mapped_pre_mixer_norm = sanitized["model.layers.0.temporal_pre_norm.weight"]
    mapped_pre_mlp_norm = sanitized["model.layers.0.channel_pre_norm.weight"]
    mapped_final_norm = sanitized["model.final_norm.weight"]
    @mx.eval(mapped_gate, mapped_up, mapped_down, mapped_in_proj, mapped_conv, mapped_pre_mixer_norm, mapped_pre_mlp_norm, mapped_final_norm)

    refute sanitized.key?("model.layers.layers.0.mlp.gate_up_proj.weight")
    refute sanitized.key?("model.layers.layers.0.mixer.in_proj.weight")
    refute sanitized.key?("model.layers.layers.0.mixer.conv1d.weight")
    refute sanitized.key?("model.layers.layers.0.pre_mixer_norm.weight")
    refute sanitized.key?("model.layers.layers.0.pre_mlp_norm.weight")
    refute sanitized.key?("model.norm.weight")
    assert sanitized.key?("model.layers.0.mlp_block.gate_proj.weight")
    assert sanitized.key?("model.layers.0.mlp_block.up_proj.weight")
    assert sanitized.key?("model.layers.0.mlp_block.down_proj.weight")
    assert sanitized.key?("model.layers.0.temporal_block.linear_x.weight")
    assert sanitized.key?("model.layers.0.temporal_block.conv_1d.weight")
    assert sanitized.key?("model.layers.0.temporal_pre_norm.weight")
    assert sanitized.key?("model.layers.0.channel_pre_norm.weight")
    assert sanitized.key?("model.final_norm.weight")
    assert_equal expected_gate.to_a, mapped_gate.to_a
    assert_equal expected_up.to_a, mapped_up.to_a
    assert_equal [4, 6, 1], mapped_conv.shape

    cache = model.make_cache
    assert_equal 3, cache.length
  end
end

class Phase28HybridLaneATRegistryTest < Minitest::Test
  def test_models_registered_and_resolve
    assert MlxLm::Models::REGISTRY.key?("nemotron_h"), "nemotron_h should be registered"
    assert MlxLm::Models::REGISTRY.key?("plamo2"), "plamo2 should be registered"

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "nemotron_h" })
    assert_equal MlxLm::Models::NemotronH::Model, model_class
    assert_equal MlxLm::Models::NemotronH::ModelArgs, args_class

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "plamo2" })
    assert_equal MlxLm::Models::Plamo2::Model, model_class
    assert_equal MlxLm::Models::Plamo2::ModelArgs, args_class
  end
end
