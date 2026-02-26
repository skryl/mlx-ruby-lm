$LOAD_PATH.unshift File.expand_path("../../lib", __dir__)
$LOAD_PATH.unshift File.expand_path("../../mlx-ruby/lib", __dir__)

require "mlx"
require "minitest/autorun"

require_relative "../../lib/mlx_lm/model_args"
require_relative "../../lib/mlx_lm/models"
require_relative "../../lib/mlx_lm/models/activations"
require_relative "../../lib/mlx_lm/models/rope_utils"
require_relative "../../lib/mlx_lm/models/switch_layers"
require_relative "../../lib/mlx_lm/models/phi3small"
require_relative "../../lib/mlx_lm/models/dots1"

class Phase24DenseLaneAGPhi3smallTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_phi3small_construct_forward_shape_and_sanitize_inv_freq_cleanup
    args = MlxLm::Models::Phi3small::ModelArgs.from_dict({
      "model_type" => "phi3small",
      "hidden_size" => 32,
      "dense_attention_every_n_layers" => 1,
      "ff_intermediate_size" => 64,
      "gegelu_limit" => 16.0,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 4,
      "layer_norm_epsilon" => 1e-5,
      "vocab_size" => 97,
      "num_key_value_heads" => 2,
      "mup_attn_multiplier" => 1.0,
      "mup_use_scaling" => true,
      "mup_embedding_multiplier" => 1.0,
      "mup_width_multiplier" => 1.0,
      "rope_embedding_base" => 10_000.0,
      "rope_position_scale" => 1.0,
      "blocksparse_block_size" => 64,
      "blocksparse_num_local_blocks" => 4,
      "blocksparse_vert_stride" => 2,
    })

    model = MlxLm::Models::Phi3small::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 3, 97], output.shape

    weights = {
      "model.layers.0.self_attn.rotary_emb.inv_freq" => @mx.zeros([1]).astype(@mx.float32),
      "model.layers.0.self_attn.position_embeddings.inv_freq" => @mx.zeros([1]).astype(@mx.float32),
      "model.layers.0.self_attn.query_key_value.weight" => @mx.zeros([64, 32]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)
    refute sanitized.key?("model.layers.0.self_attn.rotary_emb.inv_freq")
    refute sanitized.key?("model.layers.0.self_attn.position_embeddings.inv_freq")
    assert sanitized.key?("model.layers.0.self_attn.query_key_value.weight")
  end
end

class Phase24DenseLaneAGDots1Test < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_dots1_construct_forward_shape_and_sanitize_stacks_switch_glu_experts
    args = MlxLm::Models::Dots1::ModelArgs.from_dict({
      "model_type" => "dots1",
      "hidden_size" => 32,
      "num_hidden_layers" => 2,
      "intermediate_size" => 64,
      "num_attention_heads" => 4,
      "rms_norm_eps" => 1e-5,
      "vocab_size" => 101,
      "max_position_embeddings" => 256,
      "num_key_value_heads" => 2,
      "first_k_dense_replace" => 1,
      "moe_intermediate_size" => 48,
      "n_routed_experts" => 2,
      "n_shared_experts" => 1,
      "norm_topk_prob" => true,
      "num_experts_per_tok" => 1,
      "rope_theta" => 10_000.0,
      "routed_scaling_factor" => 1.0,
      "head_dim" => 8,
      "scoring_func" => "noaux_tc",
      "n_group" => 1,
      "topk_group" => 1,
      "attention_bias" => false,
      "mlp_bias" => false,
      "tie_word_embeddings" => false,
    })

    model = MlxLm::Models::Dots1::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3, 4]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 4, 101], output.shape

    weights = {
      "model.layers.1.mlp.experts.0.gate_proj.weight" => @mx.array((0...24).to_a, dtype: @mx.float32).reshape([6, 4]),
      "model.layers.1.mlp.experts.1.gate_proj.weight" => @mx.array((24...48).to_a, dtype: @mx.float32).reshape([6, 4]),
      "model.layers.1.mlp.experts.0.up_proj.weight" => @mx.array((48...72).to_a, dtype: @mx.float32).reshape([6, 4]),
      "model.layers.1.mlp.experts.1.up_proj.weight" => @mx.array((72...96).to_a, dtype: @mx.float32).reshape([6, 4]),
      "model.layers.1.mlp.experts.0.down_proj.weight" => @mx.array((0...24).to_a, dtype: @mx.float32).reshape([4, 6]),
      "model.layers.1.mlp.experts.1.down_proj.weight" => @mx.array((24...48).to_a, dtype: @mx.float32).reshape([4, 6]),
      "model.layers.0.self_attn.rotary_emb.inv_freq" => @mx.zeros([1]).astype(@mx.float32),
      "model.layers.0.self_attn.q_proj.weight" => @mx.zeros([32, 32]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)

    stacked_gate = sanitized["model.layers.1.mlp.experts.gate_proj.weight"]
    stacked_up = sanitized["model.layers.1.mlp.experts.up_proj.weight"]
    stacked_down = sanitized["model.layers.1.mlp.experts.down_proj.weight"]
    @mx.eval(stacked_gate, stacked_up, stacked_down)

    refute sanitized.key?("model.layers.1.mlp.experts.0.gate_proj.weight")
    refute sanitized.key?("model.layers.1.mlp.experts.1.gate_proj.weight")
    refute sanitized.key?("model.layers.1.mlp.experts.0.up_proj.weight")
    refute sanitized.key?("model.layers.1.mlp.experts.1.up_proj.weight")
    refute sanitized.key?("model.layers.1.mlp.experts.0.down_proj.weight")
    refute sanitized.key?("model.layers.1.mlp.experts.1.down_proj.weight")
    refute sanitized.key?("model.layers.0.self_attn.rotary_emb.inv_freq")
    assert sanitized.key?("model.layers.1.mlp.experts.gate_proj.weight")
    assert sanitized.key?("model.layers.1.mlp.experts.up_proj.weight")
    assert sanitized.key?("model.layers.1.mlp.experts.down_proj.weight")
    assert sanitized.key?("model.layers.0.self_attn.q_proj.weight")
    assert_equal [2, 6, 4], stacked_gate.shape
    assert_equal [2, 6, 4], stacked_up.shape
    assert_equal [2, 4, 6], stacked_down.shape
  end
end

class Phase24DenseLaneAGRegistryTest < Minitest::Test
  def test_models_registered_and_resolve
    assert MlxLm::Models::REGISTRY.key?("phi3small"), "phi3small should be registered"
    assert MlxLm::Models::REGISTRY.key?("dots1"), "dots1 should be registered"

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "phi3small" })
    assert_equal MlxLm::Models::Phi3small::Model, model_class
    assert_equal MlxLm::Models::Phi3small::ModelArgs, args_class

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "dots1" })
    assert_equal MlxLm::Models::Dots1::Model, model_class
    assert_equal MlxLm::Models::Dots1::ModelArgs, args_class
  end
end
