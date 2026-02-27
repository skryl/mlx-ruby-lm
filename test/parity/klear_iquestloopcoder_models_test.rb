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
require_relative "../../lib/mlx_lm/models/klear"
require_relative "../../lib/mlx_lm/models/iquestloopcoder"

class KlearIquestloopcoderModelsTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_klear_construct_forward_shape_and_sanitize_stacks_experts
    args = MlxLm::Models::Klear::ModelArgs.from_dict({
      "model_type" => "Klear",
      "hidden_size" => 32,
      "num_hidden_layers" => 2,
      "intermediate_size" => 64,
      "num_attention_heads" => 4,
      "attention_bias" => false,
      "mlp_only_layers" => [],
      "num_experts" => 2,
      "num_experts_per_tok" => 1,
      "decoder_sparse_step" => 1,
      "n_shared_experts" => 1,
      "moe_intermediate_size" => 48,
      "rms_norm_eps" => 1e-5,
      "vocab_size" => 97,
      "num_key_value_heads" => 4,
      "rope_theta" => 10_000.0,
      "max_position_embeddings" => 256,
      "norm_topk_prob" => true,
    })

    model = MlxLm::Models::Klear::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3, 4]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 4, 97], output.shape

    weights = {
      "model.layers.0.mlp.experts.0.gate_proj.weight" => @mx.array((0...24).to_a, dtype: @mx.float32).reshape([6, 4]),
      "model.layers.0.mlp.experts.1.gate_proj.weight" => @mx.array((24...48).to_a, dtype: @mx.float32).reshape([6, 4]),
      "model.layers.0.mlp.experts.0.up_proj.weight" => @mx.array((48...72).to_a, dtype: @mx.float32).reshape([6, 4]),
      "model.layers.0.mlp.experts.1.up_proj.weight" => @mx.array((72...96).to_a, dtype: @mx.float32).reshape([6, 4]),
      "model.layers.0.mlp.experts.0.down_proj.weight" => @mx.array((0...24).to_a, dtype: @mx.float32).reshape([4, 6]),
      "model.layers.0.mlp.experts.1.down_proj.weight" => @mx.array((24...48).to_a, dtype: @mx.float32).reshape([4, 6]),
      "model.layers.0.self_attn.q_proj.weight" => @mx.zeros([32, 32]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)
    stacked_gate = sanitized["model.layers.0.mlp.experts.gate_proj.weight"]
    stacked_up = sanitized["model.layers.0.mlp.experts.up_proj.weight"]
    stacked_down = sanitized["model.layers.0.mlp.experts.down_proj.weight"]
    @mx.eval(stacked_gate, stacked_up, stacked_down)

    refute sanitized.key?("model.layers.0.mlp.experts.0.gate_proj.weight")
    refute sanitized.key?("model.layers.0.mlp.experts.1.gate_proj.weight")
    refute sanitized.key?("model.layers.0.mlp.experts.0.up_proj.weight")
    refute sanitized.key?("model.layers.0.mlp.experts.1.up_proj.weight")
    refute sanitized.key?("model.layers.0.mlp.experts.0.down_proj.weight")
    refute sanitized.key?("model.layers.0.mlp.experts.1.down_proj.weight")
    assert sanitized.key?("model.layers.0.mlp.experts.gate_proj.weight")
    assert sanitized.key?("model.layers.0.mlp.experts.up_proj.weight")
    assert sanitized.key?("model.layers.0.mlp.experts.down_proj.weight")
    assert sanitized.key?("model.layers.0.self_attn.q_proj.weight")
    assert_equal [2, 6, 4], stacked_gate.shape
    assert_equal [2, 6, 4], stacked_up.shape
    assert_equal [2, 4, 6], stacked_down.shape
  end
end

class KlearIquestloopcoderModelsIquestloopcoderConstructForwardShapeAndMakeCacheHalvesTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_iquestloopcoder_construct_forward_shape_and_make_cache_halves
    args = MlxLm::Models::Iquestloopcoder::ModelArgs.from_dict({
      "model_type" => "iquestloopcoder",
      "hidden_size" => 32,
      "num_hidden_layers" => 2,
      "intermediate_size" => 64,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 2,
      "rms_norm_eps" => 1e-5,
      "vocab_size" => 89,
      "head_dim" => 8,
      "max_position_embeddings" => 256,
      "attention_bias" => false,
      "mlp_bias" => false,
      "rope_theta" => 10_000.0,
      "tie_word_embeddings" => false,
      "loop_num" => 2,
      "loop_window_size" => 4,
    })

    model = MlxLm::Models::Iquestloopcoder::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 3, 89], output.shape

    cache = model.make_cache
    assert_equal 4, cache.length
    assert_instance_of MlxLm::KVCache, cache[0]
    assert_instance_of MlxLm::KVCache, cache[1]
    assert_instance_of MlxLm::RotatingKVCache, cache[2]
    assert_instance_of MlxLm::RotatingKVCache, cache[3]
  end
end

class KlearIquestloopcoderModelsModelsRegisteredAndResolveTest < Minitest::Test
  def test_models_registered_and_resolve
    assert MlxLm::Models::REGISTRY.key?("Klear"), "Klear should be registered"
    assert MlxLm::Models::REGISTRY.key?("iquestloopcoder"), "iquestloopcoder should be registered"

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "Klear" })
    assert_equal MlxLm::Models::Klear::Model, model_class
    assert_equal MlxLm::Models::Klear::ModelArgs, args_class

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "iquestloopcoder" })
    assert_equal MlxLm::Models::Iquestloopcoder::Model, model_class
    assert_equal MlxLm::Models::Iquestloopcoder::ModelArgs, args_class
  end
end
