$LOAD_PATH.unshift File.expand_path("../../lib", __dir__)
$LOAD_PATH.unshift File.expand_path("../../mlx-ruby/lib", __dir__)

require "mlx"
require "minitest/autorun"

require_relative "../../lib/mlx_lm/model_args"
require_relative "../../lib/mlx_lm/models"
require_relative "../../lib/mlx_lm/models/activations"
require_relative "../../lib/mlx_lm/models/rope_utils"
require_relative "../../lib/mlx_lm/models/switch_layers"
require_relative "../../lib/mlx_lm/models/phixtral"
require_relative "../../lib/mlx_lm/models/minicpm3"

class PhixtralMinicpm3ModelsTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_phixtral_construct_forward_shape_and_sanitize_stacks_experts
    args = MlxLm::Models::Phixtral::ModelArgs.from_dict({
      "model_type" => "phixtral",
      "num_vocab" => 79,
      "model_dim" => 32,
      "num_heads" => 4,
      "num_layers" => 2,
      "rotary_dim" => 4,
      "num_experts_per_tok" => 1,
      "num_local_experts" => 2,
    })

    model = MlxLm::Models::Phixtral::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3, 4]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 4, 79], output.shape

    weights = {
      "transformer.h.0.moe.mlp.0.fc1.weight" => @mx.array((0...24).to_a, dtype: @mx.float32).reshape([6, 4]),
      "transformer.h.0.moe.mlp.1.fc1.weight" => @mx.array((24...48).to_a, dtype: @mx.float32).reshape([6, 4]),
      "transformer.h.0.moe.mlp.0.fc2.weight" => @mx.array((0...24).to_a, dtype: @mx.float32).reshape([4, 6]),
      "transformer.h.0.moe.mlp.1.fc2.weight" => @mx.array((24...48).to_a, dtype: @mx.float32).reshape([4, 6]),
      "transformer.h.0.moe.mlp.0.fc1.bias" => @mx.array((0...6).to_a, dtype: @mx.float32),
      "transformer.h.0.moe.mlp.1.fc1.bias" => @mx.array((6...12).to_a, dtype: @mx.float32),
      "transformer.h.0.moe.mlp.0.fc2.bias" => @mx.array((0...4).to_a, dtype: @mx.float32),
      "transformer.h.0.moe.mlp.1.fc2.bias" => @mx.array((4...8).to_a, dtype: @mx.float32),
      "transformer.h.1.mixer.wqkv.weight" => @mx.zeros([96, 32]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)
    fc1_stacked = sanitized["transformer.h.0.moe.switch_mlp.fc1.weight"]
    fc2_stacked = sanitized["transformer.h.0.moe.switch_mlp.fc2.weight"]
    @mx.eval(fc1_stacked, fc2_stacked)

    refute sanitized.key?("transformer.h.0.moe.mlp.0.fc1.weight")
    refute sanitized.key?("transformer.h.0.moe.mlp.1.fc1.weight")
    refute sanitized.key?("transformer.h.0.moe.mlp.0.fc2.weight")
    refute sanitized.key?("transformer.h.0.moe.mlp.1.fc2.weight")
    refute sanitized.key?("transformer.h.0.moe.mlp.0.fc1.bias")
    refute sanitized.key?("transformer.h.0.moe.mlp.1.fc1.bias")
    refute sanitized.key?("transformer.h.0.moe.mlp.0.fc2.bias")
    refute sanitized.key?("transformer.h.0.moe.mlp.1.fc2.bias")
    assert sanitized.key?("transformer.h.0.moe.switch_mlp.fc1.weight")
    assert sanitized.key?("transformer.h.0.moe.switch_mlp.fc2.weight")
    assert sanitized.key?("transformer.h.0.moe.switch_mlp.fc1.bias")
    assert sanitized.key?("transformer.h.0.moe.switch_mlp.fc2.bias")
    assert sanitized.key?("transformer.h.1.mixer.wqkv.weight")
    assert_equal [2, 6, 4], fc1_stacked.shape
    assert_equal [2, 4, 6], fc2_stacked.shape
  end
end

class PhixtralMinicpm3ModelsM3Test < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_minicpm3_construct_forward_shape_and_sanitize
    args_hash = {
      "model_type" => "minicpm3",
      "hidden_size" => 32,
      "dim_model_base" => 16,
      "num_hidden_layers" => 2,
      "intermediate_size" => 64,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 4,
      "rms_norm_eps" => 1e-5,
      "vocab_size" => 89,
      "q_lora_rank" => 8,
      "qk_nope_head_dim" => 4,
      "qk_rope_head_dim" => 4,
      "kv_lora_rank" => 8,
      "scale_depth" => 1.0,
      "scale_emb" => 1.25,
      "max_position_embeddings" => 256,
      "attention_bias" => false,
      "rope_theta" => 10_000.0,
      "rope_scaling" => {
        "original_max_position_embeddings" => 128,
        "short_factor" => 1.0,
        "long_factor" => 1.0,
      },
      "tie_word_embeddings" => false,
    }

    args = MlxLm::Models::MiniCPM3::ModelArgs.from_dict(args_hash)
    model = MlxLm::Models::MiniCPM3::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 3, 89], output.shape

    weights = {
      "model.embed_tokens.weight" => @mx.zeros([89, 32]).astype(@mx.float32),
      "model.layers.0.self_attn.rotary_emb.inv_freq" => @mx.zeros([8]).astype(@mx.float32),
    }
    sanitized = model.sanitize(weights)

    refute sanitized.key?("model.layers.0.self_attn.rotary_emb.inv_freq")
    assert sanitized.key?("lm_head.weight")
    assert_equal [89, 32], sanitized["lm_head.weight"].shape

    tied_args = MlxLm::Models::MiniCPM3::ModelArgs.from_dict(args_hash.merge("tie_word_embeddings" => true))
    tied_model = MlxLm::Models::MiniCPM3::Model.new(tied_args)
    tied_weights = {
      "model.embed_tokens.weight" => @mx.zeros([89, 32]).astype(@mx.float32),
      "lm_head.weight" => @mx.zeros([89, 32]).astype(@mx.float32),
    }
    tied_sanitized = tied_model.sanitize(tied_weights)
    refute tied_sanitized.key?("lm_head.weight")
  end
end

class PhixtralMinicpm3ModelsModelsRegisteredAndResolveTest < Minitest::Test
  def test_models_registered_and_resolve
    assert MlxLm::Models::REGISTRY.key?("phixtral"), "phixtral should be registered"
    assert MlxLm::Models::REGISTRY.key?("minicpm3"), "minicpm3 should be registered"

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "phixtral" })
    assert_equal MlxLm::Models::Phixtral::Model, model_class
    assert_equal MlxLm::Models::Phixtral::ModelArgs, args_class

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "minicpm3" })
    assert_equal MlxLm::Models::MiniCPM3::Model, model_class
    assert_equal MlxLm::Models::MiniCPM3::ModelArgs, args_class
  end
end
