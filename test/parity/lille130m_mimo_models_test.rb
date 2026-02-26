$LOAD_PATH.unshift File.expand_path("../../lib", __dir__)
$LOAD_PATH.unshift File.expand_path("../../mlx-ruby/lib", __dir__)

require "mlx"
require "minitest/autorun"

require_relative "../../lib/mlx_lm/model_args"
require_relative "../../lib/mlx_lm/models"
require_relative "../../lib/mlx_lm/models/cache"
require_relative "../../lib/mlx_lm/models/activations"
require_relative "../../lib/mlx_lm/models/rope_utils"
require_relative "../../lib/mlx_lm/models/lille_130m"
require_relative "../../lib/mlx_lm/models/mimo"

class Phase21DenseLaneVLille130mTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_lille_130m_construct_forward_shape_and_sanitize_rotary_weights
    args = MlxLm::Models::Lille130m::ModelArgs.from_dict({
      "model_type" => "lille-130m",
      "block_size" => 128,
      "layer_norm_eps" => 1e-5,
      "n_embd" => 96,
      "n_head" => 4,
      "n_kv_heads" => 2,
      "n_layer" => 2,
      "rope_theta" => 10_000.0,
      "vocab_size" => 89,
      "tie_word_embeddings" => true,
    })

    model = MlxLm::Models::Lille130m::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    tokens = @mx.array([[1, 2, 3, 4]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 4, 89], output.shape
    assert_equal 2, model.layers.length

    weights = {
      "transformer.layers.0.attention.rotary_emb.inv_freq" => @mx.zeros([8]).astype(@mx.float32),
      "transformer.layers.0.feed_forward.gate_proj.weight" => @mx.zeros([1]).astype(@mx.float32),
    }
    sanitized = model.sanitize(weights)

    refute sanitized.key?("transformer.layers.0.attention.rotary_emb.inv_freq")
    assert sanitized.key?("transformer.layers.0.feed_forward.gate_proj.weight")
  end
end

class Phase21DenseLaneVMimoTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_mimo_construct_forward_shape_and_sanitize_tied_embeddings_and_mtp_weights
    args = MlxLm::Models::Mimo::ModelArgs.from_dict({
      "model_type" => "mimo",
      "hidden_size" => 32,
      "num_hidden_layers" => 2,
      "intermediate_size" => 64,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 2,
      "rms_norm_eps" => 1e-5,
      "vocab_size" => 101,
      "max_position_embeddings" => 128,
      "rope_theta" => 10_000.0,
      "rope_traditional" => false,
      "tie_word_embeddings" => true,
      "num_nextn_predict_layers" => 2,
    })

    model = MlxLm::Models::Mimo::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 3, 101], output.shape
    assert_nil model.lm_head

    weights = {
      "lm_head.weight" => @mx.zeros([101, 32]).astype(@mx.float32),
      "model.layers.0.self_attn.rotary_emb.inv_freq" => @mx.zeros([8]).astype(@mx.float32),
      "model.mtp_layers.0.proj.weight" => @mx.zeros([1]).astype(@mx.float32),
      "model.embed_tokens.weight" => @mx.zeros([101, 32]).astype(@mx.float32),
    }
    sanitized = model.sanitize(weights)

    refute sanitized.key?("lm_head.weight")
    refute sanitized.key?("model.layers.0.self_attn.rotary_emb.inv_freq")
    refute sanitized.key?("model.mtp_layers.0.proj.weight")
    assert sanitized.key?("model.embed_tokens.weight")
  end
end

class Phase21DenseLaneVRegistryTest < Minitest::Test
  def test_models_registered_and_resolve
    assert MlxLm::Models::REGISTRY.key?("lille-130m"), "lille-130m should be registered"
    assert MlxLm::Models::REGISTRY.key?("mimo"), "mimo should be registered"

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "lille-130m" })
    assert_equal MlxLm::Models::Lille130m::Model, model_class
    assert_equal MlxLm::Models::Lille130m::ModelArgs, args_class

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "mimo" })
    assert_equal MlxLm::Models::Mimo::Model, model_class
    assert_equal MlxLm::Models::Mimo::ModelArgs, args_class
  end
end
