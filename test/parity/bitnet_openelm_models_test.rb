$LOAD_PATH.unshift File.expand_path("../../lib", __dir__)
$LOAD_PATH.unshift File.expand_path("../../mlx-ruby/lib", __dir__)

require "mlx"
require "minitest/autorun"

require_relative "../../lib/mlx_lm/model_args"
require_relative "../../lib/mlx_lm/models"
require_relative "../../lib/mlx_lm/models/activations"
require_relative "../../lib/mlx_lm/models/bitlinear_layers"
require_relative "../../lib/mlx_lm/models/rope_utils"
require_relative "../../lib/mlx_lm/models/bitnet"
require_relative "../../lib/mlx_lm/models/openelm"

class BitnetOpenelmModelsTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_bitnet_construct_forward_shape_and_sanitize_tied_embeddings
    args = MlxLm::Models::Bitnet::ModelArgs.from_dict({
      "model_type" => "bitnet",
      "hidden_size" => 32,
      "num_hidden_layers" => 2,
      "intermediate_size" => 64,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 2,
      "head_dim" => 8,
      "rms_norm_eps" => 1e-5,
      "vocab_size" => 101,
      "max_position_embeddings" => 256,
      "rope_theta" => 10_000.0,
      "tie_word_embeddings" => true,
    })

    model = MlxLm::Models::Bitnet::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    tokens = @mx.array([[1, 2, 3, 4]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 4, 101], output.shape

    weights = {
      "model.layers.0.self_attn.rotary_emb.inv_freq" => @mx.zeros([8]).astype(@mx.float32),
      "lm_head.weight" => @mx.zeros([101, 32]).astype(@mx.float32),
      "model.embed_tokens.weight" => @mx.zeros([101, 32]).astype(@mx.float32),
    }
    sanitized = model.sanitize(weights)

    refute sanitized.key?("model.layers.0.self_attn.rotary_emb.inv_freq")
    refute sanitized.key?("lm_head.weight")
    assert sanitized.key?("model.embed_tokens.weight")
  end
end

class BitnetOpenelmModelsOpenelmConstructForwardShapeTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_openelm_construct_forward_shape
    args = MlxLm::Models::OpenELM::ModelArgs.from_dict({
      "model_type" => "openelm",
      "head_dim" => 8,
      "num_transformer_layers" => 2,
      "model_dim" => 32,
      "vocab_size" => 89,
      "ffn_dim_divisor" => 8,
      "num_query_heads" => [4, 4],
      "num_kv_heads" => [2, 2],
      "ffn_multipliers" => [2.0, 2.5],
      "normalize_qk_projections" => true,
      "share_input_output_layers" => false,
    })

    model = MlxLm::Models::OpenELM::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 3, 89], output.shape
    assert_equal 2, model.layers.length
    refute_nil model.lm_head
  end
end

class BitnetOpenelmModelsModelsRegisteredAndResolvableTest < Minitest::Test
  def test_models_registered_and_resolvable
    assert MlxLm::Models::REGISTRY.key?("bitnet"), "bitnet should be registered"
    assert MlxLm::Models::REGISTRY.key?("openelm"), "openelm should be registered"

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "bitnet" })
    assert_equal MlxLm::Models::Bitnet::Model, model_class
    assert_equal MlxLm::Models::Bitnet::ModelArgs, args_class

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "openelm" })
    assert_equal MlxLm::Models::OpenELM::Model, model_class
    assert_equal MlxLm::Models::OpenELM::ModelArgs, args_class
  end
end
