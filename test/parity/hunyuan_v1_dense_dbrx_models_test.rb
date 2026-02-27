$LOAD_PATH.unshift File.expand_path("../../lib", __dir__)
$LOAD_PATH.unshift File.expand_path("../../mlx-ruby/lib", __dir__)

require "mlx"
require "minitest/autorun"

require_relative "../../lib/mlx_lm/model_args"
require_relative "../../lib/mlx_lm/models"
require_relative "../../lib/mlx_lm/models/hunyuan_v1_dense"
require_relative "../../lib/mlx_lm/models/dbrx"

class HunyuanV1DenseDbrxModelsV1DenseTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_hunyuan_v1_dense_construct_forward_shape_and_sanitize_tied_embeddings
    args = MlxLm::Models::HunyuanV1Dense::ModelArgs.from_dict({
      "model_type" => "hunyuan_v1_dense",
      "vocab_size" => 97,
      "hidden_size" => 32,
      "num_hidden_layers" => 2,
      "intermediate_size" => 64,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 2,
      "rms_norm_eps" => 1e-5,
      "rope_theta" => 10_000.0,
      "max_position_embeddings" => 256,
      "attention_bias" => false,
      "use_qk_norm" => true,
      "tie_word_embeddings" => true,
      "rope_scaling" => {
        "alpha" => 1.0,
        "factor" => 1.0,
        "type" => "dynamic",
      },
    })

    model = MlxLm::Models::HunyuanV1Dense::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3, 4]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 4, 97], output.shape

    weights = {
      "model.embed_tokens.weight" => @mx.zeros([97, 32]).astype(@mx.float32),
      "lm_head.weight" => @mx.zeros([97, 32]).astype(@mx.float32),
    }
    sanitized = model.sanitize(weights)

    refute sanitized.key?("lm_head.weight")
    assert sanitized.key?("model.embed_tokens.weight")
  end
end

class HunyuanV1DenseDbrxModelsTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_dbrx_construct_forward_shape_and_sanitize_splits_expert_weights
    args = MlxLm::Models::Dbrx::ModelArgs.from_dict({
      "model_type" => "dbrx",
      "vocab_size" => 101,
      "d_model" => 24,
      "n_layers" => 2,
      "n_heads" => 4,
      "attn_config" => {
        "kv_n_heads" => 2,
        "clip_qkv" => 8.0,
        "rope_theta" => 10_000.0,
      },
      "ffn_config" => {
        "ffn_hidden_size" => 16,
        "moe_num_experts" => 2,
        "moe_top_k" => 1,
      },
    })

    model = MlxLm::Models::Dbrx::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 3, 101], output.shape

    expert_w1 = @mx.array((0...12).to_a, dtype: @mx.float32).reshape([4, 3])
    expert_w2 = @mx.array((12...24).to_a, dtype: @mx.float32).reshape([4, 3])
    weights = {
      "transformer.blocks.0.ffn.experts.mlp.w1" => expert_w1,
      "transformer.blocks.0.ffn.experts.mlp.w2" => expert_w2,
      "transformer.blocks.1.norm_attn_norm.attn.out_proj.weight" => @mx.zeros([24, 24]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)

    refute sanitized.key?("transformer.blocks.0.ffn.experts.mlp.w1")
    refute sanitized.key?("transformer.blocks.0.ffn.experts.mlp.w2")
    assert sanitized.key?("transformer.blocks.0.ffn.experts.0.w1.weight")
    assert sanitized.key?("transformer.blocks.0.ffn.experts.1.w1.weight")
    assert sanitized.key?("transformer.blocks.0.ffn.experts.0.w2.weight")
    assert sanitized.key?("transformer.blocks.0.ffn.experts.1.w2.weight")
    assert sanitized.key?("transformer.blocks.1.norm_attn_norm.attn.out_proj.weight")

    assert_equal [2, 3], sanitized["transformer.blocks.0.ffn.experts.0.w1.weight"].shape
    assert_equal [2, 3], sanitized["transformer.blocks.0.ffn.experts.1.w1.weight"].shape
    assert_equal [3, 2], sanitized["transformer.blocks.0.ffn.experts.0.w2.weight"].shape
    assert_equal [3, 2], sanitized["transformer.blocks.0.ffn.experts.1.w2.weight"].shape
  end
end

class HunyuanV1DenseDbrxModelsModelsRegisteredAndResolveTest < Minitest::Test
  def test_models_registered_and_resolve
    assert MlxLm::Models::REGISTRY.key?("hunyuan_v1_dense"), "hunyuan_v1_dense should be registered"
    assert MlxLm::Models::REGISTRY.key?("dbrx"), "dbrx should be registered"

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "hunyuan_v1_dense" })
    assert_equal MlxLm::Models::HunyuanV1Dense::Model, model_class
    assert_equal MlxLm::Models::HunyuanV1Dense::ModelArgs, args_class

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "dbrx" })
    assert_equal MlxLm::Models::Dbrx::Model, model_class
    assert_equal MlxLm::Models::Dbrx::ModelArgs, args_class
  end
end
