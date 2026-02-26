$LOAD_PATH.unshift File.expand_path("../../lib", __dir__)
$LOAD_PATH.unshift File.expand_path("../../mlx-ruby/lib", __dir__)

require "mlx"
require "minitest/autorun"

require_relative "../../lib/mlx_lm/model_args"
require_relative "../../lib/mlx_lm/models"
require_relative "../../lib/mlx_lm/models/cache"
require_relative "../../lib/mlx_lm/models/switch_layers"
require_relative "../../lib/mlx_lm/models/rope_utils"
require_relative "../../lib/mlx_lm/models/minimax"
require_relative "../../lib/mlx_lm/models/nemotron_nas"

class Phase22DenseLaneAMMinimaxTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_minimax_construct_forward_shape_and_sanitize_moe_remap
    args = MlxLm::Models::Minimax::ModelArgs.from_dict({
      "model_type" => "minimax",
      "hidden_size" => 32,
      "intermediate_size" => 64,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 2,
      "max_position_embeddings" => 128,
      "num_experts_per_tok" => 1,
      "num_local_experts" => 2,
      "shared_intermediate_size" => 32,
      "num_hidden_layers" => 2,
      "rms_norm_eps" => 1e-5,
      "rope_theta" => 10_000.0,
      "rotary_dim" => 8,
      "vocab_size" => 97,
      "tie_word_embeddings" => false,
      "use_qk_norm" => true,
    })

    model = MlxLm::Models::Minimax::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    input = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(input)
    @mx.eval(output)

    assert_equal [1, 3, 97], output.shape

    weights = {
      "model.embed_tokens.weight" => @mx.zeros([97, 32]).astype(@mx.float32),
      "model.layers.0.block_sparse_moe.experts.0.w1.weight" => @mx.zeros([4, 4]).astype(@mx.float32),
      "model.layers.0.block_sparse_moe.experts.1.w1.weight" => @mx.ones([4, 4]).astype(@mx.float32),
      "model.layers.0.block_sparse_moe.experts.0.w2.weight" => @mx.zeros([4, 4]).astype(@mx.float32),
      "model.layers.0.block_sparse_moe.experts.1.w2.weight" => @mx.ones([4, 4]).astype(@mx.float32),
      "model.layers.0.block_sparse_moe.experts.0.w3.weight" => @mx.zeros([4, 4]).astype(@mx.float32),
      "model.layers.0.block_sparse_moe.experts.1.w3.weight" => @mx.ones([4, 4]).astype(@mx.float32),
    }
    sanitized = model.sanitize(weights)

    refute sanitized.key?("model.layers.0.block_sparse_moe.experts.0.w1.weight")
    refute sanitized.key?("model.layers.0.block_sparse_moe.experts.1.w1.weight")
    refute sanitized.key?("model.layers.0.block_sparse_moe.experts.0.w2.weight")
    refute sanitized.key?("model.layers.0.block_sparse_moe.experts.1.w2.weight")
    refute sanitized.key?("model.layers.0.block_sparse_moe.experts.0.w3.weight")
    refute sanitized.key?("model.layers.0.block_sparse_moe.experts.1.w3.weight")
    assert sanitized.key?("model.layers.0.block_sparse_moe.switch_mlp.gate_proj.weight")
    assert sanitized.key?("model.layers.0.block_sparse_moe.switch_mlp.down_proj.weight")
    assert sanitized.key?("model.layers.0.block_sparse_moe.switch_mlp.up_proj.weight")
    assert sanitized.key?("model.embed_tokens.weight")
    assert_equal [2, 4, 4], sanitized["model.layers.0.block_sparse_moe.switch_mlp.gate_proj.weight"].shape
    assert_equal [2, 4, 4], sanitized["model.layers.0.block_sparse_moe.switch_mlp.down_proj.weight"].shape
    assert_equal [2, 4, 4], sanitized["model.layers.0.block_sparse_moe.switch_mlp.up_proj.weight"].shape
  end
end

class Phase22DenseLaneAMNemotronNasTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_nemotron_nas_construct_forward_shape_sanitize_and_cache
    args = MlxLm::Models::NemotronNas::ModelArgs.from_dict({
      "model_type" => "nemotron-nas",
      "hidden_size" => 32,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 4,
      "rms_norm_eps" => 1e-5,
      "vocab_size" => 101,
      "hidden_act" => "silu",
      "attention_bias" => false,
      "mlp_bias" => false,
      "rope_theta" => 10_000.0,
      "rope_scaling" => {
        "type" => "linear",
        "factor" => 2.0,
      },
      "max_position_embeddings" => 128,
      "tie_word_embeddings" => true,
      "block_configs" => [
        {
          "attention" => { "n_heads_in_group" => 2 },
          "ffn" => { "ffn_mult" => 1.5 },
        },
        {
          "attention" => { "no_op" => true },
          "ffn" => { "replace_with_linear" => true },
        },
      ],
    })

    assert_equal "linear", args.rope_scaling["rope_type"]

    model = MlxLm::Models::NemotronNas::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    input = @mx.array([[1, 2, 3, 4]], dtype: @mx.int32)
    output = model.call(input)
    @mx.eval(output)

    assert_equal [1, 4, 101], output.shape
    assert_equal 2, model.layers.length
    assert_nil model.layers[1].self_attn

    weights = {
      "lm_head.weight" => @mx.zeros([101, 32]).astype(@mx.float32),
      "model.layers.0.self_attn.rotary_emb.inv_freq" => @mx.zeros([8]).astype(@mx.float32),
      "model.embed_tokens.weight" => @mx.zeros([101, 32]).astype(@mx.float32),
    }
    sanitized = model.sanitize(weights)

    refute sanitized.key?("lm_head.weight")
    refute sanitized.key?("model.layers.0.self_attn.rotary_emb.inv_freq")
    assert sanitized.key?("model.embed_tokens.weight")

    caches = model.make_cache
    assert_equal 1, caches.length
    assert_instance_of MlxLm::KVCache, caches[0]
  end
end

class Phase22DenseLaneAMRegistryTest < Minitest::Test
  def test_models_registered_and_resolve
    assert MlxLm::Models::REGISTRY.key?("minimax"), "minimax should be registered"
    assert MlxLm::Models::REGISTRY.key?("nemotron-nas"), "nemotron-nas should be registered"

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "minimax" })
    assert_equal MlxLm::Models::Minimax::Model, model_class
    assert_equal MlxLm::Models::Minimax::ModelArgs, args_class

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "nemotron-nas" })
    assert_equal MlxLm::Models::NemotronNas::Model, model_class
    assert_equal MlxLm::Models::NemotronNas::ModelArgs, args_class
  end
end
