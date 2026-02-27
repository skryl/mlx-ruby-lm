$LOAD_PATH.unshift File.expand_path("../../lib", __dir__)
$LOAD_PATH.unshift File.expand_path("../../mlx-ruby/lib", __dir__)

require "mlx"
require "minitest/autorun"

require_relative "../../lib/mlx_lm/model_args"
require_relative "../../lib/mlx_lm/models"
require_relative "../../lib/mlx_lm/models/cache"
require_relative "../../lib/mlx_lm/models/activations"
require_relative "../../lib/mlx_lm/models/rope_utils"
require_relative "../../lib/mlx_lm/models/gemma2"
require_relative "../../lib/mlx_lm/models/ernie4_5"
require_relative "../../lib/mlx_lm/models/gemma3n"
require_relative "../../lib/mlx_lm/models/ernie4_5_moe"

class Gemma3NErnie45MoeModelsGemma3nTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_gemma3n_construct_forward_shape_and_sanitize_with_shared_config_handling
    shared_config = {
      "model_type" => "gemma3n",
      "hidden_size" => 32,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 4,
      "intermediate_size" => 64,
      "vocab_size" => 111,
      "head_dim" => 8,
      "rms_norm_eps" => 1e-5,
      "rope_theta" => 10_000.0,
      "max_position_embeddings" => 256,
      "final_logit_softcapping" => 20.0,
      "query_pre_attn_scalar" => 64.0,
      "attn_logit_softcapping" => 25.0,
    }

    args = MlxLm::Models::Gemma3n::ModelArgs.from_dict(shared_config)
    refute_same shared_config, args.text_config
    args.text_config["added_by_args"] = true
    refute shared_config.key?("added_by_args")

    model = MlxLm::Models::Gemma3n::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    tokens = @mx.array([[1, 2, 3, 4]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)

    assert_equal [1, 4, 111], output.shape

    weights = {
      "model.vision_tower.patch_embed.weight" => @mx.zeros([1]).astype(@mx.float32),
      "model.audio_tower.blocks.0.weight" => @mx.zeros([1]).astype(@mx.float32),
      "model.embed_audio.proj.weight" => @mx.zeros([1]).astype(@mx.float32),
      "model.embed_vision.proj.weight" => @mx.zeros([1]).astype(@mx.float32),
      "model.language_model.embed_tokens.weight" => @mx.zeros([111, 32]).astype(@mx.float32),
    }
    sanitized = model.sanitize(weights)

    refute sanitized.key?("model.vision_tower.patch_embed.weight")
    refute sanitized.key?("model.audio_tower.blocks.0.weight")
    refute sanitized.key?("model.embed_audio.proj.weight")
    refute sanitized.key?("model.embed_vision.proj.weight")
    assert sanitized.key?("model.language_model.embed_tokens.weight")
  end
end

class Gemma3NErnie45MoeModelsErnie45MoeTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_ernie4_5_moe_construct_forward_shape_and_sanitize_stacks_experts
    args = MlxLm::Models::Ernie45Moe::ModelArgs.from_dict({
      "model_type" => "ernie4_5_moe",
      "hidden_size" => 32,
      "intermediate_size" => 64,
      "max_position_embeddings" => 256,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 4,
      "num_hidden_layers" => 2,
      "rms_norm_eps" => 1e-5,
      "vocab_size" => 101,
      "rope_theta" => 10_000.0,
      "use_bias" => false,
      "tie_word_embeddings" => false,
      "moe_num_experts" => 2,
      "moe_k" => 1,
      "moe_layer_interval" => 1,
      "moe_layer_start_index" => 0,
      "moe_num_shared_experts" => 1,
      "moe_gate_act" => "softmax",
    })

    model = MlxLm::Models::Ernie45Moe::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)

    assert_equal [1, 3, 101], output.shape

    weights = {
      "model.layers.0.mlp.experts.0.gate_proj.weight" => @mx.array([[1.0, 2.0], [3.0, 4.0]], dtype: @mx.float32),
      "model.layers.0.mlp.experts.1.gate_proj.weight" => @mx.array([[5.0, 6.0], [7.0, 8.0]], dtype: @mx.float32),
      "model.layers.0.mlp.experts.0.down_proj.weight" => @mx.array([[0.0, 1.0], [1.0, 0.0]], dtype: @mx.float32),
      "model.layers.0.mlp.experts.1.down_proj.weight" => @mx.array([[2.0, 3.0], [4.0, 5.0]], dtype: @mx.float32),
      "model.layers.0.mlp.experts.0.up_proj.weight" => @mx.array([[1.0, 1.0], [1.0, 1.0]], dtype: @mx.float32),
      "model.layers.0.mlp.experts.1.up_proj.weight" => @mx.array([[2.0, 2.0], [2.0, 2.0]], dtype: @mx.float32),
      "model.layers.0.mtp_block.weight" => @mx.zeros([1]).astype(@mx.float32),
      "model.layers.0.e_score_correction_bias" => @mx.zeros([1]).astype(@mx.float32),
      "model.embed_tokens.weight" => @mx.zeros([101, 32]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)
    gate = sanitized["model.layers.0.mlp.switch_mlp.gate_proj.weight"]
    down = sanitized["model.layers.0.mlp.switch_mlp.down_proj.weight"]
    up = sanitized["model.layers.0.mlp.switch_mlp.up_proj.weight"]
    @mx.eval(gate, down, up)

    assert_equal [2, 2, 2], gate.shape
    assert_equal [2, 2, 2], down.shape
    assert_equal [2, 2, 2], up.shape

    refute sanitized.key?("model.layers.0.mlp.experts.0.gate_proj.weight")
    refute sanitized.key?("model.layers.0.mlp.experts.1.gate_proj.weight")
    refute sanitized.key?("model.layers.0.mtp_block.weight")
    refute sanitized.key?("model.layers.0.e_score_correction_bias")
    assert sanitized.key?("model.embed_tokens.weight")
  end
end

class Gemma3NErnie45MoeModelsTest < Minitest::Test
  def test_models_registered_and_resolve
    assert MlxLm::Models::REGISTRY.key?("gemma3n"), "gemma3n should be registered"
    assert MlxLm::Models::REGISTRY.key?("ernie4_5_moe"), "ernie4_5_moe should be registered"

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "gemma3n" })
    assert_equal MlxLm::Models::Gemma3n::Model, model_class
    assert_equal MlxLm::Models::Gemma3n::ModelArgs, args_class

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "ernie4_5_moe" })
    assert_equal MlxLm::Models::Ernie45Moe::Model, model_class
    assert_equal MlxLm::Models::Ernie45Moe::ModelArgs, args_class
  end
end
