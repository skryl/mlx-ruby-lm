$LOAD_PATH.unshift File.expand_path("../../lib", __dir__)
$LOAD_PATH.unshift File.expand_path("../../mlx-ruby/lib", __dir__)

require "mlx"
require "minitest/autorun"

require_relative "../../lib/mlx_lm/model_args"
require_relative "../../lib/mlx_lm/models"
require_relative "../../lib/mlx_lm/models/cache"
require_relative "../../lib/mlx_lm/models/activations"
require_relative "../../lib/mlx_lm/models/rope_utils"
require_relative "../../lib/mlx_lm/models/qwen3"
require_relative "../../lib/mlx_lm/models/llama"
require_relative "../../lib/mlx_lm/models/qwen3_vl"
require_relative "../../lib/mlx_lm/models/smollm3"

class Phase19DenseLaneNQwen3VLTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_qwen3_vl_construct_forward_shape_and_sanitize
    args = MlxLm::Models::Qwen3VL::ModelArgs.from_dict({
      "model_type" => "qwen3_vl",
      "text_config" => {
        "model_type" => "qwen3",
        "hidden_size" => 32,
        "num_hidden_layers" => 2,
        "intermediate_size" => 64,
        "num_attention_heads" => 4,
        "num_key_value_heads" => 2,
        "rms_norm_eps" => 1e-5,
        "vocab_size" => 127,
        "head_dim" => 8,
        "max_position_embeddings" => 256,
        "rope_theta" => 10_000.0,
        "tie_word_embeddings" => false,
      },
    })

    model = MlxLm::Models::Qwen3VL::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    input = @mx.array([[1, 2, 3, 4]], dtype: @mx.int32)
    output = model.call(input)
    @mx.eval(output)

    assert_equal [1, 4, 127], output.shape

    weights = {
      "model.embed_tokens.weight" => @mx.zeros([127, 32]).astype(@mx.float32),
      "lm_head.weight" => @mx.zeros([127, 32]).astype(@mx.float32),
      "language_model.model.layers.0.self_attn.q_proj.weight" => @mx.zeros([32, 32]).astype(@mx.float32),
      "vision_tower.blocks.0.weight" => @mx.zeros([2, 2]).astype(@mx.float32),
    }
    sanitized = model.sanitize(weights)

    refute sanitized.key?("vision_tower.blocks.0.weight")
    assert sanitized.key?("language_model.model.embed_tokens.weight")
    assert sanitized.key?("language_model.lm_head.weight")
    assert sanitized.key?("language_model.model.layers.0.self_attn.q_proj.weight")
  end
end

class Phase19DenseLaneNSmolLM3Test < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_smollm3_construct_forward_shape_and_nope_placement
    args = MlxLm::Models::SmolLM3::ModelArgs.from_dict({
      "model_type" => "smollm3",
      "hidden_size" => 48,
      "num_hidden_layers" => 4,
      "intermediate_size" => 96,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 2,
      "rms_norm_eps" => 1e-5,
      "vocab_size" => 101,
      "tie_word_embeddings" => false,
      "no_rope_layers" => [1, 0, 1, 0],
    })

    model = MlxLm::Models::SmolLM3::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    assert_instance_of MlxLm::Models::SmolLM3::NoPE, model.layers[1].self_attn.rope
    assert_instance_of MlxLm::Models::SmolLM3::NoPE, model.layers[3].self_attn.rope
    refute_instance_of MlxLm::Models::SmolLM3::NoPE, model.layers[0].self_attn.rope
    refute_instance_of MlxLm::Models::SmolLM3::NoPE, model.layers[2].self_attn.rope

    input = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(input)
    @mx.eval(output)

    assert_equal [1, 3, 101], output.shape
  end
end

class Phase19DenseLaneNRegistryTest < Minitest::Test
  def test_models_registered_and_resolve
    assert MlxLm::Models::REGISTRY.key?("qwen3_vl"), "qwen3_vl should be registered"
    assert MlxLm::Models::REGISTRY.key?("smollm3"), "smollm3 should be registered"

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "qwen3_vl" })
    assert_equal MlxLm::Models::Qwen3VL::Model, model_class
    assert_equal MlxLm::Models::Qwen3VL::ModelArgs, args_class

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "smollm3" })
    assert_equal MlxLm::Models::SmolLM3::Model, model_class
    assert_equal MlxLm::Models::SmolLM3::ModelArgs, args_class
  end
end
