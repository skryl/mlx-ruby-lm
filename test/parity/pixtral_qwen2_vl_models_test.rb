$LOAD_PATH.unshift File.expand_path("../../lib", __dir__)
$LOAD_PATH.unshift File.expand_path("../../mlx-ruby/lib", __dir__)

require "mlx"
require "minitest/autorun"

require_relative "../../lib/mlx_lm/model_args"
require_relative "../../lib/mlx_lm/models"
require_relative "../../lib/mlx_lm/models/llama"
require_relative "../../lib/mlx_lm/models/qwen2"
require_relative "../../lib/mlx_lm/models/pixtral"
require_relative "../../lib/mlx_lm/models/qwen2_vl"

class PixtralQwen2VlModelsTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_pixtral_construct_forward_shape_and_sanitize
    args = MlxLm::Models::Pixtral::ModelArgs.from_dict({
      "model_type" => "pixtral",
      "text_config" => {
        "hidden_size" => 64,
        "num_hidden_layers" => 2,
        "intermediate_size" => 128,
        "vocab_size" => 97,
        "rms_norm_eps" => 1e-5,
        "max_position_embeddings" => 128,
        "rope_theta" => 10_000.0,
      },
    })

    model = MlxLm::Models::Pixtral::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)

    assert_equal [1, 3, 97], output.shape
    assert_equal false, args.text_config["tie_word_embeddings"]
    assert_equal 32, args.text_config["num_attention_heads"]

    weights = {
      "language_model.model.embed_tokens.weight" => @mx.zeros([97, 64]).astype(@mx.float32),
      "vision_tower.encoder.weight" => @mx.zeros([2, 2]).astype(@mx.float32),
      "multi_modal_projector.proj.weight" => @mx.zeros([2, 2]).astype(@mx.float32),
    }
    sanitized = model.sanitize(weights)

    refute sanitized.key?("vision_tower.encoder.weight")
    refute sanitized.key?("multi_modal_projector.proj.weight")
    assert sanitized.key?("language_model.model.embed_tokens.weight")
  end
end

class PixtralQwen2VlModelsQwen2VLTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_qwen2_vl_construct_forward_shape_and_sanitize_prefixes_language_model
    args = MlxLm::Models::Qwen2VL::ModelArgs.from_dict({
      "model_type" => "qwen2_vl",
      "hidden_size" => 32,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 4,
      "intermediate_size" => 64,
      "vocab_size" => 89,
      "rms_norm_eps" => 1e-5,
      "max_position_embeddings" => 128,
      "rope_theta" => 10_000.0,
      "tie_word_embeddings" => true,
    })

    model = MlxLm::Models::Qwen2VL::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    tokens = @mx.array([[1, 2, 3, 4]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)

    assert_equal [1, 4, 89], output.shape
    assert_equal 32, args.text_config["hidden_size"]

    weights = {
      "model.embed_tokens.weight" => @mx.zeros([89, 32]).astype(@mx.float32),
      "lm_head.weight" => @mx.zeros([89, 32]).astype(@mx.float32),
      "language_model.model.layers.0.self_attn.q_proj.weight" => @mx.zeros([32, 32]).astype(@mx.float32),
      "visual.encoder.weight" => @mx.zeros([2, 2]).astype(@mx.float32),
      "vision_tower.blocks.0.weight" => @mx.zeros([2, 2]).astype(@mx.float32),
    }
    sanitized = model.sanitize(weights)

    refute sanitized.key?("visual.encoder.weight")
    refute sanitized.key?("vision_tower.blocks.0.weight")
    assert sanitized.key?("language_model.model.embed_tokens.weight")
    assert sanitized.key?("language_model.lm_head.weight")
    assert sanitized.key?("language_model.model.layers.0.self_attn.q_proj.weight")
    assert sanitized.keys.all? { |key| key.start_with?("language_model.") }
  end
end

class PixtralQwen2VlModelsModelsRegisteredAndResolvedTest < Minitest::Test
  def test_models_registered_and_resolved
    assert MlxLm::Models::REGISTRY.key?("pixtral"), "pixtral should be registered"
    assert MlxLm::Models::REGISTRY.key?("qwen2_vl"), "qwen2_vl should be registered"

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "pixtral" })
    assert_equal MlxLm::Models::Pixtral::Model, model_class
    assert_equal MlxLm::Models::Pixtral::ModelArgs, args_class

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "qwen2_vl" })
    assert_equal MlxLm::Models::Qwen2VL::Model, model_class
    assert_equal MlxLm::Models::Qwen2VL::ModelArgs, args_class
  end
end
