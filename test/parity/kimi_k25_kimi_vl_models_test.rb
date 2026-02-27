$LOAD_PATH.unshift File.expand_path("../../lib", __dir__)
$LOAD_PATH.unshift File.expand_path("../../mlx-ruby/lib", __dir__)

require "mlx"
require "minitest/autorun"

require_relative "../../lib/mlx_lm/model_args"
require_relative "../../lib/mlx_lm/models"
require_relative "../../lib/mlx_lm/models/activations"
require_relative "../../lib/mlx_lm/models/switch_layers"
require_relative "../../lib/mlx_lm/models/deepseek"
require_relative "../../lib/mlx_lm/models/kimi_k25"
require_relative "../../lib/mlx_lm/models/kimi_vl"

class KimiK25KimiVlModelsK25Test < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_kimi_k25_construct_forward_shape_and_sanitize_drops_multimodal_towers
    args = MlxLm::Models::KimiK25::ModelArgs.from_dict({
      "model_type" => "kimi_k25",
      "text_config" => {
        "model_type" => "deepseek",
        "vocab_size" => 83,
        "hidden_size" => 32,
        "intermediate_size" => 64,
        "num_hidden_layers" => 2,
        "num_attention_heads" => 4,
        "num_key_value_heads" => 4,
        "max_position_embeddings" => 256,
        "rms_norm_eps" => 1e-5,
        "rope_theta" => 10_000.0,
        "attention_bias" => false,
      },
    })

    model = MlxLm::Models::KimiK25::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3, 4]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 4, 83], output.shape

    weights = {
      "vision_tower.encoder.weight" => @mx.zeros([1]).astype(@mx.float32),
      "vision_model.encoder.weight" => @mx.zeros([1]).astype(@mx.float32),
      "multi_modal_projector.linear.weight" => @mx.zeros([1]).astype(@mx.float32),
      "mm_projector.linear.weight" => @mx.zeros([1]).astype(@mx.float32),
      "language_model.model.layers.0.self_attn.rotary_emb.inv_freq" => @mx.zeros([8]).astype(@mx.float32),
      "language_model.model.embed_tokens.weight" => @mx.zeros([83, 32]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)

    refute sanitized.key?("vision_tower.encoder.weight")
    refute sanitized.key?("vision_model.encoder.weight")
    refute sanitized.key?("multi_modal_projector.linear.weight")
    refute sanitized.key?("mm_projector.linear.weight")
    refute sanitized.key?("language_model.model.layers.0.self_attn.rotary_emb.inv_freq")
    assert sanitized.key?("language_model.model.embed_tokens.weight")
  end
end

class KimiK25KimiVlModelsTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_kimi_vl_construct_forward_shape_and_sanitize_drops_multimodal_towers
    args = MlxLm::Models::KimiVL::ModelArgs.from_dict({
      "model_type" => "kimi_vl",
      "text_config" => {
        "model_type" => "deepseek",
        "vocab_size" => 89,
        "hidden_size" => 32,
        "intermediate_size" => 64,
        "num_hidden_layers" => 2,
        "num_attention_heads" => 4,
        "num_key_value_heads" => 4,
        "max_position_embeddings" => 256,
        "rms_norm_eps" => 1e-5,
        "rope_theta" => 10_000.0,
        "attention_bias" => false,
      },
    })

    model = MlxLm::Models::KimiVL::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 3, 89], output.shape

    weights = {
      "vision_tower.patch_embed.weight" => @mx.zeros([1]).astype(@mx.float32),
      "multi_modal_projector.linear.weight" => @mx.zeros([1]).astype(@mx.float32),
      "language_model.model.layers.0.self_attn.rotary_emb.inv_freq" => @mx.zeros([8]).astype(@mx.float32),
      "language_model.model.embed_tokens.weight" => @mx.zeros([89, 32]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)

    refute sanitized.key?("vision_tower.patch_embed.weight")
    refute sanitized.key?("multi_modal_projector.linear.weight")
    refute sanitized.key?("language_model.model.layers.0.self_attn.rotary_emb.inv_freq")
    assert sanitized.key?("language_model.model.embed_tokens.weight")
  end
end

class KimiK25KimiVlModelsModelsRegisteredAndResolveTest < Minitest::Test
  def test_models_registered_and_resolve
    assert MlxLm::Models::REGISTRY.key?("kimi_k25"), "kimi_k25 should be registered"
    assert MlxLm::Models::REGISTRY.key?("kimi_vl"), "kimi_vl should be registered"

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "kimi_k25" })
    assert_equal MlxLm::Models::KimiK25::Model, model_class
    assert_equal MlxLm::Models::KimiK25::ModelArgs, args_class

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "kimi_vl" })
    assert_equal MlxLm::Models::KimiVL::Model, model_class
    assert_equal MlxLm::Models::KimiVL::ModelArgs, args_class
  end
end
