$LOAD_PATH.unshift File.expand_path("../../lib", __dir__)
$LOAD_PATH.unshift File.expand_path("../../mlx-ruby/lib", __dir__)

require "mlx"
require "minitest/autorun"

require_relative "../../lib/mlx_lm/model_args"
require_relative "../../lib/mlx_lm/models"
require_relative "../../lib/mlx_lm/models/cache"
require_relative "../../lib/mlx_lm/models/rope_utils"
require_relative "../../lib/mlx_lm/models/gemma3_text"
require_relative "../../lib/mlx_lm/models/gemma3"

class Phase20DenseLaneQGemma3TextTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_gemma3_text_construct_forward_shape_and_sanitize_ties_embeddings
    args = MlxLm::Models::Gemma3Text::ModelArgs.from_dict({
      "model_type" => "gemma3_text",
      "hidden_size" => 32,
      "num_hidden_layers" => 4,
      "intermediate_size" => 64,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 2,
      "head_dim" => 8,
      "rms_norm_eps" => 1e-6,
      "vocab_size" => 83,
      "sliding_window" => 8,
      "sliding_window_pattern" => 3,
      "rope_theta" => 10_000.0,
      "rope_local_base_freq" => 10_000.0,
      "max_position_embeddings" => 128,
      "query_pre_attn_scalar" => 16.0,
    })

    model = MlxLm::Models::Gemma3Text::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    tokens = @mx.array([[1, 2, 3, 4]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 4, 83], output.shape

    cache = model.make_cache
    assert_equal 4, cache.length
    assert_instance_of MlxLm::RotatingKVCache, cache[0]
    assert_instance_of MlxLm::RotatingKVCache, cache[1]
    assert_instance_of MlxLm::KVCache, cache[2]
    assert_instance_of MlxLm::RotatingKVCache, cache[3]

    weights = {
      "model.embed_tokens.weight" => @mx.zeros([83, 32]).astype(@mx.float32),
      "model.layers.0.self_attn.rotary_emb.inv_freq" => @mx.zeros([8]).astype(@mx.float32),
      "model.norm.weight" => @mx.zeros([32]).astype(@mx.float32),
    }
    sanitized = model.sanitize(weights)

    refute sanitized.key?("model.layers.0.self_attn.rotary_emb.inv_freq")
    assert sanitized.key?("model.embed_tokens.weight")
    assert sanitized.key?("model.norm.weight")
    assert_equal true, model.instance_variable_get(:@tie_word_embeddings)
    assert_nil model.lm_head
  end
end

class Phase20DenseLaneQGemma3Test < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_gemma3_construct_forward_shape_and_sanitize_multimodal_prefixes
    args = MlxLm::Models::Gemma3::ModelArgs.from_dict({
      "model_type" => "gemma3",
      "vocab_size" => 97,
      "text_config" => {
        "hidden_size" => 32,
        "num_hidden_layers" => 2,
        "intermediate_size" => 64,
        "head_dim" => 4,
        "rms_norm_eps" => 1e-6,
        "sliding_window" => 8,
        "sliding_window_pattern" => 2,
        "query_pre_attn_scalar" => 16.0,
        "rope_theta" => 10_000.0,
        "max_position_embeddings" => 128,
      },
    })

    model = MlxLm::Models::Gemma3::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 3, 97], output.shape

    weights = {
      "vision_tower.patch_embed.weight" => @mx.zeros([1]).astype(@mx.float32),
      "multi_modal_projector.linear.weight" => @mx.zeros([1]).astype(@mx.float32),
      "language_model.model.embed_tokens.weight" => @mx.zeros([97, 32]).astype(@mx.float32),
      "language_model.model.layers.0.self_attn.rotary_emb.inv_freq" => @mx.zeros([8]).astype(@mx.float32),
      "language_model.model.norm.weight" => @mx.zeros([32]).astype(@mx.float32),
    }
    sanitized = model.sanitize(weights)

    refute sanitized.key?("vision_tower.patch_embed.weight")
    refute sanitized.key?("multi_modal_projector.linear.weight")
    refute sanitized.key?("language_model.model.layers.0.self_attn.rotary_emb.inv_freq")
    assert sanitized.key?("language_model.model.embed_tokens.weight")
    assert sanitized.key?("language_model.model.norm.weight")

    assert_equal true, model.language_model.instance_variable_get(:@tie_word_embeddings)
    assert_nil model.language_model.lm_head
  end
end

class Phase20DenseLaneQRegistryTest < Minitest::Test
  def test_models_registered_and_resolvable
    assert MlxLm::Models::REGISTRY.key?("gemma3_text"), "gemma3_text should be registered"
    assert MlxLm::Models::REGISTRY.key?("gemma3"), "gemma3 should be registered"

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "gemma3_text" })
    assert_equal MlxLm::Models::Gemma3Text::Model, model_class
    assert_equal MlxLm::Models::Gemma3Text::ModelArgs, args_class

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "gemma3" })
    assert_equal MlxLm::Models::Gemma3::Model, model_class
    assert_equal MlxLm::Models::Gemma3::ModelArgs, args_class
  end
end
