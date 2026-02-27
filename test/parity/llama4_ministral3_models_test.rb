$LOAD_PATH.unshift File.expand_path("../../lib", __dir__)
$LOAD_PATH.unshift File.expand_path("../../mlx-ruby/lib", __dir__)

require "mlx"
require "minitest/autorun"

require_relative "../../lib/mlx_lm/model_args"
require_relative "../../lib/mlx_lm/models"
require_relative "../../lib/mlx_lm/models/cache"
require_relative "../../lib/mlx_lm/models/activations"
require_relative "../../lib/mlx_lm/models/rope_utils"
require_relative "../../lib/mlx_lm/models/switch_layers"
require_relative "../../lib/mlx_lm/models/pipeline"
require_relative "../../lib/mlx_lm/models/llama4"
require_relative "../../lib/mlx_lm/models/ministral3"

class Llama4Ministral3ModelsLlama4Test < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_llama4_construct_forward_shape_sanitize_and_make_cache
    args = MlxLm::Models::Llama4::ModelArgs.from_dict({
      "model_type" => "llama4",
      "text_config" => {
        "model_type" => "llama4_text",
        "hidden_size" => 32,
        "num_attention_heads" => 4,
        "num_key_value_heads" => 2,
        "num_hidden_layers" => 4,
        "vocab_size" => 97,
        "intermediate_size" => 48,
        "intermediate_size_mlp" => 64,
        "num_local_experts" => 2,
        "num_experts_per_tok" => 1,
        "interleave_moe_layer_step" => 2,
        "attention_chunk_size" => 4,
        "max_position_embeddings" => 128,
        "rope_theta" => 10_000.0,
        "head_dim" => 8,
        "rms_norm_eps" => 1e-5,
        "attention_bias" => false,
        "use_qk_norm" => true,
      },
    })

    model = MlxLm::Models::Llama4::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 3, 97], output.shape

    prefix = "language_model.model.layers.1.feed_forward.experts"
    weights = {
      "vision_model.patch_embed.weight" => @mx.zeros([1]).astype(@mx.float32),
      "multi_modal_projector.linear.weight" => @mx.zeros([1]).astype(@mx.float32),
      "#{prefix}.gate_up_proj" => @mx.array((0...48).to_a, dtype: @mx.float32).reshape([2, 3, 8]),
      "#{prefix}.down_proj" => @mx.array((0...36).to_a, dtype: @mx.float32).reshape([2, 6, 3]),
      "language_model.model.layers.0.self_attn.q_proj.weight" => @mx.zeros([32, 32]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)
    gate_proj = sanitized["#{prefix}.gate_proj.weight"]
    up_proj = sanitized["#{prefix}.up_proj.weight"]
    down_proj = sanitized["#{prefix}.down_proj.weight"]
    @mx.eval(gate_proj, up_proj, down_proj)

    refute sanitized.key?("vision_model.patch_embed.weight")
    refute sanitized.key?("multi_modal_projector.linear.weight")
    refute sanitized.key?("#{prefix}.gate_up_proj")
    refute sanitized.key?("#{prefix}.down_proj")
    assert_equal [2, 4, 3], gate_proj.shape
    assert_equal [2, 4, 3], up_proj.shape
    assert_equal [2, 3, 6], down_proj.shape
    assert sanitized.key?("language_model.model.layers.0.self_attn.q_proj.weight")

    cache = model.make_cache
    assert_equal 4, cache.length
    assert_instance_of MlxLm::ChunkedKVCache, cache[0]
    assert_instance_of MlxLm::ChunkedKVCache, cache[1]
    assert_instance_of MlxLm::ChunkedKVCache, cache[2]
    assert_instance_of MlxLm::KVCache, cache[3]
  end
end

class Llama4Ministral3ModelsMinistral3Test < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_ministral3_construct_forward_shape_sanitize_and_make_cache
    args = MlxLm::Models::Ministral3::ModelArgs.from_dict({
      "model_type" => "ministral3",
      "hidden_size" => 32,
      "num_hidden_layers" => 4,
      "intermediate_size" => 64,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 2,
      "head_dim" => 8,
      "max_position_embeddings" => 128,
      "rms_norm_eps" => 1e-5,
      "vocab_size" => 101,
      "tie_word_embeddings" => true,
      "sliding_window" => 8,
      "layer_types" => ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
      "rope_parameters" => {
        "rope_theta" => 10_000.0,
        "llama_4_scaling_beta" => 0.1,
        "original_max_position_embeddings" => 128,
      },
    })

    model = MlxLm::Models::Ministral3::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3, 4]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 4, 101], output.shape

    weights = {
      "model.layers.0.self_attn.rotary_emb.inv_freq" => @mx.zeros([8]).astype(@mx.float32),
      "lm_head.weight" => @mx.zeros([101, 32]).astype(@mx.float32),
      "model.embed_tokens.weight" => @mx.zeros([101, 32]).astype(@mx.float32),
      "model.layers.0.self_attn.q_proj.weight" => @mx.zeros([32, 32]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)
    refute sanitized.key?("model.layers.0.self_attn.rotary_emb.inv_freq")
    refute sanitized.key?("lm_head.weight")
    assert sanitized.key?("model.embed_tokens.weight")
    assert sanitized.key?("model.layers.0.self_attn.q_proj.weight")

    cache = model.make_cache
    assert_equal 4, cache.length
    assert_instance_of MlxLm::RotatingKVCache, cache[0]
    assert_instance_of MlxLm::KVCache, cache[1]
    assert_instance_of MlxLm::RotatingKVCache, cache[2]
    assert_instance_of MlxLm::KVCache, cache[3]
  end
end

class Llama4Ministral3ModelsTest < Minitest::Test
  def test_models_registered_and_resolve
    assert MlxLm::Models::REGISTRY.key?("llama4"), "llama4 should be registered"
    assert MlxLm::Models::REGISTRY.key?("ministral3"), "ministral3 should be registered"

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "llama4" })
    assert_equal MlxLm::Models::Llama4::Model, model_class
    assert_equal MlxLm::Models::Llama4::ModelArgs, args_class

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "ministral3" })
    assert_equal MlxLm::Models::Ministral3::Model, model_class
    assert_equal MlxLm::Models::Ministral3::ModelArgs, args_class
  end
end
