$LOAD_PATH.unshift File.expand_path("../../lib", __dir__)
$LOAD_PATH.unshift File.expand_path("../../mlx-ruby/lib", __dir__)

require "mlx"
require "minitest/autorun"

require_relative "../../lib/mlx_lm/model_args"
require_relative "../../lib/mlx_lm/models"
require_relative "../../lib/mlx_lm/models/switch_layers"
require_relative "../../lib/mlx_lm/models/mistral3"
require_relative "../../lib/mlx_lm/models/solar_open"

class Mistral3SolarOpenModelsMistral3Test < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_mistral3_construct_forward_shape_and_sanitize_drops_multimodal_and_rotary
    args = MlxLm::Models::Mistral3::ModelArgs.from_dict({
      "model_type" => "mistral3",
      "text_config" => {
        "model_type" => "llama",
        "hidden_size" => 32,
        "num_hidden_layers" => 2,
        "num_attention_heads" => 4,
        "num_key_value_heads" => 4,
        "intermediate_size" => 64,
        "vocab_size" => 96,
        "rms_norm_eps" => 1e-5,
        "rope_theta" => 10_000.0,
        "max_position_embeddings" => 256,
        "tie_word_embeddings" => true,
      },
    })

    model = MlxLm::Models::Mistral3::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    tokens = @mx.array([[1, 2, 3, 4]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 4, 96], output.shape

    weights = {
      "vision_tower.patch_embed.weight" => @mx.zeros([1]).astype(@mx.float32),
      "multi_modal_projector.linear.weight" => @mx.zeros([1]).astype(@mx.float32),
      "language_model.model.embed_tokens.weight" => @mx.zeros([96, 32]).astype(@mx.float32),
      "language_model.model.layers.0.self_attn.rotary_emb.inv_freq" => @mx.zeros([8]).astype(@mx.float32),
      "language_model.lm_head.weight" => @mx.zeros([96, 32]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)

    refute sanitized.key?("vision_tower.patch_embed.weight")
    refute sanitized.key?("multi_modal_projector.linear.weight")
    refute sanitized.key?("language_model.model.layers.0.self_attn.rotary_emb.inv_freq")
    refute sanitized.key?("language_model.lm_head.weight")
    assert sanitized.key?("language_model.model.embed_tokens.weight")
  end
end

class Mistral3SolarOpenModelsTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_solar_open_construct_forward_shape_and_sanitize_stacks_experts
    args = MlxLm::Models::SolarOpen::ModelArgs.from_dict({
      "model_type" => "solar_open",
      "vocab_size" => 96,
      "hidden_size" => 32,
      "intermediate_size" => 64,
      "moe_intermediate_size" => 16,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 4,
      "head_dim" => 8,
      "n_shared_experts" => 1,
      "n_routed_experts" => 2,
      "routed_scaling_factor" => 1.0,
      "num_experts_per_tok" => 1,
      "first_k_dense_replace" => 1,
      "norm_topk_prob" => false,
      "max_position_embeddings" => 128,
      "rms_norm_eps" => 1e-5,
      "rope_theta" => 10_000.0,
      "tie_word_embeddings" => false,
      "partial_rotary_factor" => 1.0,
      "attention_bias" => false,
      "use_qk_norm" => false,
      "n_group" => 1,
      "topk_group" => 1,
      "scoring_func" => "sigmoid",
      "topk_method" => "noaux_tc",
    })

    model = MlxLm::Models::SolarOpen::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 3, 96], output.shape

    weights = {
      "model.embed_tokens.weight" => @mx.zeros([96, 32]).astype(@mx.float32),
      "model.layers.1.mlp.experts.0.gate_proj.weight" => @mx.array([[1.0, 2.0], [3.0, 4.0]], dtype: @mx.float32),
      "model.layers.1.mlp.experts.1.gate_proj.weight" => @mx.array([[5.0, 6.0], [7.0, 8.0]], dtype: @mx.float32),
      "model.layers.2.mlp.gate_proj.weight" => @mx.ones([2, 2]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)
    stacked = sanitized["model.layers.1.mlp.switch_mlp.gate_proj.weight"]
    @mx.eval(stacked)

    assert_equal [2, 2, 2], stacked.shape
    assert_equal [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], stacked.to_a
    refute sanitized.key?("model.layers.1.mlp.experts.0.gate_proj.weight")
    refute sanitized.key?("model.layers.1.mlp.experts.1.gate_proj.weight")
    refute sanitized.key?("model.layers.2.mlp.gate_proj.weight")
    assert sanitized.key?("model.embed_tokens.weight")
  end
end

class Mistral3SolarOpenModelsModelsRegisteredAndResolvableTest < Minitest::Test
  def test_models_registered_and_resolvable
    assert MlxLm::Models::REGISTRY.key?("mistral3"), "mistral3 should be registered"
    assert MlxLm::Models::REGISTRY.key?("solar_open"), "solar_open should be registered"

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "mistral3" })
    assert_equal MlxLm::Models::Mistral3::Model, model_class
    assert_equal MlxLm::Models::Mistral3::ModelArgs, args_class

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "solar_open" })
    assert_equal MlxLm::Models::SolarOpen::Model, model_class
    assert_equal MlxLm::Models::SolarOpen::ModelArgs, args_class
  end
end
