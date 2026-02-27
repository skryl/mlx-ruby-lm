$LOAD_PATH.unshift File.expand_path("../../lib", __dir__)
$LOAD_PATH.unshift File.expand_path("../../mlx-ruby/lib", __dir__)

require "mlx"
require "minitest/autorun"

require_relative "../../lib/mlx_lm/model_args"
require_relative "../../lib/mlx_lm/models"
require_relative "../../lib/mlx_lm/models/activations"
require_relative "../../lib/mlx_lm/models/rope_utils"
require_relative "../../lib/mlx_lm/models/switch_layers"
require_relative "../../lib/mlx_lm/models/deepseek"
require_relative "../../lib/mlx_lm/models/deepseek_v32"
require_relative "../../lib/mlx_lm/models/glm_moe_dsa"

class DeepseekV32GlmMoeDsaModelsV32Test < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_deepseek_v32_construct_forward_shape_and_sanitize
    args = MlxLm::Models::DeepseekV32::ModelArgs.from_dict({
      "model_type" => "deepseek_v32",
      "hidden_size" => 32,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 4,
      "intermediate_size" => 64,
      "moe_intermediate_size" => 16,
      "vocab_size" => 97,
      "rms_norm_eps" => 1e-5,
      "max_position_embeddings" => 256,
      "rope_theta" => 10_000.0,
      "n_routed_experts" => 2,
      "n_shared_experts" => 1,
      "num_experts_per_tok" => 1,
      "moe_layer_freq" => 1,
      "first_k_dense_replace" => 0,
    })

    model = MlxLm::Models::DeepseekV32::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 3, 97], output.shape

    weights = {
      "model.layers.0.mlp.experts.0.gate_proj.weight" => @mx.array([[1.0, 2.0], [3.0, 4.0]], dtype: @mx.float32),
      "model.layers.0.mlp.experts.1.gate_proj.weight" => @mx.array([[5.0, 6.0], [7.0, 8.0]], dtype: @mx.float32),
      "model.layers.0.self_attn.rotary_emb.inv_freq" => @mx.zeros([8]).astype(@mx.float32),
      "model.layers.1.self_attn.q_proj.weight" => @mx.zeros([32, 32]).astype(@mx.float32),
      "model.layers.2.mlp.gate_proj.weight" => @mx.zeros([2, 2]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)
    stacked = sanitized["model.layers.0.mlp.switch_mlp.gate_proj.weight"]
    @mx.eval(stacked)

    refute sanitized.key?("model.layers.0.mlp.experts.0.gate_proj.weight")
    refute sanitized.key?("model.layers.0.mlp.experts.1.gate_proj.weight")
    refute sanitized.key?("model.layers.0.self_attn.rotary_emb.inv_freq")
    refute sanitized.key?("model.layers.2.mlp.gate_proj.weight")
    assert sanitized.key?("model.layers.1.self_attn.q_proj.weight")
    assert_equal [2, 2, 2], stacked.shape
    assert_equal [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], stacked.to_a
  end
end

class DeepseekV32GlmMoeDsaModelsTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_glm_moe_dsa_rope_parameters_mapping_construct_forward_shape_and_sanitize
    args = MlxLm::Models::GlmMoeDsa::ModelArgs.from_dict({
      "model_type" => "glm_moe_dsa",
      "hidden_size" => 32,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 4,
      "intermediate_size" => 64,
      "moe_intermediate_size" => 16,
      "vocab_size" => 83,
      "rms_norm_eps" => 1e-5,
      "max_position_embeddings" => 256,
      "rope_theta" => 10_000.0,
      "n_routed_experts" => 2,
      "n_shared_experts" => 1,
      "num_experts_per_tok" => 1,
      "moe_layer_freq" => 1,
      "first_k_dense_replace" => 0,
      "rope_parameters" => {
        "rope_theta" => 12_345.0,
        "type" => "yarn",
        "factor" => 8.0,
      },
    })

    assert_equal args.rope_parameters, args.rope_scaling
    assert_equal 12_345.0, args.rope_theta

    model = MlxLm::Models::GlmMoeDsa::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3, 4]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 4, 83], output.shape

    weights = {
      "model.layers.0.mlp.experts.0.up_proj.weight" => @mx.array([[1.0, 2.0], [3.0, 4.0]], dtype: @mx.float32),
      "model.layers.0.mlp.experts.1.up_proj.weight" => @mx.array([[5.0, 6.0], [7.0, 8.0]], dtype: @mx.float32),
      "model.layers.0.self_attn.rotary_emb.inv_freq" => @mx.zeros([8]).astype(@mx.float32),
      "model.layers.2.mlp.up_proj.weight" => @mx.zeros([2, 2]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)
    stacked = sanitized["model.layers.0.mlp.switch_mlp.up_proj.weight"]
    @mx.eval(stacked)

    refute sanitized.key?("model.layers.0.mlp.experts.0.up_proj.weight")
    refute sanitized.key?("model.layers.0.mlp.experts.1.up_proj.weight")
    refute sanitized.key?("model.layers.0.self_attn.rotary_emb.inv_freq")
    refute sanitized.key?("model.layers.2.mlp.up_proj.weight")
    assert_equal [2, 2, 2], stacked.shape
  end
end

class DeepseekV32GlmMoeDsaModelsModelsRegisteredAndResolveTest < Minitest::Test
  def test_models_registered_and_resolve
    assert MlxLm::Models::REGISTRY.key?("deepseek_v32"), "deepseek_v32 should be registered"
    assert MlxLm::Models::REGISTRY.key?("glm_moe_dsa"), "glm_moe_dsa should be registered"

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "deepseek_v32" })
    assert_equal MlxLm::Models::DeepseekV32::Model, model_class
    assert_equal MlxLm::Models::DeepseekV32::ModelArgs, args_class

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "glm_moe_dsa" })
    assert_equal MlxLm::Models::GlmMoeDsa::Model, model_class
    assert_equal MlxLm::Models::GlmMoeDsa::ModelArgs, args_class
  end
end
