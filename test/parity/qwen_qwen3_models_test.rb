$LOAD_PATH.unshift File.expand_path("../../mlx-ruby/lib", __dir__)

require "mlx"
require "minitest/autorun"
require_relative "../../lib/mlx_lm/model_args"
require_relative "../../lib/mlx_lm/models"
require_relative "../../lib/mlx_lm/models/activations"
require_relative "../../lib/mlx_lm/models/rope_utils"
require_relative "../../lib/mlx_lm/models/qwen"
require_relative "../../lib/mlx_lm/models/qwen3"

class QwenQwen3ModelsTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_qwen_instantiation_and_forward_shape
    args = MlxLm::Models::Qwen::ModelArgs.from_dict({
      "model_type" => "qwen",
      "hidden_size" => 32,
      "num_attention_heads" => 2,
      "num_hidden_layers" => 2,
      "kv_channels" => 16,
      "intermediate_size" => 64,
      "vocab_size" => 100,
      "no_bias" => true,
    })
    model = MlxLm::Models::Qwen::Model.new(args)

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)

    assert_instance_of MlxLm::Models::Qwen::Model, model
    assert_equal [1, 3, 100], output.shape
  end

  def test_qwen_sanitize_removes_rotary_freqs_only
    args = MlxLm::Models::Qwen::ModelArgs.from_dict({
      "model_type" => "qwen",
      "hidden_size" => 32,
      "num_attention_heads" => 2,
      "num_hidden_layers" => 1,
      "kv_channels" => 16,
      "intermediate_size" => 64,
      "vocab_size" => 100,
    })
    model = MlxLm::Models::Qwen::Model.new(args)
    weights = {
      "transformer.wte.weight" => @mx.zeros([100, 32]).astype(@mx.float32),
      "transformer.h.0.attn.rotary_emb.inv_freq" => @mx.zeros([16]).astype(@mx.float32),
      "lm_head.weight" => @mx.zeros([100, 32]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)

    refute sanitized.key?("transformer.h.0.attn.rotary_emb.inv_freq")
    assert sanitized.key?("transformer.wte.weight")
    assert sanitized.key?("lm_head.weight")
  end
end

class QwenQwen3ModelsQwen3Test < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_qwen3_instantiation_and_forward_shape
    args = MlxLm::Models::Qwen3::ModelArgs.from_dict({
      "model_type" => "qwen3",
      "hidden_size" => 32,
      "num_hidden_layers" => 2,
      "intermediate_size" => 64,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "rms_norm_eps" => 1e-6,
      "vocab_size" => 100,
      "max_position_embeddings" => 256,
      "rope_theta" => 10_000.0,
      "head_dim" => 16,
      "tie_word_embeddings" => true,
    })
    model = MlxLm::Models::Qwen3::Model.new(args)

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)

    assert_instance_of MlxLm::Models::Qwen3::Model, model
    assert_equal [1, 3, 100], output.shape
  end

  def test_qwen3_sanitize_drops_lm_head_when_tied
    args = MlxLm::Models::Qwen3::ModelArgs.from_dict({
      "model_type" => "qwen3",
      "hidden_size" => 32,
      "num_hidden_layers" => 1,
      "intermediate_size" => 64,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "rms_norm_eps" => 1e-6,
      "vocab_size" => 100,
      "head_dim" => 16,
      "tie_word_embeddings" => true,
    })
    model = MlxLm::Models::Qwen3::Model.new(args)
    weights = {
      "model.embed_tokens.weight" => @mx.zeros([100, 32]).astype(@mx.float32),
      "lm_head.weight" => @mx.zeros([100, 32]).astype(@mx.float32),
      "model.layers.0.self_attn.rotary_emb.inv_freq" => @mx.zeros([16]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)

    refute sanitized.key?("lm_head.weight")
    assert sanitized.key?("model.embed_tokens.weight")
    assert sanitized.key?("model.layers.0.self_attn.rotary_emb.inv_freq")
  end
end
