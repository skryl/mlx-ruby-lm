$LOAD_PATH.unshift File.expand_path("../../lib", __dir__)
$LOAD_PATH.unshift File.expand_path("../../mlx-ruby/lib", __dir__)

require "mlx"
require "minitest/autorun"

require_relative "../../lib/mlx_lm/model_args"
require_relative "../../lib/mlx_lm/models"
require_relative "../../lib/mlx_lm/models/cache"
require_relative "../../lib/mlx_lm/models/activations"
require_relative "../../lib/mlx_lm/models/cohere2"
require_relative "../../lib/mlx_lm/models/internlm3"

class Cohere2Internlm3ModelsCohere2Test < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_cohere2_construct_forward_shape_and_cache_pattern
    args = MlxLm::Models::Cohere2::ModelArgs.from_dict({
      "model_type" => "cohere2",
      "hidden_size" => 32,
      "head_dim" => 16,
      "num_hidden_layers" => 4,
      "intermediate_size" => 64,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "vocab_size" => 128,
      "sliding_window" => 8,
      "sliding_window_pattern" => 2,
      "attention_bias" => false,
      "layer_norm_bias" => false,
      "logit_scale" => 0.5,
    })

    model = MlxLm::Models::Cohere2::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    input = @mx.array([[1, 2, 3]], @mx.int32)
    output = model.call(input)
    @mx.eval(output)

    assert_equal [1, 3, 128], output.shape

    caches = model.make_cache
    assert_equal 4, caches.length
    assert_instance_of MlxLm::RotatingKVCache, caches[0]
    assert_instance_of MlxLm::KVCache, caches[1]
    assert_instance_of MlxLm::RotatingKVCache, caches[2]
    assert_instance_of MlxLm::KVCache, caches[3]
  end
end

class Cohere2Internlm3ModelsM3Test < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_internlm3_construct_forward_shape_and_rope_scaling_validation
    args = MlxLm::Models::InternLM3::ModelArgs.from_dict({
      "model_type" => "internlm3",
      "hidden_size" => 32,
      "num_hidden_layers" => 2,
      "intermediate_size" => 64,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "rms_norm_eps" => 1e-6,
      "vocab_size" => 96,
      "bias" => true,
      "qkv_bias" => true,
      "rope_scaling" => {
        "factor" => 2.0,
        "rope_type" => "linear",
      },
      "tie_word_embeddings" => false,
    })

    model = MlxLm::Models::InternLM3::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    input = @mx.array([[1, 2, 3, 4]], @mx.int32)
    output = model.call(input)
    @mx.eval(output)

    assert_equal [1, 4, 96], output.shape

    error = assert_raises(ArgumentError) do
      MlxLm::Models::InternLM3::ModelArgs.from_dict({
        "model_type" => "internlm3",
        "hidden_size" => 32,
        "num_hidden_layers" => 1,
        "intermediate_size" => 64,
        "num_attention_heads" => 2,
        "vocab_size" => 96,
        "rms_norm_eps" => 1e-6,
        "rope_scaling" => {
          "factor" => 2.0,
          "rope_type" => "unsupported",
        },
      })
    end
    assert_match("rope_type", error.message)
  end
end
