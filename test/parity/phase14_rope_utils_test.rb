$LOAD_PATH.unshift File.expand_path("../../mlx-ruby/lib", __dir__)

require "mlx"
require "minitest/autorun"
require "json"
require_relative "../../lib/mlx_lm/models/rope_utils"

module Phase14ParityHelpers
  def python_eval(code)
    result = `python3 -c '#{code.gsub("'", "'\\\\''")}'`
    raise "Python eval failed: #{result}" unless $?.success?
    JSON.parse(result)
  end

  def assert_arrays_close(expected, actual, atol: 1e-5, msg: nil)
    expected = expected.flatten if expected.is_a?(Array) && expected.first.is_a?(Array)
    actual = actual.flatten if actual.is_a?(Array) && actual.first.is_a?(Array)
    assert_equal expected.length, actual.length, "#{msg} - length mismatch"
    expected.zip(actual).each_with_index do |(e, a), i|
      assert_in_delta e, a, atol, "#{msg} - element #{i}: expected #{e}, got #{a}"
    end
  end
end

class Phase14SuScaledRoPETest < Minitest::Test
  include Phase14ParityHelpers

  def setup
    @mx = MLX::Core
  end

  def test_su_scaled_rope_matches_python
    rope = MlxLm::Models::SuScaledRoPE.new(
      8,
      base: 10_000.0,
      max_position_embeddings: 131_072,
      original_max_position_embeddings: 4096,
      short_factor: [1.0, 1.0, 1.0, 1.0],
      long_factor: [1.0, 1.1, 1.2, 1.3]
    )

    x = @mx.arange(0, 16, 1, @mx.float32).reshape([1, 1, 2, 8])
    y = rope.call(x, offset: 2)

    @mx.eval(rope._freqs, y)

    py = python_eval(<<~PY)
      import json
      import sys
      import mlx.core as mx

      sys.path.insert(0, "mlx-lm")
      from mlx_lm.models.rope_utils import SuScaledRoPE

      rope = SuScaledRoPE(
          dims=8,
          base=10000.0,
          max_position_embeddings=131072,
          original_max_position_embeddings=4096,
          short_factor=[1.0, 1.0, 1.0, 1.0],
          long_factor=[1.0, 1.1, 1.2, 1.3],
      )
      x = mx.arange(0, 16, dtype=mx.float32).reshape(1, 1, 2, 8)
      y = rope(x, offset=2)

      mx.eval(rope._freqs, y)
      print(json.dumps({
          "scale": float(rope._scale),
          "freqs": rope._freqs.tolist(),
          "output": y.tolist(),
      }))
    PY

    assert_in_delta py["scale"], rope._scale, 1e-6
    assert_arrays_close py["freqs"], rope._freqs.to_a, atol: 1e-3, msg: "SuScaledRoPE freqs"
    assert_arrays_close py["output"], y.to_a, atol: 1e-5, msg: "SuScaledRoPE output"
  end
end

class Phase14Llama3RoPETest < Minitest::Test
  include Phase14ParityHelpers

  def setup
    @mx = MLX::Core
  end

  def test_llama3_rope_matches_python
    scaling_config = {
      "type" => "llama3",
      "factor" => 8.0,
      "low_freq_factor" => 1.0,
      "high_freq_factor" => 4.0,
      "original_max_position_embeddings" => 8192,
    }

    rope = MlxLm::Models::Llama3RoPE.new(
      dims: 8,
      max_position_embeddings: 2048,
      traditional: false,
      base: 10_000.0,
      scaling_config: scaling_config
    )

    x = @mx.arange(0, 16, 1, @mx.float32).reshape([1, 1, 2, 8])
    y = rope.call(x, offset: 1)

    @mx.eval(rope._freqs, y)

    py = python_eval(<<~PY)
      import json
      import sys
      import mlx.core as mx

      sys.path.insert(0, "mlx-lm")
      from mlx_lm.models.rope_utils import Llama3RoPE

      rope = Llama3RoPE(
          dims=8,
          max_position_embeddings=2048,
          traditional=False,
          base=10000.0,
          scaling_config={
              "type": "llama3",
              "factor": 8.0,
              "low_freq_factor": 1.0,
              "high_freq_factor": 4.0,
              "original_max_position_embeddings": 8192,
          },
      )
      x = mx.arange(0, 16, dtype=mx.float32).reshape(1, 1, 2, 8)
      y = rope(x, offset=1)

      mx.eval(rope._freqs, y)
      print(json.dumps({
          "freqs": rope._freqs.tolist(),
          "output": y.tolist(),
      }))
    PY

    assert_arrays_close py["freqs"], rope._freqs.to_a, atol: 1e-3, msg: "Llama3RoPE freqs"
    assert_arrays_close py["output"], y.to_a, atol: 1e-5, msg: "Llama3RoPE output"
  end
end

class Phase14YarnRoPETest < Minitest::Test
  include Phase14ParityHelpers

  def setup
    @mx = MLX::Core
  end

  def test_yarn_rope_matches_python
    rope = MlxLm::Models::YarnRoPE.new(
      8,
      traditional: false,
      max_position_embeddings: 2048,
      base: 10_000.0,
      scaling_factor: 4.0,
      original_max_position_embeddings: 4096,
      beta_fast: 32,
      beta_slow: 1,
      mscale: 1.5,
      mscale_all_dim: 0.5
    )

    x = @mx.arange(0, 24, 1, @mx.float32).reshape([1, 1, 2, 12])
    y = rope.call(x, offset: 3)

    @mx.eval(rope._freqs, y)

    py = python_eval(<<~PY)
      import json
      import sys
      import mlx.core as mx

      sys.path.insert(0, "mlx-lm")
      from mlx_lm.models.rope_utils import YarnRoPE

      rope = YarnRoPE(
          dims=8,
          traditional=False,
          max_position_embeddings=2048,
          base=10000.0,
          scaling_factor=4.0,
          original_max_position_embeddings=4096,
          beta_fast=32,
          beta_slow=1,
          mscale=1.5,
          mscale_all_dim=0.5,
      )
      x = mx.arange(0, 24, dtype=mx.float32).reshape(1, 1, 2, 12)
      y = rope(x, offset=3)

      mx.eval(rope._freqs, y)
      print(json.dumps({
          "mscale": float(rope.mscale),
          "freqs": rope._freqs.tolist(),
          "output": y.tolist(),
      }))
    PY

    assert_in_delta py["mscale"], rope.mscale, 1e-6
    assert_arrays_close py["freqs"], rope._freqs.to_a, atol: 1e-3, msg: "YarnRoPE freqs"
    assert_arrays_close py["output"], y.to_a, atol: 1e-5, msg: "YarnRoPE output"
  end
end

class Phase14RoPEFactoryTest < Minitest::Test
  def test_initialize_rope_default_and_linear
    default_rope = MlxLm::Models.initialize_rope(8, 10_000.0, false)
    assert_instance_of MLX::NN::RoPE, default_rope
    assert_in_delta 1.0, default_rope.scale, 1e-8

    linear_rope = MlxLm::Models.initialize_rope(
      8,
      10_000.0,
      false,
      { "type" => "linear", "factor" => 4.0 }
    )
    assert_instance_of MLX::NN::RoPE, linear_rope
    assert_in_delta 0.25, linear_rope.scale, 1e-8
  end

  def test_initialize_rope_variant_dispatch
    llama3 = MlxLm::Models.initialize_rope(
      8,
      10_000.0,
      false,
      {
        "type" => "llama3",
        "factor" => 8.0,
        "low_freq_factor" => 1.0,
        "high_freq_factor" => 4.0,
        "original_max_position_embeddings" => 8192,
      },
      max_position_embeddings: 2048
    )
    assert_instance_of MlxLm::Models::Llama3RoPE, llama3

    yarn = MlxLm::Models.initialize_rope(
      8,
      10_000.0,
      false,
      {
        "rope_type" => "deepseek_yarn",
        "factor" => 4.0,
      },
      max_position_embeddings: 2048
    )
    assert_instance_of MlxLm::Models::YarnRoPE, yarn

    longrope = MlxLm::Models.initialize_rope(
      8,
      10_000.0,
      false,
      {
        "type" => "longrope",
        "original_max_position_embeddings" => 4096,
        "short_factor" => [1.0, 1.0, 1.0, 1.0],
        "long_factor" => [1.0, 1.0, 1.0, 1.0],
      },
      max_position_embeddings: 131_072
    )
    assert_instance_of MlxLm::Models::SuScaledRoPE, longrope
  end

  def test_initialize_rope_mrope_and_error_paths
    mrope = MlxLm::Models.initialize_rope(
      8,
      10_000.0,
      false,
      {
        "rope_type" => "mrope",
        "mrope_section" => [16, 16, 16],
      }
    )
    assert_instance_of MLX::NN::RoPE, mrope

    err = assert_raises(ArgumentError) do
      MlxLm::Models.initialize_rope(
        8,
        10_000.0,
        false,
        {
          "rope_type" => "mrope",
          "mrope_section" => [16, 16],
        }
      )
    end
    assert_match(/MRoPE currently only supports 3 sections/, err.message)

    unsupported = assert_raises(ArgumentError) do
      MlxLm::Models.initialize_rope(
        8,
        10_000.0,
        false,
        { "type" => "unknown_rope" }
      )
    end
    assert_match(/Unsupported RoPE type unknown_rope/, unsupported.message)
  end
end
