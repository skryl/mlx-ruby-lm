require_relative "../test_helper"
require_relative "../../lib/mlx_lm/models/mla"

class Phase14MLATest < Minitest::Test
  include ParityTestHelpers

  def setup
    @mx = MLX::Core
  end

  def test_multilinear_forward_matches_matmul_when_transposed
    layer = MlxLm::Models::MLA::MultiLinear.new(16, 8, 3)
    x = @mx.random_uniform([2, 3, 4, 16], -1.0, 1.0, @mx.float32)

    expected = @mx.matmul(x, @mx.swapaxes(layer.weight, -1, -2))
    actual = layer.call(x)

    @mx.eval(expected, actual)
    assert_equal expected.shape, actual.shape
    assert_arrays_close(expected.tolist, actual.tolist, atol: 1e-6, msg: "transpose=true should match matmul")
  end

  def test_multilinear_forward_matches_matmul_without_transpose
    layer = MlxLm::Models::MLA::MultiLinear.new(16, 8, 3)
    x = @mx.random_uniform([2, 3, 4, 8], -1.0, 1.0, @mx.float32)

    expected = @mx.matmul(x, layer.weight)
    actual = layer.call(x, transpose: false)

    @mx.eval(expected, actual)
    assert_equal expected.shape, actual.shape
    assert_arrays_close(expected.tolist, actual.tolist, atol: 1e-6, msg: "transpose=false should match matmul")
  end

  def test_to_quantized_returns_quantized_multilinear_and_runs_forward
    layer = MlxLm::Models::MLA::MultiLinear.new(64, 32, 2)
    qlayer = layer.to_quantized(group_size: 32, bits: 4, mode: "affine")

    assert_instance_of MlxLm::Models::MLA::QuantizedMultiLinear, qlayer
    assert_equal 32, qlayer.group_size
    assert_equal 4, qlayer.bits
    assert_equal "affine", qlayer.mode

    x = @mx.random_uniform([1, 2, 5, 64], -1.0, 1.0, @mx.float32)
    y_fp = layer.call(x)
    y_q = qlayer.call(x)
    @mx.eval(y_fp, y_q)

    assert_equal y_fp.shape, y_q.shape
    assert y_q.dtype == y_fp.dtype

    fp_flat = y_fp.tolist.flatten
    q_flat = y_q.tolist.flatten
    mae = fp_flat.zip(q_flat).map { |a, b| (a - b).abs }.sum / fp_flat.length.to_f
    assert mae < 0.25, "quantized output drift too high: mae=#{mae}"
  end

  def test_to_quantized_rejects_quantized_input
    layer = MlxLm::Models::MLA::MultiLinear.new(16, 8, 2)
    assert_raises(ArgumentError) do
      layer.to_quantized(group_size: 32, bits: 4, quantize_input: true)
    end
  end
end
