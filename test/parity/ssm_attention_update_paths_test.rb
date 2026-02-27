# frozen_string_literal: true

require_relative "../test_helper"
require_relative "../../lib/mlx_lm/models/ssm"

class SsmAttentionUpdatePathsTest < Minitest::Test
  def test_compute_dt_applies_softplus_and_clip
    mx = MLX::Core

    dt = mx.array([[-10.0, 0.0, 10.0]], dtype: mx.float32)
    dt_bias = mx.array([[0.0, 0.0, 0.0]], dtype: mx.float32)

    out = MlxLm::Models::SSM.compute_dt(dt, dt_bias, [0.1, 1.0])
    actual = out.tolist.flatten

    expected = [-10.0, 0.0, 10.0].map do |v|
      softplus = Math.log1p(Math.exp(v))
      [[softplus, 0.1].max, 1.0].min
    end

    expected.zip(actual).each do |e, a|
      assert_in_delta e, a, 1e-5
    end
  end

  def test_segsum_masked_positions_become_negative_infinity
    mx = MLX::Core

    x = mx.array([[[1.0, 2.0, 3.0]]], dtype: mx.float32)
    mask = mx.array([[1, 1, 0]], dtype: mx.float32)

    out = MlxLm::Models::SSM.segsum(x, mask: mask)
    values = out.tolist

    assert_equal 1, values.length
    assert_equal 1, values[0].length

    # Any relation touching masked index 2 should be -inf in the masked segsum output.
    row = values[0][0]
    assert row[2][0].infinite?
    assert row[2][1].infinite?
    assert row[2][2].infinite?
  end

  def test_ssm_attn_shapes
    mx = MLX::Core

    batch = 1
    seq = 3
    heads = 2
    head_dim = 2
    groups = 1
    state_dim = 3

    x = mx.random_uniform([batch, seq, heads, head_dim], -0.5, 0.5, mx.float32)
    a_log = mx.random_uniform([heads], -0.2, 0.2, mx.float32)
    b = mx.random_uniform([batch, seq, groups, state_dim], -0.5, 0.5, mx.float32)
    c = mx.random_uniform([batch, seq, groups, state_dim], -0.5, 0.5, mx.float32)
    d = mx.random_uniform([heads], -0.2, 0.2, mx.float32)
    dt = mx.random_uniform([batch, seq, heads], 0.01, 0.2, mx.float32)
    dt_bias = mx.zeros([heads], mx.float32)

    y, state = MlxLm::Models::SSM.ssm_attn(
      x, a_log, b, c, d, dt, dt_bias
    )

    assert_equal [batch, seq, heads, head_dim], y.shape
    assert_equal [batch, heads, head_dim, state_dim], state.shape
  end

  def test_ssm_update_dispatches_to_attn_path
    mx = MLX::Core

    x = mx.random_uniform([1, 2, 2, 2], -0.5, 0.5, mx.float32)
    a_log = mx.random_uniform([2], -0.2, 0.2, mx.float32)
    b = mx.random_uniform([1, 2, 1, 3], -0.5, 0.5, mx.float32)
    c = mx.random_uniform([1, 2, 1, 3], -0.5, 0.5, mx.float32)
    d = mx.random_uniform([2], -0.2, 0.2, mx.float32)
    dt = mx.random_uniform([1, 2, 2], 0.01, 0.2, mx.float32)
    dt_bias = mx.zeros([2], mx.float32)

    y, state = MlxLm::Models::SSM.ssm_update(x, a_log, b, c, d, dt, dt_bias)

    assert_equal [1, 2, 2, 2], y.shape
    assert_equal [1, 2, 2, 3], state.shape
  end

  def test_ssm_update_kernel_explicitly_not_implemented
    assert_raises(NotImplementedError) do
      MlxLm::Models::SSM.ssm_update_kernel(nil)
    end
  end
end
