require_relative "../test_helper"
require_relative "../../lib/mlx_lm/models/activations"
require_relative "../../lib/mlx_lm/models/gated_delta"

class Phase15ActivationsParityTest < Minitest::Test
  include ParityTestHelpers

  def setup
    @mx = MLX::Core
  end

  def test_swiglu_xielu_and_xielu_module_match_python
    x = @mx.array([[-2.0, -0.5, 0.0, 1.5]], dtype: @mx.float32)
    gate = @mx.array([[0.25, -1.0, 0.5, 2.0]], dtype: @mx.float32)
    alpha_p = @mx.array(0.2, dtype: @mx.float32)
    alpha_n = @mx.array(-0.1, dtype: @mx.float32)
    beta = @mx.array(0.5, dtype: @mx.float32)
    eps = @mx.array(-1e-6, dtype: @mx.float32)

    swiglu = MlxLm::Models::Activations.swiglu(gate, x)
    xielu = MlxLm::Models::Activations.xielu(x, alpha_p, alpha_n, beta, eps)
    layer = MlxLm::Models::Activations::XieLU.new
    layer_out = layer.call(x)

    @mx.eval(swiglu, xielu, layer_out, layer.alpha_p, layer.alpha_n, layer.beta, layer.eps)

    py = python_eval(<<~PY)
      import json
      import sys
      import mlx.core as mx

      sys.path.insert(0, "mlx-lm")
      from mlx_lm.models.activations import swiglu, xielu, XieLU

      x = mx.array([[-2.0, -0.5, 0.0, 1.5]], dtype=mx.float32)
      gate = mx.array([[0.25, -1.0, 0.5, 2.0]], dtype=mx.float32)
      alpha_p = mx.array(0.2, dtype=mx.float32)
      alpha_n = mx.array(-0.1, dtype=mx.float32)
      beta = mx.array(0.5, dtype=mx.float32)
      eps = mx.array(-1e-6, dtype=mx.float32)

      y_swiglu = swiglu(gate, x)
      y_xielu = xielu(x, alpha_p, alpha_n, beta, eps)

      layer = XieLU()
      y_layer = layer(x)

      mx.eval(y_swiglu, y_xielu, y_layer, layer.alpha_p, layer.alpha_n, layer.beta, layer.eps)
      print(json.dumps({
          "swiglu": y_swiglu.tolist(),
          "xielu": y_xielu.tolist(),
          "xielu_layer": y_layer.tolist(),
          "alpha_p": float(layer.alpha_p),
          "alpha_n": float(layer.alpha_n),
          "beta": float(layer.beta),
          "eps": float(layer.eps),
      }))
    PY

    assert_arrays_close py["swiglu"], swiglu.to_a, atol: 1e-6, msg: "swiglu parity"
    assert_arrays_close py["xielu"], xielu.to_a, atol: 1e-6, msg: "xielu parity"
    assert_arrays_close py["xielu_layer"], layer_out.to_a, atol: 1e-6, msg: "XieLU layer parity"
    assert_in_delta py["alpha_p"], layer.alpha_p.to_a, 1e-6
    assert_in_delta py["alpha_n"], layer.alpha_n.to_a, 1e-6
    assert_in_delta py["beta"], layer.beta.to_a, 1e-6
    assert_in_delta py["eps"], layer.eps.to_a, 1e-6
  end
end

class Phase15GatedDeltaParityTest < Minitest::Test
  include ParityTestHelpers

  def setup
    @mx = MLX::Core
  end

  def test_gated_delta_ops_scalar_gating_masked_matches_python
    q, k, v, state, mask = base_tensors

    g = @mx.sigmoid((@mx.arange(0, 12, 1, @mx.float32).reshape([1, 3, 4]) - 6.0) / 4.0)
    beta = @mx.sigmoid((@mx.arange(0, 12, 1, @mx.float32).reshape([1, 3, 4]) - 4.0) / 5.0)

    y_rb, st_rb = MlxLm::Models::GatedDelta.gated_delta_ops(q, k, v, g, beta, state, mask)
    @mx.eval(y_rb, st_rb)

    py = python_eval(<<~PY)
      import json
      import sys
      import mlx.core as mx

      sys.path.insert(0, "mlx-lm")
      from mlx_lm.models.gated_delta import gated_delta_ops

      q = (mx.arange(0, 18, dtype=mx.float32).reshape(1, 3, 2, 3) - 9.0) / 8.0
      k = (mx.arange(100, 118, dtype=mx.float32).reshape(1, 3, 2, 3) - 109.0) / 9.0
      v = (mx.arange(0, 24, dtype=mx.float32).reshape(1, 3, 4, 2) - 12.0) / 7.0
      state = (mx.arange(0, 24, dtype=mx.float32).reshape(1, 4, 2, 3) - 10.0) / 6.0
      mask = mx.array([[True, False, True]])

      g = mx.sigmoid((mx.arange(0, 12, dtype=mx.float32).reshape(1, 3, 4) - 6.0) / 4.0)
      beta = mx.sigmoid((mx.arange(0, 12, dtype=mx.float32).reshape(1, 3, 4) - 4.0) / 5.0)

      y, st = gated_delta_ops(q, k, v, g, beta, state, mask)
      mx.eval(y, st)
      print(json.dumps({"y": y.tolist(), "state": st.tolist()}))
    PY

    assert_arrays_close py["y"], y_rb.to_a, atol: 1e-5, msg: "gated_delta_ops output parity"
    assert_arrays_close py["state"], st_rb.to_a, atol: 1e-5, msg: "gated_delta_ops state parity"
  end

  def test_gated_delta_update_vectorized_gating_matches_python_and_ops_reference
    q, k, v, state, mask = base_tensors
    a = (@mx.arange(0, 36, 1, @mx.float32).reshape([1, 3, 4, 3]) - 18.0) / 10.0
    b = (@mx.arange(0, 12, 1, @mx.float32).reshape([1, 3, 4]) - 5.0) / 6.0
    a_log = @mx.log(@mx.array([[1.1], [1.4], [1.8], [2.2]], dtype: @mx.float32))
    dt_bias = @mx.array(
      [
        [0.05, -0.02, 0.01],
        [0.10, 0.00, -0.10],
        [-0.03, 0.07, 0.02],
        [0.00, -0.04, 0.08],
      ],
      dtype: @mx.float32
    )

    beta = @mx.sigmoid(b)
    g = MlxLm::Models::GatedDelta.compute_g(a_log, a, dt_bias)
    y_ref, st_ref = MlxLm::Models::GatedDelta.gated_delta_ops(q, k, v, g, beta, state, mask)

    y_update, st_update = MlxLm::Models::GatedDelta.gated_delta_update(
      q, k, v, a, b, a_log, dt_bias, state, mask, use_kernel: false
    )
    y_kernel, st_kernel = MlxLm::Models::GatedDelta.gated_delta_update(
      q, k, v, a, b, a_log, dt_bias, state, mask, use_kernel: true
    )

    @mx.eval(y_ref, st_ref, y_update, st_update, y_kernel, st_kernel)

    assert_arrays_close y_ref.to_a, y_update.to_a, atol: 1e-5, msg: "update should match ops reference (output)"
    assert_arrays_close st_ref.to_a, st_update.to_a, atol: 1e-5, msg: "update should match ops reference (state)"
    assert_arrays_close y_update.to_a, y_kernel.to_a, atol: 1e-6, msg: "kernel path should match fallback output"
    assert_arrays_close st_update.to_a, st_kernel.to_a, atol: 1e-6, msg: "kernel path should match fallback state"

    py = python_eval(<<~PY)
      import json
      import sys
      import mlx.core as mx

      sys.path.insert(0, "mlx-lm")
      from mlx_lm.models.gated_delta import gated_delta_update

      q = (mx.arange(0, 18, dtype=mx.float32).reshape(1, 3, 2, 3) - 9.0) / 8.0
      k = (mx.arange(100, 118, dtype=mx.float32).reshape(1, 3, 2, 3) - 109.0) / 9.0
      v = (mx.arange(0, 24, dtype=mx.float32).reshape(1, 3, 4, 2) - 12.0) / 7.0
      state = (mx.arange(0, 24, dtype=mx.float32).reshape(1, 4, 2, 3) - 10.0) / 6.0
      mask = mx.array([[True, False, True]])

      a = (mx.arange(0, 36, dtype=mx.float32).reshape(1, 3, 4, 3) - 18.0) / 10.0
      b = (mx.arange(0, 12, dtype=mx.float32).reshape(1, 3, 4) - 5.0) / 6.0
      A_log = mx.log(mx.array([[1.1], [1.4], [1.8], [2.2]], dtype=mx.float32))
      dt_bias = mx.array(
          [
              [0.05, -0.02, 0.01],
              [0.10, 0.00, -0.10],
              [-0.03, 0.07, 0.02],
              [0.00, -0.04, 0.08],
          ],
          dtype=mx.float32,
      )

      y, st = gated_delta_update(
          q, k, v, a, b, A_log, dt_bias, state=state, mask=mask, use_kernel=False
      )
      mx.eval(y, st)
      print(json.dumps({"y": y.tolist(), "state": st.tolist()}))
    PY

    assert_arrays_close py["y"], y_update.to_a, atol: 1e-5, msg: "gated_delta_update output parity"
    assert_arrays_close py["state"], st_update.to_a, atol: 1e-5, msg: "gated_delta_update state parity"
  end

  private

  def base_tensors
    q = (@mx.arange(0, 18, 1, @mx.float32).reshape([1, 3, 2, 3]) - 9.0) / 8.0
    k = (@mx.arange(100, 118, 1, @mx.float32).reshape([1, 3, 2, 3]) - 109.0) / 9.0
    v = (@mx.arange(0, 24, 1, @mx.float32).reshape([1, 3, 4, 2]) - 12.0) / 7.0
    state = (@mx.arange(0, 24, 1, @mx.float32).reshape([1, 4, 2, 3]) - 10.0) / 6.0
    mask = @mx.array([[true, false, true]])

    [q, k, v, state, mask]
  end
end
