module MlxLm
  module Models
    module GatedDelta
      module_function

      def compute_g(a_log, a, dt_bias)
        mx = MLX::Core
        decay = mx.exp(a_log.astype(mx.float32)) * MLX::NN.softplus(a + dt_bias)
        mx.exp(
          mx.multiply(-1.0, decay)
        ).astype(a.dtype)
      end

      def gated_delta_kernel(q, k, v, g, beta, state, mask = nil)
        # TODO: Add a Metal custom-kernel specialization for prefill throughput parity.
        gated_delta_ops(q, k, v, g, beta, state, mask)
      end

      def gated_delta_ops(q, k, v, g, beta, state = nil, mask = nil)
        mx = MLX::Core
        bsz, steps, hk, dk = q.shape
        v_shape = v.shape
        hv = v_shape[-2]
        dv = v_shape[-1]

        state ||= mx.zeros([bsz, hv, dv, dk], q.dtype)

        repeat_factor = hv / hk
        if repeat_factor > 1
          q = mx.repeat(q, repeat_factor, -2)
          k = mx.repeat(k, repeat_factor, -2)
        end

        q_steps = mx.split(q, steps, 1).map { |x| mx.squeeze(x, 1) }
        k_steps = mx.split(k, steps, 1).map { |x| mx.squeeze(x, 1) }
        v_steps = mx.split(v, steps, 1).map { |x| mx.squeeze(x, 1) }
        g_steps = mx.split(g, steps, 1).map { |x| mx.squeeze(x, 1) }
        beta_steps = mx.split(beta, steps, 1).map { |x| mx.squeeze(x, 1) }
        mask_steps =
          if mask.nil?
            nil
          elsif mask.ndim == 1
            [mask]
          else
            mx.split(mask, steps, 1).map { |x| mx.squeeze(x, 1) }
          end

        ys = []
        steps.times do |t|
          y, state = _gated_delta_step_ops(
            q_steps[t],
            k_steps[t],
            v_steps[t],
            g_steps[t],
            beta_steps[t],
            state,
            mask_steps&.[](t)
          )
          ys << y
        end

        [mx.stack(ys, 1), state]
      end

      def gated_delta_update(
        q,
        k,
        v,
        a,
        b,
        a_log,
        dt_bias,
        state = nil,
        mask = nil,
        use_kernel: true
      )
        mx = MLX::Core
        beta = mx.sigmoid(b)
        g = compute_g(a_log, a, dt_bias)

        if state.nil?
          bsz, = q.shape
          dk = q.shape[-1]
          hv = v.shape[-2]
          dv = v.shape[-1]
          state = mx.zeros([bsz, hv, dv, dk], q.dtype)
        end

        if use_kernel && metal_kernel_available?
          gated_delta_kernel(q, k, v, g, beta, state, mask)
        else
          gated_delta_ops(q, k, v, g, beta, state, mask)
        end
      end

      def _gated_delta_step_ops(q, k, v, g, beta, state, mask = nil)
        mx = MLX::Core
        old_state = state

        decay = case g.ndim
        when 2
          mx.expand_dims(g, [2, 3])
        when 3
          mx.expand_dims(g, 2)
        else
          raise ArgumentError, "Unsupported gating shape #{g.shape.inspect}"
        end

        state = state * decay
        k_expanded = mx.expand_dims(k, 2)
        kv_mem = (state * k_expanded).sum(-1)
        delta = (v - kv_mem) * mx.expand_dims(beta, -1)
        state = state + k_expanded * mx.expand_dims(delta, -1)
        y = (state * mx.expand_dims(q, 2)).sum(-1)

        unless mask.nil?
          mask_shape = [mask.shape[0]] + [1] * (state.ndim - 1)
          state = mx.where(mask.reshape(mask_shape), state, old_state)
        end

        [y, state]
      end
      private_class_method :_gated_delta_step_ops

      def metal_kernel_available?
        mx = MLX::Core
        return false unless mx.respond_to?(:metal_is_available) && mx.metal_is_available
        return false unless mx.respond_to?(:default_device)

        device = mx.default_device
        device.respond_to?(:type) && device.type == :gpu
      end
      private_class_method :metal_kernel_available?
    end
  end
end
