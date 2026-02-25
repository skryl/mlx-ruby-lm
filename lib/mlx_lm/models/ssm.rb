module MlxLm
  module Models
    module SSM
      module_function

      def compute_dt(dt, dt_bias, time_step_limit = [0.001, 100.0])
        dt = MLX::NN.softplus(dt + dt_bias)
        MLX::Core.clip(dt, time_step_limit[0], time_step_limit[1])
      end

      def segsum(x, mask: nil)
        mx = MLX::Core
        l = x.shape[-1]

        unless mask.nil?
          mask_e = mx.expand_dims(mask, 1)
          x = x * mask_e
        end

        x = mx.repeat(mx.expand_dims(x, -1), l, -1)
        x = mx.tril(x, -1)
        x_segsum = mx.cumsum(x, -2)

        unless mask.nil?
          mask_e = mx.expand_dims(mask, 1)
          valid = mx.multiply(mx.expand_dims(mask_e, -1), mx.expand_dims(mask_e, -2))
          x_segsum = mx.where(valid, x_segsum, -Float::INFINITY)
        end

        x_segsum
      end

      # Baseline implementation for SSD-SSM using explicit recurrence.
      def ssm_attn(
        x,
        a_log,
        b,
        c,
        d,
        dt,
        dt_bias,
        state: nil,
        time_step_limit: [0.001, 100.0],
        mask: nil,
        lengths: nil,
        step: 256
      )
        _ = step
        raise NotImplementedError, "length-aware SSM path is not implemented yet" unless lengths.nil?

        mx = MLX::Core
        batch_size, seq_len, num_heads, head_dim = x.shape
        _, _, num_groups, state_dim = b.shape

        repeats = num_heads / num_groups
        dt = compute_dt(dt, dt_bias, time_step_limit)
        dt = mx.expand_dims(dt, 0) if dt.ndim == 2
        a = mx.multiply(-1.0, mx.exp(a_log).astype(dt.dtype))

        state ||= mx.zeros([batch_size, num_heads, head_dim, state_dim], x.dtype)

        ys = []
        seq_len.times do |t|
          x_t = _slice_step(x, t)
          dt_t = _slice_step(dt, t)
          b_t = _slice_step(b, t)
          c_t = _slice_step(c, t)

          if repeats > 1
            b_t = mx.repeat(b_t, repeats, 1)
            c_t = mx.repeat(c_t, repeats, 1)
          end

          decay = mx.exp(dt_t * a.reshape([1, num_heads]))
          prev_state = state
          state = state * decay.reshape([batch_size, num_heads, 1, 1])

          dB = dt_t.reshape([batch_size, num_heads, 1, 1]) * b_t.reshape([batch_size, num_heads, 1, state_dim])
          state = state + x_t.reshape([batch_size, num_heads, head_dim, 1]) * dB

          y_t = (state * c_t.reshape([batch_size, num_heads, 1, state_dim])).sum(-1)
          y_t = y_t + x_t * d.reshape([1, num_heads, 1])

          unless mask.nil?
            m_t = _slice_step(mask, t)
            m_t = m_t.reshape([batch_size, 1, 1])
            state = mx.where(m_t, state, prev_state)
            y_t = mx.where(m_t, y_t, mx.zeros(y_t.shape, y_t.dtype))
          end

          ys << y_t
        end

        [mx.stack(ys, 1), state]
      end

      def ssm_update_kernel(*_args, **_kwargs)
        raise NotImplementedError,
              "SSM metal kernel path is not implemented in mlx-ruby-lm yet"
      end

      def ssm_update(
        hidden_states,
        a_log,
        b,
        c,
        d,
        dt,
        dt_bias,
        state: nil,
        time_step_limit: [0.001, 100.0],
        mask: nil,
        lengths: nil
      )
        mx = MLX::Core
        seq_len = hidden_states.shape[1]

        use_attn_path = seq_len > 1 ||
          state.nil? ||
          !mx.respond_to?(:metal_is_available) ||
          !mx.metal_is_available ||
          !mx.respond_to?(:default_device) ||
          (mx.default_device.respond_to?(:type) && mx.default_device.type != :gpu)

        if use_attn_path
          return ssm_attn(
            hidden_states,
            a_log,
            b,
            c,
            d,
            dt,
            dt_bias,
            state: state,
            time_step_limit: time_step_limit,
            mask: mask,
            lengths: lengths
          )
        end

        ssm_update_kernel(
          hidden_states,
          a_log,
          b,
          c,
          d,
          dt,
          dt_bias,
          state,
          time_step_limit
        )
      end

      def _slice_step(array, idx)
        mx = MLX::Core
        tail = idx.zero? ? array : mx.split(array, [idx], 1)[1]
        mx.squeeze(mx.split(tail, [1], 1)[0], 1)
      end
      private_class_method :_slice_step
    end
  end
end
