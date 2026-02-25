module MlxLm
  module Models
    module SwitchLayers
      # Gather-sort helper: reorder tokens so same-expert tokens are contiguous.
      # Returns [sorted_x, sorted_indices, inv_order].
      def self.gather_sort(x, indices)
        mx = MLX::Core
        m = indices.shape[-1]
        flat_indices = mx.flatten(indices)
        order = mx.argsort(flat_indices)
        inv_order = mx.argsort(order)
        token_ids = mx.floor_divide(order, m)
        sorted_x = mx.take(mx.flatten(x, 0, -3), token_ids, 0)
        sorted_indices = mx.take(flat_indices, order)
        [sorted_x, sorted_indices, inv_order]
      end

      # Scatter-unsort helper: restore original token order after sorted computation.
      def self.scatter_unsort(x, inv_order, shape = nil)
        mx = MLX::Core
        x = mx.take(x, inv_order, 0)
        x = mx.unflatten(x, 0, shape) if shape
        x
      end

      # SwitchLinear: batched expert linear layer using gather_mm.
      # Stacks all expert weights into a single [num_experts, output_dims, input_dims] tensor
      # and dispatches via mx.gather_mm.
      class SwitchLinear < MLX::NN::Module
        def initialize(input_dims, output_dims, num_experts, bias: false)
          super()
          mx = MLX::Core
          scale = Math.sqrt(1.0 / input_dims)
          self.weight = mx.random_uniform(
            [num_experts, output_dims, input_dims],
            scale * -1.0, scale, mx.float32
          )
          self.bias = mx.zeros([num_experts, output_dims]) if bias
        end

        def call(x, indices, sorted_indices: false)
          mx = MLX::Core
          x = mx.gather_mm(
            x,
            mx.swapaxes(weight, -1, -2),
            nil,
            indices,
            sorted_indices
          )
          if respond_to?(:bias)
            x = x + mx.expand_dims(mx.take(bias, indices), -2)
          end
          x
        end
      end

      # SwitchGLU: batched expert MLP with SwiGLU activation using SwitchLinear.
      # Replaces per-token expert routing loops with gather_mm for ONNX traceability.
      class SwitchGLU < MLX::NN::Module
        def initialize(input_dims, hidden_dims, num_experts, bias: false)
          super()
          self.gate_proj = SwitchLinear.new(input_dims, hidden_dims, num_experts, bias: bias)
          self.up_proj = SwitchLinear.new(input_dims, hidden_dims, num_experts, bias: bias)
          self.down_proj = SwitchLinear.new(hidden_dims, input_dims, num_experts, bias: bias)
        end

        def call(x, indices)
          mx = MLX::Core
          x = mx.expand_dims(x, [-2, -3])

          # Sort optimization for many tokens
          do_sort = indices.size >= 64
          idx = indices
          inv_order = nil

          if do_sort
            x, idx, inv_order = SwitchLayers.gather_sort(x, indices)
          end

          idx = mx.stop_gradient(idx)

          x_up = up_proj.call(x, idx, sorted_indices: do_sort)
          x_gate = gate_proj.call(x, idx, sorted_indices: do_sort)

          # SwiGLU activation: silu(gate) * up
          x = down_proj.call(
            MLX::NN.silu(x_gate) * x_up,
            idx,
            sorted_indices: do_sort
          )

          if do_sort
            x = SwitchLayers.scatter_unsort(x, inv_order, indices.shape)
          end

          mx.squeeze(x, -2)
        end
      end
    end
  end
end
