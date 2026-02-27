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
            x = x + mx.expand_dims(mx.take(bias, indices, 0), -2)
          end
          x
        end

        def to_quantized(group_size: nil, bits: nil, mode: "affine", quantize_input: false)
          raise ArgumentError, "Quantized input is not supported." if quantize_input

          QuantizedSwitchLinear.from_switch_linear(self, group_size, bits, mode: mode)
        end
      end

      # Quantized version of SwitchLinear using gather_qmm.
      class QuantizedSwitchLinear < MLX::NN::Module
        attr_reader :group_size, :bits, :mode

        def initialize(input_dims, output_dims, num_experts, bias: false, group_size: nil, bits: nil, mode: "affine")
          super()

          @group_size, @bits = MLX::NN.__send__(:defaults_for_mode, mode, group_size, bits)
          @mode = mode

          mx = MLX::Core
          scale = Math.sqrt(1.0 / input_dims)
          q_weight, q_scales, *q_biases = mx.quantize(
            mx.random_uniform(
              [num_experts, output_dims, input_dims],
              scale * -1.0,
              scale,
              mx.float32
            ),
            @group_size,
            @bits,
            @mode
          )
          self.weight = q_weight
          self.scales = q_scales
          self.biases = q_biases.empty? ? nil : q_biases[0]
          self.bias = mx.zeros([num_experts, output_dims]) if bias

          freeze
        end

        def call(x, indices, sorted_indices: false)
          mx = MLX::Core
          q_biases = respond_to?(:biases) ? biases : nil
          x = mx.gather_qmm(
            x,
            weight,
            scales,
            q_biases,
            nil,
            indices,
            true,
            @group_size,
            @bits,
            @mode,
            sorted_indices
          )
          if respond_to?(:bias)
            x = x + mx.expand_dims(mx.take(bias, indices, 0), -2)
          end
          x
        end

        def self.from_switch_linear(linear_layer, group_size = nil, bits = nil, mode: "affine")
          num_experts, output_dims, input_dims = linear_layer.weight.shape
          out = new(
            input_dims,
            output_dims,
            num_experts,
            bias: false,
            group_size: group_size,
            bits: bits,
            mode: mode
          )
          q_weight, q_scales, *q_biases = MLX::Core.quantize(
            linear_layer.weight,
            out.group_size,
            out.bits,
            out.mode
          )
          out.weight = q_weight
          out.scales = q_scales
          out.biases = q_biases.empty? ? nil : q_biases[0]
          out.bias = linear_layer.bias if linear_layer.state.key?("bias")
          out
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

          idx = mx.stop_gradient(idx) if training

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

      # Python name compatibility alias.
      class SwiGLU < SwitchGLU
      end

      # Batched expert MLP with configurable activation.
      class SwitchMLP < MLX::NN::Module
        def initialize(input_dims, hidden_dims, num_experts, activation: nil, bias: false)
          super()
          self.fc1 = SwitchLinear.new(input_dims, hidden_dims, num_experts, bias: bias)
          self.fc2 = SwitchLinear.new(hidden_dims, input_dims, num_experts, bias: bias)
          self.activation = activation || MLX::NN::GELU.new("precise")
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

          idx = mx.stop_gradient(idx) if training

          x = fc1.call(x, idx, sorted_indices: do_sort)
          x = activation.call(x)
          x = fc2.call(x, idx, sorted_indices: do_sort)

          if do_sort
            x = SwitchLayers.scatter_unsort(x, inv_order, indices.shape)
          end

          mx.squeeze(x, -2)
        end
      end
    end
  end
end
