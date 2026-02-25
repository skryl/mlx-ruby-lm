module MlxLm
  module Models
    module MLA
      class MultiLinear < MLX::NN::Module
        def initialize(input_dims, output_dims, num_heads)
          super()
          scale = Math.sqrt(1.0 / input_dims)
          self.weight = MLX::Core.uniform([num_heads, output_dims, input_dims], -scale, scale)
        end

        def call(x, transpose: true)
          if transpose
            MLX::Core.matmul(x, MLX::Core.swapaxes(weight, -1, -2))
          else
            MLX::Core.matmul(x, weight)
          end
        end

        def to_quantized(group_size: nil, bits: nil, mode: "affine", quantize_input: false)
          raise ArgumentError, "Quantized input is not supported." if quantize_input

          QuantizedMultiLinear.from_multi_linear(self, group_size, bits, mode: mode)
        end
      end

      class QuantizedMultiLinear < MLX::NN::Module
        attr_reader :group_size, :bits, :mode

        def initialize(input_dims, output_dims, num_heads, group_size = nil, bits = nil, mode: "affine")
          super()

          @group_size, @bits = MLX::NN.__send__(:defaults_for_mode, mode, group_size, bits)
          @mode = mode

          scale = Math.sqrt(1.0 / input_dims)
          weight = MLX::Core.uniform([num_heads, output_dims, input_dims], -scale, scale)
          q_weight, q_scales, *q_biases = MLX::Core.quantize(weight, @group_size, @bits, @mode)
          self.weight = q_weight
          self.scales = q_scales
          self.biases = q_biases.empty? ? nil : q_biases[0]

          freeze
        end

        def call(x, transpose: true)
          MLX::Core.quantized_matmul(
            x,
            weight,
            scales,
            biases,
            transpose,
            @group_size,
            @bits,
            @mode
          )
        end

        def self.from_multi_linear(multi_linear_layer, group_size = nil, bits = nil, mode: "affine")
          num_heads, output_dims, input_dims = multi_linear_layer.weight.shape
          out = new(input_dims, output_dims, num_heads, group_size, bits, mode: mode)
          q_weight, q_scales, *q_biases = MLX::Core.quantize(
            multi_linear_layer.weight,
            out.group_size,
            out.bits,
            out.mode
          )
          out.weight = q_weight
          out.scales = q_scales
          out.biases = q_biases.empty? ? nil : q_biases[0]
          out
        end
      end
    end
  end
end
