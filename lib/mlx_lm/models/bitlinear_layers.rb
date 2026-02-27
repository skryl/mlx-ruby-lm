module MlxLm
  module Models
    class BitLinear < MLX::NN::Module
      attr_reader :in_features, :out_features, :invert_weight_scales

      def initialize(
        in_features,
        out_features,
        bias: true,
        invert_weight_scales: false
      )
        super()
        mx = MLX::Core

        @in_features = in_features
        @out_features = out_features
        @invert_weight_scales = invert_weight_scales

        packed_out_features = (out_features + 3) / 4
        self.weight = mx.zeros([packed_out_features, in_features], mx.uint8)
        self.weight_scale = mx.array([1.0], dtype: mx.float32)
        self.bias = mx.zeros([out_features], mx.float32) if bias
      end

      def call(x)
        y = execute_matmul_kernel(x, weight)
        state.key?("bias") ? MLX::Core.add(y, bias) : y
      end

      def execute_matmul_kernel(x, packed_weights)
        # TODO(phase1e): switch to a custom Metal kernel once MLX Ruby exposes
        # a stable fast-kernel API equivalent to Python's mx.fast.metal_kernel.
        execute_matmul_fallback(x, packed_weights)
      end

      private

      def execute_matmul_fallback(x, packed_weights)
        input_dims = x.shape[-1]
        unless input_dims == @in_features
          raise ArgumentError, "Expected input features #{@in_features}, got #{input_dims}"
        end

        ternary_weight = unpack_packed_weights(packed_weights, x.dtype)
        out = MLX::Core.matmul(x, ternary_weight.T)

        scale = weight_scale.astype(x.dtype)
        scale = MLX::Core.divide(1.0, scale) if invert_weight_scales
        MLX::Core.multiply(out, scale)
      end

      def unpack_packed_weights(packed_weights, dtype)
        mx = MLX::Core

        w0 = (mx.bitwise_and(packed_weights, 0x03).astype(dtype) - 1.0)
        w1 = (mx.bitwise_and(mx.right_shift(packed_weights, 2), 0x03).astype(dtype) - 1.0)
        w2 = (mx.bitwise_and(mx.right_shift(packed_weights, 4), 0x03).astype(dtype) - 1.0)
        w3 = (mx.bitwise_and(mx.right_shift(packed_weights, 6), 0x03).astype(dtype) - 1.0)

        expanded = mx.concatenate([w0, w1, w2, w3], 0)
        return expanded if expanded.shape[0] == @out_features

        keep = mx.arange(0, @out_features, 1, mx.int32)
        mx.take(expanded, keep, 0)
      end
    end

    module_function

    def bitnet_quantize(model, quantization_config = {})
      modules_to_not_convert = Array(bitlinear_config_value(quantization_config, "modules_to_not_convert", []))
                               .map(&:to_s)
      invert_weight_scales = bitlinear_config_value(quantization_config, "linear_class", "").to_s != "autobitlinear"

      replacements = []
      leaves = model.leaf_modules
      flat = MLX::Utils.tree_flatten(leaves, is_leaf: lambda { |node| node.is_a?(MLX::NN::Module) })

      flat.each do |path, layer|
        path_s = path.to_s
        next if modules_to_not_convert.include?(path_s)
        next unless layer.is_a?(MLX::NN::Linear)

        out_features, in_features = layer.weight.shape
        replacements << [
          path_s,
          BitLinear.new(
            in_features,
            out_features,
            bias: layer.state.key?("bias"),
            invert_weight_scales: invert_weight_scales
          ),
        ]
      end

      model.update_modules(MLX::Utils.tree_unflatten(replacements)) unless replacements.empty?
      model
    end

    def bitlinear_config_value(config, key, default = nil)
      return default if config.nil?
      return config[key] if config.key?(key)

      config.fetch(key.to_sym, default)
    end
    private_class_method :bitlinear_config_value
  end
end
