module MlxLm
  module ConvertUtils
    DTYPE_MAP = {
      "float32" => :float32,
      "float16" => :float16,
      "bfloat16" => :bfloat16,
      "int8" => :int8,
      "int32" => :int32,
    }.freeze

    module_function

    # Convert an MLX array to a different dtype.
    def convert_dtype(array, target_dtype)
      if target_dtype.is_a?(MLX::Core::Dtype)
        return array.astype(target_dtype)
      end
      dtype_sym = target_dtype.is_a?(Symbol) ? target_dtype : DTYPE_MAP[target_dtype.to_s]
      raise ArgumentError, "Unknown dtype: #{target_dtype}" unless dtype_sym
      array.astype(MLX::Core::Dtype.new(dtype_sym))
    end

    # Count total number of parameters in a model.
    def count_parameters(model)
      params = MLX::Utils.tree_flatten(model.parameters)
      total = 0
      params.each { |_, v| total += v.size }
      total
    end

    # Estimate total model size in bytes.
    def model_size_bytes(model)
      mx = MLX::Core
      params = MLX::Utils.tree_flatten(model.parameters)
      total = 0
      params.each do |_, v|
        bytes_per_elem = case v.dtype
                         when mx.float32 then 4
                         when mx.float16, mx.bfloat16 then 2
                         when mx.int32 then 4
                         when mx.int8, mx.uint8 then 1
                         when mx.int16, mx.uint16 then 2
                         when mx.int64 then 8
                         else 4
                         end
        total += v.size * bytes_per_elem
      end
      total
    end
  end
end
