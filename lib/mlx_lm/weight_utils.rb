require "safetensors"

module MlxLm
  module WeightUtils
    module_function

    DTYPE_UNPACK = {
      "F32" => ["e*", :float32],
      "F16" => ["e*", :float16],  # handled specially
      "BF16" => ["e*", :bfloat16], # handled specially
      "F64" => ["E*", :float32],
      "I8" => ["c*", :int8],
      "I16" => ["s<*", :int16],
      "I32" => ["l<*", :int32],
      "I64" => ["q<*", :int64],
      "U8" => ["C*", :uint8],
      "U16" => ["S<*", :uint16],
      "U32" => ["L<*", :uint32],
    }.freeze

    # Load a single safetensors file, returning { name => MLX::Core::Array }
    def load_safetensors(path)
      mx = MLX::Core
      raw = File.binread(path)
      tensors = Safetensors.deserialize(raw)
      result = {}
      tensors.each do |name, info|
        result[name] = _tensor_to_mlx(info, mx)
      end
      result
    end

    # Load all model*.safetensors shards from a directory
    def load_sharded_safetensors(directory)
      pattern = File.join(directory, "model*.safetensors")
      files = Dir.glob(pattern).sort

      if files.empty?
        raise "No safetensors found in #{directory}"
      end

      weights = {}
      files.each do |f|
        weights.merge!(load_safetensors(f))
      end
      weights
    end

    def _tensor_to_mlx(info, mx)
      shape = info["shape"]
      dtype_str = info["dtype"]
      data = info["data"]

      # For F32/float32, unpack as little-endian floats
      if dtype_str == "F32" || dtype_str == "float32"
        values = data.unpack("e*")
        mx.array(values).reshape(shape)
      elsif dtype_str == "F16"
        # 16-bit float: unpack as uint16, create array, view as float16
        values = data.unpack("S<*")
        mx.array(values, dtype: mx.uint16).view(mx.float16).reshape(shape)
      elsif dtype_str == "BF16"
        values = data.unpack("S<*")
        mx.array(values, dtype: mx.uint16).view(mx.bfloat16).reshape(shape)
      elsif dtype_str == "I32"
        values = data.unpack("l<*")
        mx.array(values, dtype: mx.int32).reshape(shape)
      elsif dtype_str == "I64"
        values = data.unpack("q<*")
        mx.array(values, dtype: mx.int64).reshape(shape)
      elsif dtype_str == "U8"
        values = data.unpack("C*")
        mx.array(values, dtype: mx.uint8).reshape(shape)
      else
        # Fallback: try F32
        values = data.unpack("e*")
        mx.array(values).reshape(shape)
      end
    end

    # Convert a flat weight dict like:
    #   { "model.layers.0.weight" => tensor, ... }
    # into a nested hash/array structure like:
    #   { "model" => { "layers" => [{ "weight" => tensor }] } }
    #
    # Numeric path segments become array indices.
    def tree_unflatten(flat)
      root = {}

      flat.each do |dotted_key, value|
        parts = dotted_key.split(".")
        node = root

        parts.each_with_index do |part, i|
          is_last = (i == parts.length - 1)
          next_part = parts[i + 1] unless is_last

          if is_last
            _set_in_node(node, part, value)
          else
            node = _ensure_child(node, part, next_part)
          end
        end
      end

      _finalize_arrays(root)
    end

    # Set a value in the current node (hash or array)
    def _set_in_node(node, key, value)
      if node.is_a?(Hash)
        if key.match?(/\A\d+\z/)
          idx = key.to_i
          node[idx] = value
        else
          node[key] = value
        end
      end
    end

    # Ensure a child container exists for the given key
    def _ensure_child(node, key, next_key)
      numeric_key = key.match?(/\A\d+\z/)
      next_is_numeric = next_key && next_key.match?(/\A\d+\z/)

      if numeric_key
        idx = key.to_i
        existing = node[idx]
        if existing.nil?
          child = next_is_numeric ? {} : {}
          node[idx] = child
          child
        else
          existing
        end
      else
        existing = node[key]
        if existing.nil?
          child = {}
          node[key] = child
          child
        else
          existing
        end
      end
    end

    # Recursively convert hashes with all-integer keys into arrays
    def _finalize_arrays(node)
      return node unless node.is_a?(Hash)

      # First, recurse into children
      node.each do |k, v|
        node[k] = _finalize_arrays(v)
      end

      # Check if all keys are integers (indicating this should be an array)
      if node.keys.all? { |k| k.is_a?(Integer) }
        max_idx = node.keys.max
        arr = Array.new(max_idx + 1)
        node.each { |k, v| arr[k] = v }
        arr
      else
        node
      end
    end

    private_class_method :_set_in_node, :_ensure_child, :_finalize_arrays, :_tensor_to_mlx
  end
end
