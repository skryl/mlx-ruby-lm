module MlxLm
  # Simple KV Cache — concatenates new K,V to existing.
  # Uses simple concatenation since MLX Ruby doesn't support in-place slice assignment.
  class KVCache
    attr_reader :offset

    def initialize
      @keys = nil
      @values = nil
      @offset = 0
    end

    def update_and_fetch(keys, values)
      mx = MLX::Core
      if @keys.nil?
        @keys = keys
        @values = values
      else
        @keys = mx.concatenate([@keys, keys], 2)
        @values = mx.concatenate([@values, values], 2)
      end
      @offset += keys.shape[2]
      return @keys, @values
    end

    def size
      @offset
    end

    def state
      [@keys, @values]
    end

    def state=(v)
      @keys, @values = v
      @offset = @keys ? @keys.shape[2] : 0
    end
  end

  # Rotating KV Cache — fixed maximum size, old entries rotate out.
  class RotatingKVCache
    attr_reader :offset

    def initialize(max_size:, keep: 0)
      @max_size = max_size
      @keep = keep
      @keys = nil
      @values = nil
      @offset = 0
    end

    def size
      [@offset, @max_size].min
    end

    def update_and_fetch(keys, values)
      mx = MLX::Core
      if @keys.nil?
        @keys = keys
        @values = values
        @offset += keys.shape[2]
      else
        @keys = mx.concatenate([@keys, keys], 2)
        @values = mx.concatenate([@values, values], 2)
        @offset += keys.shape[2]

        # Trim if exceeding max_size
        if @keys.shape[2] > @max_size
          excess = @keys.shape[2] - @max_size
          if @keep > 0
            # Keep first @keep tokens + last (max_size - @keep) tokens
            kept_k = mx.split(@keys, [@keep], 2)[0]
            tail_k = mx.split(@keys, [excess + @keep], 2)[1]
            @keys = mx.concatenate([kept_k, tail_k], 2)

            kept_v = mx.split(@values, [@keep], 2)[0]
            tail_v = mx.split(@values, [excess + @keep], 2)[1]
            @values = mx.concatenate([kept_v, tail_v], 2)
          else
            @keys = mx.split(@keys, [excess], 2)[1]
            @values = mx.split(@values, [excess], 2)[1]
          end
        end
      end

      return @keys, @values
    end
  end

  module Cache
    module_function

    def make_prompt_cache(model, max_kv_size: nil)
      if model.respond_to?(:make_cache)
        return model.make_cache
      end

      num_layers = model.layers.length
      if max_kv_size
        Array.new(num_layers) { RotatingKVCache.new(max_size: max_kv_size, keep: 4) }
      else
        Array.new(num_layers) { KVCache.new }
      end
    end

    def save_prompt_cache(path, cache)
      mx = MLX::Core
      tensors = {}

      cache.each_with_index do |layer_cache, i|
        keys, values = layer_cache.state
        next unless keys

        mx.eval(keys, values)
        tensors["layer.#{i}.keys"] = keys
        tensors["layer.#{i}.values"] = values
      end

      # Also save metadata
      tensors["_meta_offsets"] = mx.array(cache.map(&:offset), mx.int32)

      # Serialize using safetensors gem
      st_tensors = {}
      tensors.each do |name, arr|
        arr = arr.astype(mx.float32) unless [mx.float32, mx.int32].include?(arr.dtype)
        mx.eval(arr)
        data = arr.tolist
        data = data.flatten if data.is_a?(::Array) && data.first.is_a?(::Array)
        data = [data].flatten

        if arr.dtype == mx.int32
          binary = data.map(&:to_i).pack("l<*")
          st_tensors[name] = { "dtype" => "int32", "shape" => arr.shape, "data" => binary }
        else
          binary = data.pack("e*")
          st_tensors[name] = { "dtype" => "float32", "shape" => arr.shape, "data" => binary }
        end
      end

      File.binwrite(path, Safetensors.serialize(st_tensors))
    end

    def load_prompt_cache(path, model)
      loaded = WeightUtils.load_safetensors(path)
      mx = MLX::Core

      offsets = loaded["_meta_offsets"]
      mx.eval(offsets)
      offset_list = offsets.tolist
      offset_list = [offset_list].flatten

      num_layers = model.layers.length
      cache = Array.new(num_layers) { KVCache.new }

      num_layers.times do |i|
        keys = loaded["layer.#{i}.keys"]
        values = loaded["layer.#{i}.values"]
        next unless keys

        cache[i].state = [keys, values]
      end

      cache
    end
  end
end
