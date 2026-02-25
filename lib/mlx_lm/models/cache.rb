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
  end
end
