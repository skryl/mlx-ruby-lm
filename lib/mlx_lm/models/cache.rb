module MlxLm
  # Ruby constant names cannot begin with "_", so this is the _BaseCache abstraction.
  class BaseCache
    def state
      []
    end

    def state=(value)
      return if value.nil? || (value.respond_to?(:empty?) && value.empty?)

      raise ArgumentError, "This cache has no state but a state was set."
    end

    def meta_state
      ""
    end

    def meta_state=(value)
      return if value.nil? || (value.respond_to?(:empty?) && value.empty?)

      raise ArgumentError, "This cache has no meta_state but a meta_state was set."
    end

    def is_trimmable
      false
    end

    def size
      0
    end

    def nbytes
      raise NotImplementedError, "Cache sub-class must implement nbytes"
    end

    def empty
      raise NotImplementedError, "Cache sub-class must implement empty"
    end

    def self.from_state(state, meta_state)
      obj = allocate
      obj.state = state
      obj.meta_state = meta_state
      obj
    end
  end

  # Simple KV Cache — concatenates new K,V to existing.
  # Uses simple concatenation since MLX Ruby doesn't support in-place slice assignment.
  class KVCache < BaseCache
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

    def is_trimmable
      true
    end

    def trim(n)
      return 0 if @keys.nil? || n <= 0

      n = [@offset, n].min
      @offset -= n
      @keys = _slice_prefix(@keys, @offset)
      @values = _slice_prefix(@values, @offset)
      n
    end

    def to_quantized(group_size: 64, bits: 4)
      quant_cache = QuantizedKVCache.new(group_size: group_size, bits: bits)
      return quant_cache if @keys.nil?

      mx = MLX::Core
      qk = mx.quantize(@keys, group_size, bits)
      qv = mx.quantize(@values, group_size, bits)
      quant_cache.state = [qk, qv]
      quant_cache
    end

    def empty
      @keys.nil?
    end

    def nbytes
      return 0 if @keys.nil?

      @keys.nbytes + @values.nbytes
    end

    def self.merge(caches)
      non_empty = caches.reject(&:empty)
      return new if non_empty.empty?

      mx = MLX::Core
      template_k, template_v = non_empty.first.state
      target_len = non_empty.map(&:size).max

      rows_k = caches.map do |cache|
        if cache.empty
          shape = template_k.shape.dup
          shape[0] = 1
          shape[2] = target_len
          mx.zeros(shape, template_k.dtype)
        else
          keys, _values = cache.state
          _left_pad_seq(keys, target_len)
        end
      end

      rows_v = caches.map do |cache|
        if cache.empty
          shape = template_v.shape.dup
          shape[0] = 1
          shape[2] = target_len
          mx.zeros(shape, template_v.dtype)
        else
          _keys, values = cache.state
          _left_pad_seq(values, target_len)
        end
      end

      out = new
      out.state = [mx.concatenate(rows_k, 0), mx.concatenate(rows_v, 0)]
      out
    end

    private

    def _slice_prefix(array, length)
      return array if array.shape[2] == length

      MLX::Core.split(array, [length], 2)[0]
    end

    def self._left_pad_seq(array, target_len)
      return array if array.shape[2] == target_len

      mx = MLX::Core
      pad = target_len - array.shape[2]
      pad_shape = array.shape.dup
      pad_shape[2] = pad
      padding = mx.zeros(pad_shape, array.dtype)
      mx.concatenate([padding, array], 2)
    end
  end

  # Rotating KV Cache — fixed maximum size, old entries rotate out.
  class RotatingKVCache < BaseCache
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

    def state
      [@keys, @values]
    end

    def state=(v)
      @keys, @values = v
      @offset = @keys ? @keys.shape[2] : 0
    end

    def meta_state
      [@keep, @max_size, @offset]
    end

    def meta_state=(v)
      @keep, @max_size, @offset = v.map(&:to_i)
    end

    def is_trimmable
      @offset < @max_size
    end

    def trim(n)
      return 0 if @keys.nil? || n <= 0

      n = [@offset, n].min
      @offset -= n
      keep_len = [@keys.shape[2], @offset].min
      @keys = _slice_prefix(@keys, keep_len)
      @values = _slice_prefix(@values, keep_len)
      n
    end

    def empty
      @keys.nil?
    end

    def nbytes
      return 0 if @keys.nil?

      @keys.nbytes + @values.nbytes
    end

    def self.merge(caches)
      KVCache.merge(caches)
    end

    private

    def _slice_prefix(array, length)
      return array if array.shape[2] == length

      MLX::Core.split(array, [length], 2)[0]
    end
  end

  class QuantizedKVCache < BaseCache
    attr_reader :offset, :group_size, :bits

    def initialize(group_size: 64, bits: 8)
      @keys = nil
      @values = nil
      @offset = 0
      @group_size = group_size
      @bits = bits
    end

    def update_and_fetch(keys, values)
      mx = MLX::Core
      qk = mx.quantize(keys, @group_size, @bits)
      qv = mx.quantize(values, @group_size, @bits)

      if @keys.nil?
        @keys = qk
        @values = qv
      else
        @keys = _concat_quantized(@keys, qk)
        @values = _concat_quantized(@values, qv)
      end

      @offset += keys.shape[2]
      [@keys, @values]
    end

    def size
      @offset
    end

    def state
      [@keys, @values]
    end

    def state=(v)
      @keys, @values = v
      @offset = @keys ? @keys[0].shape[2] : 0
    end

    def meta_state
      [@offset, @group_size, @bits]
    end

    def meta_state=(v)
      @offset, @group_size, @bits = v.map(&:to_i)
    end

    def is_trimmable
      true
    end

    def trim(n)
      return 0 if @keys.nil? || n <= 0

      n = [@offset, n].min
      @offset -= n
      @keys = _slice_quantized(@keys, @offset)
      @values = _slice_quantized(@values, @offset)
      n
    end

    def empty
      @keys.nil?
    end

    def nbytes
      return 0 if @keys.nil?

      _sum_nbytes(@keys) + _sum_nbytes(@values)
    end

    private

    def _concat_quantized(lhs, rhs)
      lhs.each_with_index.map do |item, i|
        MLX::Core.concatenate([item, rhs[i]], 2)
      end
    end

    def _slice_quantized(tensors, length)
      tensors.map do |item|
        item.shape[2] == length ? item : MLX::Core.split(item, [length], 2)[0]
      end
    end

    def _sum_nbytes(tensors)
      tensors.reduce(0) { |acc, t| acc + t.nbytes }
    end
  end

  class ArraysCache < BaseCache
    attr_reader :cache
    attr_accessor :left_padding, :lengths

    def initialize(size, left_padding: nil)
      @cache = Array.new(size)
      @left_padding = left_padding ? MLX::Core.array(left_padding) : nil
      @lengths = nil
    end

    def []=(idx, value)
      @cache[idx] = value
    end

    def [](idx)
      @cache[idx]
    end

    def state
      @cache
    end

    def state=(v)
      @cache = v
    end

    def meta_state
      [@left_padding, @lengths]
    end

    def meta_state=(v)
      @left_padding, @lengths = v
    end

    def filter(batch_indices)
      idx = _indices_array(batch_indices)
      @cache = @cache.map { |c| c.nil? ? nil : MLX::Core.take(c, idx, 0) }
    end

    def extend(other)
      @cache = @cache.zip(other.cache).map do |c, o|
        if c.nil?
          o
        elsif o.nil?
          c
        else
          MLX::Core.concatenate([c, o], 0)
        end
      end

      if @left_padding && other.left_padding
        @left_padding = MLX::Core.concatenate([@left_padding, other.left_padding], 0)
      end
      if @lengths && other.lengths
        @lengths = MLX::Core.concatenate([@lengths, other.lengths], 0)
      end
    end

    def extract(idx)
      single = _indices_array([idx])
      out = ArraysCache.new(@cache.length)
      out.state = @cache.map { |c| c.nil? ? nil : MLX::Core.take(c, single, 0) }
      if @left_padding
        out.left_padding = MLX::Core.take(@left_padding, single, 0)
      end
      if @lengths
        out.lengths = MLX::Core.take(@lengths, single, 0)
      end
      out
    end

    def prepare(lengths: nil, **_kwargs)
      @lengths = lengths.nil? ? nil : MLX::Core.array(lengths)
    end

    def finalize
      @lengths = nil
      @left_padding = nil
    end

    def advance(n)
      @lengths = MLX::Core.subtract(@lengths, n) if @lengths
      @left_padding = MLX::Core.subtract(@left_padding, n) if @left_padding
    end

    def make_mask(n)
      mx = MLX::Core
      pos = mx.arange(n).reshape([1, n])
      if @left_padding
        mx.greater_equal(pos, @left_padding.reshape([@left_padding.shape[0], 1]))
      elsif @lengths
        mx.less(pos, @lengths.reshape([@lengths.shape[0], 1]))
      else
        nil
      end
    end

    def self.merge(caches)
      mx = MLX::Core
      n_state = caches[0].cache.length
      batch = caches.length
      out = new(n_state)

      n_state.times do |e|
        init = caches.map { |c| c[e] }.find { |v| !v.nil? }
        next if init.nil?

        shape = init.shape.dup
        shape[0] = 1
        zero = mx.zeros(shape, init.dtype)
        rows = caches.map { |c| c[e] || zero }
        out[e] = mx.concatenate(rows, 0)
      end

      left_padding_values = caches.map(&:left_padding).compact
      out.left_padding = mx.concatenate(left_padding_values, 0) if left_padding_values.length == batch

      length_values = caches.map(&:lengths).compact
      out.lengths = mx.concatenate(length_values, 0) if length_values.length == batch

      out
    end

    def empty
      @cache.empty? || @cache[0].nil?
    end

    def nbytes
      @cache.compact.reduce(0) { |acc, c| acc + c.nbytes }
    end

    private

    def _indices_array(indices)
      return indices if indices.is_a?(MLX::Core::Array)

      MLX::Core.array(indices, dtype: MLX::Core.int32)
    end
  end

  class ChunkedKVCache < BaseCache
    attr_reader :offset, :chunk_size, :start_position

    def initialize(chunk_size)
      @keys = nil
      @values = nil
      @offset = 0
      @chunk_size = chunk_size
      @start_position = 0
    end

    def maybe_trim_front
      return if @keys.nil? || @keys.shape[2] < @chunk_size

      excess = @keys.shape[2] - @chunk_size
      return if excess <= 0

      @start_position += excess
      @keys = _slice_tail(@keys, @chunk_size)
      @values = _slice_tail(@values, @chunk_size)
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
      [@keys, @values]
    end

    def size
      @offset - @start_position
    end

    def state
      [@keys, @values]
    end

    def state=(v)
      @keys, @values = v
      @offset = @keys ? @keys.shape[2] : 0
    end

    def meta_state
      [@chunk_size, @start_position]
    end

    def meta_state=(v)
      @chunk_size, @start_position = v.map(&:to_i)
    end

    def is_trimmable
      true
    end

    def trim(n)
      return 0 if @keys.nil? || n <= 0

      available = @offset - @start_position
      n = [available, n].min
      @offset -= n
      keep_len = @offset - @start_position
      @keys = _slice_prefix(@keys, keep_len)
      @values = _slice_prefix(@values, keep_len)
      n
    end

    def empty
      @keys.nil?
    end

    def nbytes
      return 0 if @keys.nil?

      @keys.nbytes + @values.nbytes
    end

    private

    def _slice_prefix(array, length)
      return array if array.shape[2] == length

      MLX::Core.split(array, [length], 2)[0]
    end

    def _slice_tail(array, length)
      return array if array.shape[2] == length

      split_idx = array.shape[2] - length
      MLX::Core.split(array, [split_idx], 2)[1]
    end
  end

  class CacheList < BaseCache
    attr_reader :caches

    def initialize(*caches)
      @caches = caches
    end

    def [](idx)
      @caches[idx]
    end

    def is_trimmable
      @caches.all?(&:is_trimmable)
    end

    def trim(n)
      trimmed = 0
      @caches.each do |cache|
        trimmed = cache.trim(n)
      end
      trimmed
    end

    def state
      @caches.map(&:state)
    end

    def state=(v)
      @caches.zip(v).each do |cache, cache_state|
        cache.state = cache_state
      end
    end

    def meta_state
      [
        @caches.map { |c| c.class.name.split("::").last },
        @caches.map(&:meta_state),
      ]
    end

    def meta_state=(v)
      _classes, states = v
      @caches.zip(states).each do |cache, cache_state|
        cache.meta_state = cache_state
      end
    end

    def filter(batch_indices)
      @caches.each { |cache| cache.filter(batch_indices) if cache.respond_to?(:filter) }
    end

    def extend(other)
      @caches.zip(other.caches).each do |cache, other_cache|
        next unless cache.class.instance_method(:extend).owner != Object

        cache.extend(other_cache)
      end
    end

    def self.merge(caches)
      merged = caches[0].caches.each_index.map do |i|
        batch = caches.map { |c| c.caches[i] }
        unless batch[0].class.respond_to?(:merge)
          raise NotImplementedError, "#{batch[0].class} does not implement .merge"
        end

        batch[0].class.merge(batch)
      end
      new(*merged)
    end

    def extract(idx)
      CacheList.new(*@caches.map { |cache| cache.extract(idx) })
    end

    def prepare(**kwargs)
      @caches.each { |cache| cache.prepare(**kwargs) if cache.respond_to?(:prepare) }
    end

    def finalize
      @caches.each { |cache| cache.finalize if cache.respond_to?(:finalize) }
    end

    def size
      @caches.map(&:size).max || 0
    end

    def empty
      @caches.empty? || @caches[0].empty
    end

    def nbytes
      @caches.reduce(0) { |acc, cache| acc + cache.nbytes }
    end

    def self.from_state(state, meta_state)
      classes, metas = meta_state
      caches = state.each_with_index.map do |sub_state, i|
        klass = MlxLm.const_get(classes[i])
        klass.from_state(sub_state, metas[i])
      end
      new(*caches)
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
