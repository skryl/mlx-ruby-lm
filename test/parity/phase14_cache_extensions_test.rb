require_relative "../test_helper"

class Phase14CacheExtensionsTest < Minitest::Test
  include ParityTestHelpers

  def setup
    @mx = MLX::Core
  end

  def test_base_cache_default_contract
    cache = MlxLm::BaseCache.new

    assert_equal [], cache.state
    assert_equal "", cache.meta_state
    assert_raises(ArgumentError) { cache.state = [@mx.ones([1])] }
    assert_raises(ArgumentError) { cache.meta_state = "invalid" }
  end

  def test_arrays_cache_masks_extract_and_finalize
    cache = MlxLm::ArraysCache.new(2, left_padding: [1, 2])
    cache[0] = @mx.ones([2, 3])
    cache[1] = @mx.ones([2, 3]) * 2.0

    mask = cache.make_mask(4)
    @mx.eval(mask)
    assert_equal [[false, true, true, true], [false, false, true, true]], mask.tolist

    cache.left_padding = nil
    cache.prepare(lengths: [3, 1])
    len_mask = cache.make_mask(4)
    @mx.eval(len_mask)
    assert_equal [[true, true, true, false], [true, false, false, false]], len_mask.tolist

    extracted = cache.extract(1)
    @mx.eval(extracted[0], extracted[1])
    assert_equal [1, 3], extracted[0].shape
    assert_equal [1, 3], extracted[1].shape

    cache.advance(1)
    advanced_mask = cache.make_mask(4)
    @mx.eval(advanced_mask)
    assert_equal [[true, true, false, false], [false, false, false, false]], advanced_mask.tolist

    cache.finalize
    assert_nil cache.make_mask(4)
  end

  def test_quantized_kv_cache_update_trim_and_restore
    cache = MlxLm::QuantizedKVCache.new(group_size: 32, bits: 4)

    k1 = @mx.ones([1, 2, 3, 32])
    v1 = @mx.ones([1, 2, 3, 32]) * 2.0
    qk, qv = cache.update_and_fetch(k1, v1)
    @mx.eval(*qk, *qv)

    assert_equal 3, cache.offset
    assert_equal [1, 2, 3, 4], qk[0].shape
    assert_equal [1, 2, 3, 1], qk[1].shape

    k2 = @mx.ones([1, 2, 2, 32]) * 3.0
    v2 = @mx.ones([1, 2, 2, 32]) * 4.0
    qk, qv = cache.update_and_fetch(k2, v2)
    @mx.eval(*qk, *qv)

    assert_equal 5, cache.offset
    assert_equal [1, 2, 5, 4], qk[0].shape

    trimmed = cache.trim(2)
    assert_equal 2, trimmed
    assert_equal 3, cache.offset
    assert_equal 3, cache.state[0][0].shape[2]

    restored = MlxLm::QuantizedKVCache.from_state(cache.state, cache.meta_state)
    assert_equal 3, restored.offset
    assert_equal 32, restored.group_size
    assert_equal 4, restored.bits
  end

  def test_chunked_kv_cache_front_trim_and_trim
    cache = MlxLm::ChunkedKVCache.new(4)

    k1 = @mx.ones([1, 1, 3, 8])
    v1 = @mx.ones([1, 1, 3, 8])
    cache.update_and_fetch(k1, v1)

    k2 = @mx.ones([1, 1, 3, 8]) * 2.0
    v2 = @mx.ones([1, 1, 3, 8]) * 3.0
    keys, values = cache.update_and_fetch(k2, v2)
    @mx.eval(keys, values)

    assert_equal [1, 1, 6, 8], keys.shape
    assert_equal 6, cache.offset

    cache.maybe_trim_front
    keys, values = cache.state
    @mx.eval(keys, values)

    assert_equal 2, cache.start_position
    assert_equal [1, 1, 4, 8], keys.shape
    assert_equal 4, cache.size

    trimmed = cache.trim(1)
    assert_equal 1, trimmed
    assert_equal 5, cache.offset
    assert_equal 3, cache.size
    assert_equal [1, 1, 3, 8], cache.state[0].shape
  end

  def test_cache_list_trim_and_restore
    k = @mx.ones([1, 1, 4, 8])
    v = @mx.ones([1, 1, 4, 8])

    kv1 = MlxLm::KVCache.new
    kv2 = MlxLm::KVCache.new
    kv1.update_and_fetch(k, v)
    kv2.update_and_fetch(k, v)

    list = MlxLm::CacheList.new(kv1, kv2)
    assert list.is_trimmable
    assert_equal 4, list.size

    trimmed = list.trim(2)
    assert_equal 2, trimmed
    assert_equal 2, kv1.offset
    assert_equal 2, kv2.offset

    arr = MlxLm::ArraysCache.new(1)
    arr[0] = @mx.ones([1, 4])
    mixed = MlxLm::CacheList.new(arr, MlxLm::KVCache.new)
    refute mixed.is_trimmable

    restored = MlxLm::CacheList.from_state(mixed.state, mixed.meta_state)
    assert_instance_of MlxLm::ArraysCache, restored[0]
    assert_instance_of MlxLm::KVCache, restored[1]
  end
end
