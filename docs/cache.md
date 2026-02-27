# Cache

[Back to Documentation Index](index.md)

This page documents the cache hierarchy in `lib/mlx_lm/models/cache.rb`.

## Class Hierarchy

- `MlxLm::BaseCache`
  - `MlxLm::KVCache`
  - `MlxLm::RotatingKVCache`
  - `MlxLm::QuantizedKVCache`
  - `MlxLm::ArraysCache`
  - `MlxLm::ChunkedKVCache`
  - `MlxLm::CacheList`

All cache classes share:

- `state` / `state=`
- `meta_state` / `meta_state=`
- `size`
- `nbytes`
- `empty`
- `is_trimmable`
- `.from_state(state, meta_state)`

## `KVCache`

Purpose:

- Standard autoregressive KV cache.
- Appends new keys/values along sequence axis `2`.

Key methods:

- `update_and_fetch(keys, values)` -> `[all_keys, all_values]`
- `trim(n)` trims from the tail (keeps earliest prefix)
- `to_quantized(group_size: 64, bits: 4)` -> `QuantizedKVCache`
- `.merge(caches)` batches multiple caches by left-padding shorter sequences

## `RotatingKVCache`

Purpose:

- Fixed-size sliding-window cache.

Constructor:

```ruby
MlxLm::RotatingKVCache.new(max_size:, keep: 0)
```

Behavior:

- On overflow:
  - if `keep > 0`, keeps first `keep` tokens and newest tail
  - else drops oldest tokens from the front
- `size` is capped at `max_size`.

## `QuantizedKVCache`

Purpose:

- Stores KV tensors in quantized form.

Constructor:

```ruby
MlxLm::QuantizedKVCache.new(group_size: 64, bits: 8)
```

Behavior:

- `update_and_fetch` quantizes incoming keys/values via `MLX::Core.quantize`.
- Internal `state` stores quantization tuples (arrays of tensors).
- Supports `trim`, `nbytes`, and `.from_state`.

## `ArraysCache`

Purpose:

- Generic state-array cache used by recurrent/stateful model blocks.

Features:

- Stores arbitrary per-slot arrays (`cache[idx]`).
- Optional `left_padding` and `lengths`.
- Helpers:
  - `filter(batch_indices)`
  - `extend(other)`
  - `extract(idx)`
  - `prepare(lengths: ...)`
  - `advance(n)`
  - `make_mask(n)`
  - `finalize`
- `.merge(caches)` merges batched state.

## `ChunkedKVCache`

Purpose:

- KV cache with explicit chunk retention.

Key behavior:

- `update_and_fetch` appends sequence.
- `maybe_trim_front` keeps only the most recent `chunk_size` tokens and tracks dropped prefix in `start_position`.
- `size` is `offset - start_position`.

## `CacheList`

Purpose:

- Composite cache over multiple sub-caches (for mixed cache types in one layer).

Features:

- Indexing (`cache_list[idx]`)
- Composite `state`/`meta_state`
- `trim`, `filter`, `extend`, `extract`, `prepare`, `finalize`
- `.merge(caches)`
- `.from_state(state, meta_state)` reconstructs cache classes by class name

## Prompt Cache Helpers (`MlxLm::Cache`)

### `make_prompt_cache(model, max_kv_size: nil)`

Behavior:

- If model implements `make_cache`, that is used directly.
- Otherwise:
  - `Array<KVCache>` for each model layer
  - or `Array<RotatingKVCache>` when `max_kv_size` is provided (`keep: 4`)

### `save_prompt_cache(path, cache)`

Behavior:

- Serializes per-layer `keys`/`values` tensors to Safetensors.
- Also writes `_meta_offsets`.
- Dtypes are normalized to `float32` or `int32` before serialization.

### `load_prompt_cache(path, model)`

Behavior:

- Loads tensors via `MlxLm::WeightUtils.load_safetensors`.
- Reconstructs `Array<KVCache>` with one entry per `model.layers.length`.
- Restores `layer.<i>.keys` / `layer.<i>.values` where present.

## Caveats

- `save_prompt_cache` assumes each entry behaves like a KV cache (`state` destructures to `[keys, values]` and `offset` exists).
- `load_prompt_cache` always rebuilds plain `KVCache` entries, even if the model normally uses `RotatingKVCache`, `ArraysCache`, `CacheList`, or other custom cache structures.
- `_meta_offsets` are loaded, but current `load_prompt_cache` path does not apply them to cache objects.

[Back to Documentation Index](index.md)
