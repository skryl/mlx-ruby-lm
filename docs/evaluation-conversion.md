# Evaluation and Conversion Utilities

[Back to Documentation Index](index.md)

This page covers utility modules in:

- `lib/mlx_lm/perplexity.rb`
- `lib/mlx_lm/benchmark.rb`
- `lib/mlx_lm/convert_utils.rb`

## `MlxLm::Perplexity`

### `compute(model, tokens, batch_size: nil)`

- Calls `log_likelihood`.
- Computes `exp(-log_likelihood / (tokens.size - 1))`.
- Returns a numeric perplexity value.

### `log_likelihood(model, tokens, batch_size: nil)`

- Accepts `Array` or `MLX::Core::Array`.
- Converts input to int32 array if needed.
- Runs one full forward pass on shape `[1, total_tokens]`.
- Computes log-softmax over vocab at each position.
- Sums `log P(token[i+1] | token[0..i])` across sequence.

Caveat:

- `batch_size:` exists in signature but is not used in the current implementation.

## `MlxLm::Benchmark`

### `measure_generation(model, prompt_tokens: 32, gen_tokens: 64, vocab_size: 32000)`

Returns:

- `:prompt_tokens`
- `:prompt_time`
- `:prompt_tps`
- `:generation_tokens`
- `:generation_time`
- `:generation_tps`

Behavior:

- Builds a random int32 prompt.
- Creates cache via `MlxLm::Cache.make_prompt_cache(model)`.
- Measures prompt prefill and iterative decode throughput.
- Uses greedy argmax token selection.

### `model_stats(model)`

Returns:

- `:total_params` from `model.parameters`
- `:num_layers` (`model.layers.length` when available, else `0`)

## `MlxLm::ConvertUtils`

### `convert_dtype(array, target_dtype)`

Supports:

- Direct `MLX::Core::Dtype`
- Symbol/string names in map:
  - `float32`
  - `float16`
  - `bfloat16`
  - `int8`
  - `int32`

Raises `ArgumentError` for unknown dtype names.

### `count_parameters(model)`

- Flattens `model.parameters`.
- Returns total element count as an integer.

### `model_size_bytes(model)`

- Flattens `model.parameters`.
- Estimates bytes from dtype width mapping (`float32` => 4, `float16` => 2, etc.).
- Falls back to `4` bytes per element for unknown dtypes.

## Caveats

- `Perplexity` currently runs the whole token sequence in one pass; there is no chunked evaluation path yet.
- `Benchmark.measure_generation` is a throughput microbenchmark on synthetic random prompts and greedy decoding; it is not a quality benchmark.
- `count_parameters` and `model_size_bytes` only account for model parameter tensors, not runtime cache memory or other process overhead.

[Back to Documentation Index](index.md)
