# Generation and Sampling

[Back to Documentation Index](index.md)

This page documents the current generation surfaces in `lib/mlx_lm/generate.rb` and sampling/logits processors in `lib/mlx_lm/sample_utils.rb`.

## Primary APIs

```ruby
MlxLm::Generate.generate_step(
  prompt,
  model,
  max_tokens: 256,
  sampler: nil,
  logits_processors: nil,
  max_kv_size: nil,
  prompt_cache: nil,
  prefill_step_size: 2048
) # => Enumerator yielding [token_id, logprobs]

MlxLm::Generate.stream_generate(model, tokenizer, prompt, max_tokens: 256, **kwargs)
# => Enumerator yielding MlxLm::GenerationResponse

MlxLm::Generate.generate(model, tokenizer, prompt, verbose: false, **kwargs)
# => String
```

```ruby
MlxLm::SampleUtils.make_sampler(
  temp: 0.0,
  top_p: 0.0,
  min_p: 0.0,
  min_tokens_to_keep: 1,
  top_k: 0
) # => Proc(logprobs) -> token

MlxLm::SampleUtils.make_logits_processors(repetition_penalty: nil, repetition_context_size: 20)
# => Array<Proc(tokens, logits) -> logits>

MlxLm::SampleUtils.apply_top_k(logprobs, top_k)
MlxLm::SampleUtils.apply_top_p(logprobs, top_p)
MlxLm::SampleUtils.apply_min_p(logprobs, min_p, min_tokens_to_keep = 1)
MlxLm::SampleUtils.categorical_sampling(logits, temp)
MlxLm::SampleUtils.make_repetition_penalty(penalty, context_size = 20)
```

## `generate_step` Behavior

`generate_step` is the low-level token iterator.

- Requires non-empty prompt (`ArgumentError` if empty).
- Uses `sampler ||= ->(x) { mx.argmax(x, -1) }` (greedy default).
- Creates/uses prompt cache:
  - uses provided `prompt_cache` when passed.
  - otherwise uses `Cache.make_prompt_cache(model, max_kv_size: max_kv_size)`.
- Prefills prompt in chunks of `prefill_step_size` for all but the last prompt token.
- Computes logits for the last position, normalizes to log-probabilities, samples next token, and yields per step.
- Yields `[token_id, logprobs]` where:
  - `token_id` is `y.item` (Ruby scalar integer).
  - `logprobs` is a 1D MLX array of vocab log-probabilities.
- Stops after exactly `max_tokens` yielded tokens.

Logits processor contract:

- Each processor is called as `processor.call(tokens, logits)`.
- `tokens` is a 1D token history accumulated inside `generate_step`.
- Processors are applied before log-softmax and sampling.

## `stream_generate` Behavior

`stream_generate` wraps `generate_step` and emits structured responses.

- Accepts prompt as `String`, Ruby token array, or `MLX::Core::Array`.
- If prompt is a string, encodes it with tokenizer first.
- Coerces prompt to `MLX::Core.array(..., dtype: uint32)` when needed.
- Yields `MlxLm::GenerationResponse` with fields:
  - `text`, `token`, `logprobs`
  - `prompt_tokens`, `prompt_tps`
  - `generation_tokens`, `generation_tps`
  - `peak_memory`, `finish_reason`

Finish behavior:

- If generated token is in `tokenizer.eos_token_ids`, emits one final response with `finish_reason: "stop"` and breaks.
- Otherwise sets `finish_reason: "length"` only when internal condition `(n + 1) == max_tokens`.

## `generate` Behavior

`generate` iterates `stream_generate`, concatenates `resp.text`, and returns the final string.

- `verbose: true` prints streamed text plus token-per-second summary.
- `verbose: false` returns string only.

## Sampling Utilities

`make_sampler`:

- `temp == 0.0` returns greedy argmax sampler immediately.
- `temp > 0` composes filters in this order: top-p, min-p, top-k, then categorical sampling.
- Filtering is optional and controlled by arguments:
  - top-p active only when `0 < top_p < 1.0`
  - min-p active when `min_p != 0.0`
  - top-k active when `top_k > 0`

`make_logits_processors`:

- Returns `[]` by default.
- Adds repetition-penalty processor only when `repetition_penalty` is set and non-zero.

`apply_top_k(logprobs, top_k)`:

- Masks non-top-k tokens to `-Infinity`.
- Requires integer `top_k` with `top_k > 0` and `top_k < vocab_size`.

`apply_top_p(logprobs, top_p)`:

- Computes cumulative probabilities and masks tokens outside retained nucleus set.
- Returns filtered logprobs with masked positions set to `-Infinity`.

`apply_min_p(logprobs, min_p, min_tokens_to_keep = 1)`:

- Enforces probability floor relative to highest-probability token.
- Always keeps at least `min_tokens_to_keep` highest tokens.

`categorical_sampling(logits, temp)`:

- Samples with `MLX::Core.categorical(logits * (1.0 / temp))`.

`make_repetition_penalty(penalty, context_size = 20)`:

- Returns processor `(tokens, logits) -> logits`.
- Uses last `context_size` tokens.
- Applies penalty by:
  - multiplying negative logits by `penalty`
  - dividing non-negative logits by `penalty`

## Example: Sampler + Logits Processors

```ruby
sampler = MlxLm::SampleUtils.make_sampler(temp: 0.7, top_p: 0.9, top_k: 40)
processors = MlxLm::SampleUtils.make_logits_processors(
  repetition_penalty: 1.1,
  repetition_context_size: 20
)

MlxLm::Generate.stream_generate(
  model,
  tokenizer,
  "Explain rotary embeddings simply.",
  max_tokens: 128,
  sampler: sampler,
  logits_processors: processors
).each do |resp|
  print resp.text
end
```

## Caveats

- `make_sampler(temp: 0.0, ...)` ignores `top_p`, `min_p`, and `top_k` and always uses greedy argmax.
- `apply_top_k` currently enforces `top_k < vocab_size` (not `<=`).
- `make_logits_processors` is opt-in; generation helpers do not auto-attach repetition penalty unless you pass `logits_processors:`.
- With `generate_step` + logits processors, prefill consumes all prompt tokens except the last token before processor tracking begins; processor token history does not include the full original prompt for prompts longer than one token.
- Current `stream_generate`/`generate` behavior is off by one for `max_tokens > 1` (they emit up to `max_tokens - 1` responses/tokens in non-EOS paths).
- `GenerationResponse#peak_memory` is currently fixed to `0.0` (placeholder, not measured memory).

[Back to Documentation Index](index.md)
