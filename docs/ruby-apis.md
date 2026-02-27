# Ruby APIs

[Back to Documentation Index](index.md)

Detailed API inventory for the surfaces loaded by:

```ruby
require "mlx"
require "mlx_lm"
```

## Primary Entry Points

| Module/Class | Purpose | Key Methods |
| --- | --- | --- |
| `MlxLm::LoadUtils` | Load model + tokenizer from local directory | `load`, `load_model`, `load_tokenizer` |
| `MlxLm::Generate` | Token generation APIs | `generate_step`, `stream_generate`, `generate` |
| `MlxLm::SampleUtils` | Samplers and logits processors | `make_sampler`, `make_logits_processors`, `apply_top_k`, `apply_top_p`, `apply_min_p`, `make_repetition_penalty` |
| `MlxLm::ChatTemplate` | Message-to-prompt formatting | `apply`, `apply_default`, `apply_chatml` |
| `MlxLm::Server` | OpenAI-style HTTP server | `start` |
| `MlxLm::Models` | Model registry and resolver | `register`, `get_classes`, constants `REGISTRY`, `REMAPPING` |

## Loading and Tokenization

### `MlxLm::LoadUtils`

- `load(model_path, tokenizer_config: nil)` -> `[model, tokenizer]`
- `load_model(model_path)`:
  - loads `config.json` via `MlxLm::Config.load`
  - resolves model classes from `MlxLm::Models.get_classes`
  - loads safetensor shards and applies model `sanitize` if available
  - applies quantization from config when `config["quantization"]` is present
- `load_tokenizer(model_path)`:
  - loads `tokenizer.json`
  - reads EOS metadata from `tokenizer_config.json` / `config.json`
  - returns `MlxLm::TokenizerWrapper`

### `MlxLm::TokenizerWrapper`

- `encode(text, add_special_tokens: true)`
- `decode(ids, skip_special_tokens: false)`
- token helpers: `eos_token`, `eos_token_id`, `eos_token_ids`, `bos_token`, `bos_token_id`
- vocab helpers: `vocab_size`, `id_to_token`, `token_to_id`
- `detokenizer` -> `MlxLm::StreamingDetokenizer`

### `MlxLm::StreamingDetokenizer`

- Incremental decode helper for streaming generation.
- Methods: `add_token`, `finalize`, `text`, `reset`.

## Generation and Sampling

### `MlxLm::Generate`

- `generate_step(prompt, model, ...)`:
  - low-level token-step enumerator
  - yields `[token_id, logprobs]`
- `stream_generate(model, tokenizer, prompt, max_tokens: 256, **kwargs)`:
  - yields `MlxLm::GenerationResponse`
  - `GenerationResponse` fields:
    - `text`, `token`, `logprobs`
    - `prompt_tokens`, `prompt_tps`
    - `generation_tokens`, `generation_tps`
    - `peak_memory`, `finish_reason`
- `generate(model, tokenizer, prompt, verbose: false, **kwargs)`:
  - non-streaming convenience wrapper
  - optionally prints streaming text + perf stats when `verbose: true`

### `MlxLm::SampleUtils`

- `make_sampler(temp:, top_p:, min_p:, min_tokens_to_keep:, top_k:)`
- `make_logits_processors(repetition_penalty:, repetition_context_size:)`
- filtering/sampling helpers:
  - `apply_top_k`
  - `apply_min_p`
  - `apply_top_p`
  - `categorical_sampling`
  - `make_repetition_penalty`

## Server and Prompt Formatting

### `MlxLm::ChatTemplate`

- `apply(messages, template: :default)` returns prompt text.
- current templates: `:default`, `:chatml` (same formatting path).

### `MlxLm::Server`

- `start(model_path:, host: "127.0.0.1", port: 8080)` boots a WEBrick server.
- endpoints:
  - `GET /v1/models`
  - `POST /v1/chat/completions` (supports streaming SSE when `stream: true`)
- schema helper classes:
  - `ChatCompletionRequest`
  - `ChatCompletionResponse`
  - `ChatCompletionChunk`
  - `ModelsListResponse`

## Utility Modules

| Module | Purpose | Key Methods |
| --- | --- | --- |
| `MlxLm::Quantize` | Quantize/dequantize model layers | `quantize_model`, `dequantize_model`, `bits_per_weight` |
| `MlxLm::Tuner` | LoRA adapters and layer injection | `LoRALinear`, `LoRAEmbedding`, `apply_lora_layers` |
| `MlxLm::Perplexity` | Log-likelihood and perplexity | `log_likelihood`, `compute` |
| `MlxLm::Benchmark` | Throughput and model stats | `measure_generation`, `model_stats` |
| `MlxLm::ConvertUtils` | Dtype and size helpers | `convert_dtype`, `count_parameters`, `model_size_bytes` |
| `MlxLm::Config` | Config file loader | `load` |
| `MlxLm::WeightUtils` | Safetensors loaders and tree utils | `load_safetensors`, `load_sharded_safetensors`, `tree_unflatten` |

## Minimal End-to-End Example

```ruby
require "mlx"
require "mlx_lm"

model, tokenizer = MlxLm::LoadUtils.load("/path/to/model")
sampler = MlxLm::SampleUtils.make_sampler(temp: 0.7, top_p: 0.9)

text = MlxLm::Generate.generate(
  model,
  tokenizer,
  "Explain KV cache in one paragraph.",
  max_tokens: 128,
  sampler: sampler
)

puts text
```

## Related Docs

- [Documentation Index](index.md)
- [Installation](installation.md)
- [CLI Usage](cli.md)
- [Models](models.md)
