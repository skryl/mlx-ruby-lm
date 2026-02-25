# MLX-LM Python → Ruby Conversion Plan

## Overview

Convert the `mlx-lm` Python package into a Ruby gem (`mlx-ruby-lm`) that provides
100% identical functionality, built on top of the `mlx-ruby` gem. The Python
implementation serves as reference only — all code is rewritten idiomatically in Ruby.

Each phase includes **parity tests** that run the equivalent operation in both Python
(via `mlx-lm`) and Ruby (via `mlx-ruby-lm`), comparing outputs numerically within
floating-point tolerance.

---

## Phase 1: Project Scaffolding & Core Infrastructure

**Goal:** Establish the gem structure, configuration system, and weight-loading primitives.

### Deliverables

1. **Gem skeleton**
   - `mlx-ruby-lm.gemspec` with dependency on `mlx` gem (>= 0.30.7)
   - `lib/mlx_lm.rb` entry point
   - Directory layout:
     ```
     lib/mlx_lm/
       version.rb
       model_args.rb        # BaseModelArgs equivalent
       weight_utils.rb      # safetensors loading / tree-map helpers
       config.rb            # config.json + generation_config.json parser
     ```
   - `Rakefile` with test, build tasks
   - `test/test_helper.rb` with parity-test harness (runs Python via subprocess,
     compares against Ruby)

2. **BaseModelArgs**
   - Ruby data class (or `Struct`/`Data`) mirroring Python `BaseModelArgs`
   - Supports arbitrary kwargs → instance attributes
   - `from_dict` class method that filters unknown keys with warnings

3. **Weight loading utilities**
   - `load_safetensors(path)` → returns hash of `{ "layer.weight" => MLX::Core::Array }`
   - `load_sharded_safetensors(directory)` → loads `model*.safetensors` shards
   - `tree_unflatten(flat_hash)` → nested module-state hash

4. **Config loading**
   - Parse `config.json` → `ModelArgs` for the appropriate architecture
   - Parse `generation_config.json` → generation defaults (temperature, top_p, etc.)

### Parity Tests — Phase 1

| # | Test | Method |
|---|------|--------|
| 1.1 | `BaseModelArgs` round-trips from a config dict identically | Compare field values |
| 1.2 | `load_safetensors` returns tensors with identical shapes/dtypes | Shape + dtype check |
| 1.3 | `load_safetensors` tensor values match Python within `atol=1e-6` | Numerical comparison |
| 1.4 | `tree_unflatten` produces identical nested key structure | Key-tree diff |
| 1.5 | Config parsing extracts identical model hyperparameters | Field-by-field comparison |

---

## Phase 2: Tokenizer Integration

**Goal:** Provide tokenizer encode/decode that produces identical token IDs and text.

### Deliverables

1. **TokenizerWrapper**
   - Wraps the `tokenizers` Ruby gem (HuggingFace Rust tokenizers bindings)
   - Falls back to a Python subprocess bridge for tokenizers that require
     `trust_remote_code` or custom HF `AutoTokenizer` behavior
   - API: `encode(text) → Array<Integer>`, `decode(ids) → String`
   - `eos_token_id`, `bos_token_id`, `vocab_size`

2. **Chat template support**
   - Jinja2-compatible template rendering (via `liquid` or `jinja` Ruby gem,
     or a minimal Jinja subset interpreter)
   - `apply_chat_template(messages, add_generation_prompt:)` → token IDs

3. **StreamingDetokenizer**
   - Base class + SentencePiece and BPE implementations
   - Incremental `add_token(id)` → emits text segments without O(T²) re-decoding
   - `finalize` for trailing output

### Parity Tests — Phase 2

| # | Test | Method |
|---|------|--------|
| 2.1 | `encode(text)` produces identical token ID arrays for 10+ diverse strings | Exact match |
| 2.2 | `decode(ids)` produces identical text | Exact string match |
| 2.3 | `apply_chat_template` on a multi-turn conversation → identical IDs | Exact match |
| 2.4 | StreamingDetokenizer accumulates to the same final string | String match |
| 2.5 | Special token handling (BOS, EOS, pad) matches Python | Token ID match |
| 2.6 | `encode` + `decode` round-trip preserves text for 20 samples | String match |

---

## Phase 3: KV Cache & Base Model Architecture (Llama)

**Goal:** Implement the Llama architecture as the foundational model pattern and
all KV cache variants.

### Deliverables

1. **KV Cache implementations** (in `lib/mlx_lm/models/cache.rb`)
   - `KVCache` — simple concatenation cache
   - `RotatingKVCache` — fixed-size circular buffer
   - `QuantizedKVCache` — quantized K/V storage
   - `CacheList` — container for per-layer caches
   - `make_prompt_cache(model)` factory
   - `save_prompt_cache` / `load_prompt_cache` for safetensors serialization

2. **Llama model** (in `lib/mlx_lm/models/llama.rb`)
   - `LlamaModelArgs < BaseModelArgs`
   - `LlamaAttention` — GQA with RoPE, KV cache integration
   - `LlamaMLP` — gate/up/down projections with SiLU
   - `LlamaTransformerBlock` — attention + MLP + RMSNorm
   - `LlamaModel` — embedding + N blocks + final norm + lm_head
   - Forward pass: `model.call(input_ids, cache: nil) → logits`

3. **Model registration**
   - `MODEL_REGISTRY` hash mapping `model_type` string → model class
   - `get_model_class(config)` lookup

### Parity Tests — Phase 3

| # | Test | Method |
|---|------|--------|
| 3.1 | Llama forward pass (random weights, fixed input) → logits match `atol=1e-4` | Numerical |
| 3.2 | RoPE embeddings match Python for multiple sequence lengths | Numerical `atol=1e-6` |
| 3.3 | GQA attention output matches Python | Numerical `atol=1e-5` |
| 3.4 | KVCache state after N forward passes has identical shape/values | Numerical |
| 3.5 | RotatingKVCache wraps correctly at boundary | Numerical + shape |
| 3.6 | QuantizedKVCache quantize/dequantize round-trip within tolerance | Numerical `atol=0.1` |
| 3.7 | Full Llama model with loaded HF weights → identical logits for prompt | Numerical `atol=1e-3` |
| 3.8 | `make_prompt_cache` → save → load round-trip produces identical cache | Numerical |

---

## Phase 4: Sampling & Generation Engine

**Goal:** Implement token-by-token generation with all sampling strategies, producing
identical output sequences given the same seed.

### Deliverables

1. **Sampling** (`lib/mlx_lm/sample_utils.rb`)
   - `make_sampler(temperature:, top_p:, top_k:, min_p:, xtc_probability:, xtc_threshold:)`
   - Categorical sampling with MLX random
   - Greedy (argmax) when temperature == 0

2. **Logits processors** (`lib/mlx_lm/sample_utils.rb`)
   - `make_logits_processors(repetition_penalty:, repetition_context_size:, logit_bias:)`
   - `RepetitionPenalty` — penalizes repeated tokens in context window
   - `LogitBias` — additive bias to specific token logits

3. **Generation core** (`lib/mlx_lm/generate.rb`)
   - `generate_step(prompt_tokens, model, cache, sampler, logits_processors)` — yields
     `(token, logprobs)` per step
   - Prompt prefill with configurable `prefill_step_size` (default 2048)
   - `generate(model, tokenizer, prompt, **kwargs)` → `GenerationResult`
   - `stream_generate(model, tokenizer, prompt, **kwargs)` → `Enumerator` yielding
     text chunks via `StreamingDetokenizer`
   - `batch_generate(model, tokenizer, prompts, **kwargs)` → array of results
   - Stop conditions: `max_tokens`, EOS, stop strings

### Parity Tests — Phase 4

| # | Test | Method |
|---|------|--------|
| 4.1 | Greedy generation (temp=0) produces identical token sequence | Exact match |
| 4.2 | Top-p sampling with fixed seed → identical tokens | Exact match |
| 4.3 | Top-k sampling with fixed seed → identical tokens | Exact match |
| 4.4 | Min-p sampling with fixed seed → identical tokens | Exact match |
| 4.5 | Repetition penalty changes logits identically | Numerical `atol=1e-6` |
| 4.6 | Logit bias applied correctly | Numerical `atol=1e-6` |
| 4.7 | `generate` end-to-end with Llama → identical output text | String match |
| 4.8 | `stream_generate` yields identical incremental text segments | String match per chunk |
| 4.9 | Stop string detection halts at same position | Token count match |
| 4.10 | `batch_generate` produces identical results to sequential calls | String match |
| 4.11 | Generation timing/token count metadata matches | Approximate match |

---

## Phase 5: Model Loading Pipeline & HuggingFace Integration

**Goal:** Load models from HuggingFace Hub or local paths, matching Python's `load()` exactly.

### Deliverables

1. **HuggingFace Hub download** (`lib/mlx_lm/hub_utils.rb`)
   - `snapshot_download(repo_id, revision:, allow_patterns:)` using the `huggingface_hub`
     Ruby gem or HTTP API calls to HF
   - Local cache management (`~/.cache/huggingface/hub/`)
   - Support for gated/private models via HF token

2. **Model loading** (`lib/mlx_lm/utils.rb`)
   - `load(path_or_hf_repo, tokenizer_config:, adapter_path:, lazy:)` →
     `[model, tokenizer]`
   - Dynamic model class lookup via `MODEL_REGISTRY`
   - Weight dequantization / requantization on load
   - Adapter (LoRA) weight merging on load

3. **Quantization-aware loading**
   - Detect `quantization` key in config → apply MLX quantization
   - Detect `quantization_config` → handle AWQ/GPTQ weight transforms
   - `_dequantize_linear(model)` for full-precision fallback

### Parity Tests — Phase 5

| # | Test | Method |
|---|------|--------|
| 5.1 | `load("mlx-community/Llama-3.2-1B-Instruct-4bit")` → identical model config | Field match |
| 5.2 | Loaded model weights have identical shapes and dtypes | Shape + dtype |
| 5.3 | Loaded model weights match numerically (for non-quantized) | `atol=1e-5` |
| 5.4 | Quantized model logits match Python for same prompt | `atol=1e-2` |
| 5.5 | Tokenizer loaded from HF produces identical encodings | Exact match |
| 5.6 | `load` with `lazy=True` defers evaluation correctly | Shape match, lazy check |

---

## Phase 6: Popular Model Architectures (Batch 1)

**Goal:** Implement the most commonly used architectures beyond Llama.

### Deliverables

1. **Mistral** — sliding window attention, different RoPE config
2. **Gemma / Gemma2** — GeGLU, different normalization, logit soft-capping
3. **Qwen2 / Qwen2.5** — bias in attention, different FFN
4. **Phi3** — su/yarn RoPE scaling, block-sparse attention
5. **Mixtral** — Mixture-of-Experts with top-k gating
6. **Starcoder2** — code-focused architecture
7. **Cohere** — layernorm placement variations

Each model follows the established pattern:
- `ModelArgs < BaseModelArgs`
- `Attention`, `MLP`, `TransformerBlock`, `Model` classes
- Registered in `MODEL_REGISTRY`

### Parity Tests — Phase 6

| # | Test | Method |
|---|------|--------|
| 6.1 | Each model: forward pass with random weights → logits match | `atol=1e-4` |
| 6.2 | Each model: loaded from HF → greedy generation matches Python | Exact token match |
| 6.3 | Mixtral: MoE routing selects same experts | Expert index match |
| 6.4 | Mixtral: MoE output matches Python | `atol=1e-4` |
| 6.5 | Gemma: soft-capping produces identical logits | `atol=1e-5` |
| 6.6 | Phi3: su/yarn RoPE matches Python | `atol=1e-5` |
| 6.7 | Each model: end-to-end `generate` matches Python output | String match |

---

## Phase 7: Quantization Engine

**Goal:** Full quantization support — convert, load, and run quantized models.

### Deliverables

1. **MLX native quantization** (`lib/mlx_lm/convert.rb`)
   - `convert(model, config, quantize:)` → quantized model + config
   - Affine quantization (default) — 2, 4, 8-bit
   - MXFP4, MXFP8, NVFP4 modes
   - Per-layer quantization config (skip embeddings, lm_head)

2. **AWQ support** (`lib/mlx_lm/quant/awq.rb`)
   - `AwqQuantizer` — activation-aware weight quantization
   - `_transform_awq_weights` — convert AutoAWQ format to MLX format

3. **GPTQ support** (`lib/mlx_lm/quant/gptq.rb`)
   - `GptqQuantizer` — Hessian-based quantization

4. **Model saving** (`lib/mlx_lm/utils.rb`)
   - `save_model(path, model, tokenizer, config)` — safetensors + config.json
   - Shard large models into multiple files

### Parity Tests — Phase 7

| # | Test | Method |
|---|------|--------|
| 7.1 | Affine quantize → dequantize round-trip matches Python | `atol=0.5` (quantization noise) |
| 7.2 | 4-bit quantized Linear forward matches Python | `atol=1e-2` |
| 7.3 | AWQ weight transform produces identical packed weights | Exact match |
| 7.4 | Quantized model generation output matches Python | String match (greedy) |
| 7.5 | Saved quantized model loads back identically | Weight match |
| 7.6 | MXFP4 quantization matches Python | `atol=1e-1` |
| 7.7 | Per-layer quant config (skip lm_head) applied identically | Config match |

---

## Phase 8: Model Architectures (Batch 2 — Extended)

**Goal:** Expand architecture coverage to include advanced and specialized models.

### Deliverables

1. **Deepseek V2/V3** — MLA attention, MoE with shared experts
2. **Falcon** — multi-query attention, alibi
3. **StableLM** — partial rotary embeddings
4. **Qwen2-MoE** — shared expert + fine-grained MoE
5. **Gemma3** — interleaved local/global attention
6. **Mamba / Mamba2** — SSM (state-space model), `ArraysCache`
7. **RWKV7** — linear attention / RNN hybrid
8. **RecurrentGemma** — recurrent variant
9. **Grok** — large MoE
10. **OpenELM** — per-layer scaling

### Parity Tests — Phase 8

| # | Test | Method |
|---|------|--------|
| 8.1 | Each model: forward pass → logits match Python | `atol=1e-4` |
| 8.2 | Mamba: SSM state update matches Python | `atol=1e-5` |
| 8.3 | Deepseek MLA attention matches Python | `atol=1e-4` |
| 8.4 | MoE routing + output for each MoE model matches | `atol=1e-4` |
| 8.5 | Each model: greedy generation matches Python | Token match |

---

## Phase 9: LoRA & Fine-Tuning

**Goal:** Full LoRA fine-tuning pipeline with training loop.

### Deliverables

1. **LoRA layers** (`lib/mlx_lm/tuner/lora.rb`)
   - `LoRALinear` — low-rank adaptation of linear layers
   - `LoRASwitchLinear` — MoE-compatible LoRA
   - `LoRAEmbedding` — embedding LoRA
   - `apply_lora(model, config)` — patch model layers with LoRA

2. **Training framework** (`lib/mlx_lm/tuner/trainer.rb`)
   - `TrainingArgs` data class (batch_size, epochs, lr, etc.)
   - `train(model, tokenizer, args, train_dataset, val_dataset)` — main loop
   - `evaluate(model, dataset)` — validation
   - Cross-entropy loss with padding mask
   - Gradient accumulation
   - Checkpointing (save/resume)

3. **Dataset loading** (`lib/mlx_lm/tuner/datasets.rb`)
   - Load JSONL/JSON training data
   - Chat-formatted dataset → token sequences
   - Completion-formatted dataset
   - Batch iterator with padding

4. **Adapter fusion** (`lib/mlx_lm/fuse.rb`)
   - `fuse_model(model, adapter_path)` — merge LoRA weights into base
   - De-quantize before fusion option
   - Save fused model

### Parity Tests — Phase 9

| # | Test | Method |
|---|------|--------|
| 9.1 | LoRALinear forward matches Python for same weights | `atol=1e-5` |
| 9.2 | LoRA gradient computation matches Python | `atol=1e-4` |
| 9.3 | One training step produces identical weight updates | `atol=1e-4` |
| 9.4 | Training loss curve matches Python over 50 steps | `atol=1e-2` per step |
| 9.5 | Fused model weights match Python fusion | `atol=1e-5` |
| 9.6 | Fused model generation matches Python | String match (greedy) |
| 9.7 | Checkpoint save/load round-trip preserves training state | Exact match |

---

## Phase 10: CLI & OpenAI-Compatible Server

**Goal:** Feature-complete CLI and HTTP server matching Python's interface.

### Deliverables

1. **CLI** (`lib/mlx_lm/cli.rb`, `exe/mlx_lm`)
   - `mlx_lm generate` — text generation
   - `mlx_lm chat` — interactive REPL
   - `mlx_lm convert` — model conversion/quantization
   - `mlx_lm lora` — LoRA fine-tuning
   - `mlx_lm fuse` — adapter fusion
   - `mlx_lm server` — start HTTP server
   - `mlx_lm benchmark` — performance benchmarking
   - `mlx_lm evaluate` — model evaluation
   - `mlx_lm manage` — download/cache management
   - Argument parsing matching Python's argparse interface

2. **HTTP Server** (`lib/mlx_lm/server.rb`)
   - `POST /v1/completions` — text completion
   - `POST /v1/chat/completions` — chat completion
   - `GET /v1/models` — list models
   - Server-Sent Events streaming
   - Request queuing
   - OpenAI-compatible request/response schema
   - Built on `webrick` or `puma`

3. **Chat REPL** (`lib/mlx_lm/chat.rb`)
   - Interactive conversation with history
   - System prompt support
   - Streaming output to terminal

### Parity Tests — Phase 10

| # | Test | Method |
|---|------|--------|
| 10.1 | CLI `generate` output matches Python CLI | String match |
| 10.2 | CLI argument parsing accepts all Python flags | Flag coverage check |
| 10.3 | Server `/v1/chat/completions` response schema matches OpenAI spec | Schema validation |
| 10.4 | Server streaming response matches Python server | Chunk-by-chunk match |
| 10.5 | Server non-streaming response body matches Python | JSON diff |
| 10.6 | Chat REPL multi-turn context matches Python | Token sequence match |

---

## Phase 11: Model Architectures (Batch 3 — Long Tail)

**Goal:** Complete coverage of remaining architectures.

### Deliverables

All remaining models from `mlx-lm/mlx_lm/models/`, including:

1. **Transformer variants:** OLMo, OLMoE, Nemotron, Exaone, InternLM2, Minicpm,
   GraniteSmall, GraniteMoE, Dbrx, Jamba, Arctic, Telechat, Hunyuan, Solar, PlaMo
2. **Multimodal text backbones:** Qwen3.5-text, Llama4-text, Gemma3-text,
   LFM2-VL-text, Kimi-VL text encoder
3. **Specialized:** BitNet (ternary weights), Phixtral, Ministral3

Each follows the standard pattern and is registered in `MODEL_REGISTRY`.

### Parity Tests — Phase 11

| # | Test | Method |
|---|------|--------|
| 11.1 | Each model: forward pass → logits match | `atol=1e-4` |
| 11.2 | Each model: greedy generation matches Python | Token match |
| 11.3 | BitNet ternary weight handling matches Python | Exact weight match |

---

## Phase 12: Advanced Features & Polish

**Goal:** Implement remaining advanced capabilities and achieve full feature parity.

### Deliverables

1. **Prompt caching**
   - `cache_prompt(model, tokenizer, prompt)` → saved cache file
   - Load cached prompt for fast repeated inference

2. **Speculative decoding**
   - Draft model integration in generation loop
   - Accept/reject logic

3. **Distributed inference**
   - Pipeline parallelism (`pipeline_load`)
   - Tensor parallelism (`sharded_load`)
   - Distributed generation with `MLX::Distributed`

4. **Evaluation**
   - Perplexity computation (`lib/mlx_lm/perplexity.rb`)
   - Log-likelihood scoring

5. **Benchmarking**
   - Tokens/sec measurement
   - Memory tracking
   - Prompt processing vs generation speed

6. **GGUF support**
   - GGUF tokenizer loading
   - GGUF weight export

7. **Model sharing / upload**
   - `upload(path, repo_id)` to HuggingFace Hub
   - Model card generation

### Parity Tests — Phase 12

| # | Test | Method |
|---|------|--------|
| 12.1 | Cached prompt generation matches uncached | String match |
| 12.2 | Perplexity score matches Python within 0.1% | Numerical |
| 12.3 | Speculative decoding output matches standard generation | String match |
| 12.4 | GGUF export produces identical file | Byte-level or weight match |
| 12.5 | Benchmark tokens/sec within 20% of Python (same hardware) | Approximate |

---

## Parity Test Infrastructure

### Test Harness Design

```ruby
# test/test_helper.rb

module ParityTest
  # Runs a Python snippet, captures output as JSON, returns parsed result
  def python_eval(code)
    result = `python3 -c "#{code}"`
    JSON.parse(result)
  end

  # Compares MLX arrays between Ruby and Python
  def assert_array_parity(ruby_array, python_values, atol: 1e-5)
    ruby_values = ruby_array.tolist
    assert_in_delta_array(ruby_values, python_values, atol)
  end

  # Compares generation output
  def assert_generation_parity(ruby_text, python_text)
    assert_equal python_text.strip, ruby_text.strip
  end
end
```

### Running Parity Tests

```bash
# Run all parity tests
bundle exec rake test:parity

# Run parity tests for a specific phase
bundle exec rake test:parity[phase3]

# Run parity tests for a specific model
bundle exec rake test:parity[llama]

# Generate parity report
bundle exec rake test:parity_report
```

### CI Integration

- Each phase PR must pass all parity tests for that phase and prior phases
- Parity test failures block merge
- Nightly full-suite parity run against latest `mlx-lm` release

---

## Dependency Strategy

| Python Dependency | Ruby Equivalent | Notes |
|---|---|---|
| `mlx` | `mlx` gem (mlx-ruby) | Already available, feature-complete |
| `transformers` | `tokenizers` gem + custom code | HF tokenizers Rust bindings for Ruby |
| `numpy` | `mlx` gem array ops | MLX covers all needed operations |
| `huggingface_hub` | HTTP API + `down` gem or custom | Direct HF Hub HTTP API calls |
| `sentencepiece` | `tokenizers` gem | Covers SentencePiece models |
| `jinja2` | `liquid` or custom Jinja subset | For chat templates |
| `pyyaml` | `yaml` (stdlib) | Built into Ruby |
| `tqdm` | `ruby-progressbar` | Progress bars |
| `protobuf` | `google-protobuf` gem | If needed for legacy tokenizers |

---

## Milestone Summary

| Phase | Scope | Est. Files | Cumulative Parity Tests |
|-------|-------|-----------|------------------------|
| 1 | Scaffolding & infrastructure | ~10 | 5 |
| 2 | Tokenizer | ~5 | 11 |
| 3 | KV Cache & Llama | ~5 | 19 |
| 4 | Generation engine | ~3 | 30 |
| 5 | Model loading & HF integration | ~4 | 36 |
| 6 | 7 popular architectures | ~7 | 43 |
| 7 | Quantization | ~5 | 50 |
| 8 | 10 extended architectures | ~10 | 55 |
| 9 | LoRA & fine-tuning | ~6 | 62 |
| 10 | CLI & server | ~5 | 68 |
| 11 | ~30 remaining architectures | ~30 | 71 |
| 12 | Advanced features & polish | ~8 | 76 |

**After Phase 5**, the gem is usable for basic inference with Llama models.
**After Phase 7**, quantized inference works for 8 architectures.
**After Phase 10**, the gem is a full drop-in replacement for common use cases.
**After Phase 12**, the gem has 100% feature parity with `mlx-lm`.

---

## Key Design Decisions

1. **Idiomatic Ruby** — Use Ruby conventions (snake_case, blocks/procs for callbacks,
   `Enumerable` for streaming, keyword arguments) rather than transliterating Python.

2. **Module pattern for models** — Each model is a `MLX::NN::Module` subclass, matching
   how `mlx-ruby` already structures neural network layers.

3. **Lazy evaluation preserved** — Follow MLX's lazy computation model; only `eval`
   when results are needed.

4. **Tokenizer strategy** — Primary: `tokenizers` gem (Rust HF bindings). Fallback:
   Python subprocess for exotic tokenizers. Goal: eliminate Python fallback by Phase 12.

5. **No monkey-patching** — Clean module hierarchy under `MlxLm::` namespace.

6. **Progressive testing** — Every phase is independently testable. Earlier phases
   don't depend on later phases being complete.

---

## MLX-Ruby Gaps & Workarounds

The following issues were discovered during implementation. Each required a
workaround in the `mlx-ruby-lm` codebase.

### API Issues

| # | Issue | Severity | Workaround |
|---|-------|----------|------------|
| 1 | `mx.array(values, dtype: ...)` raises `ArgumentError` — the `dtype:` keyword is rejected even though the error message says it accepts `Dtype`, symbol, or string | High | Create float32 array first, then `.astype(mx.int32)` etc. |
| 2 | `mx.mean(x, axis, keepdims: true)` not supported — no `keepdims` parameter on `mean` | Medium | `mx.expand_dims(mx.mean(x, axis), -1)` |
| 3 | `MLX::NN::Dropout.new(p: 0.5)` rejects keyword arg — constructor only accepts positional `Dropout.new(0.5)` | Low | Use positional argument |
| 4 | `mx.random_uniform` only works with float dtypes — passing `mx.int32` raises error | Low | Generate as float32, then `.astype(mx.int32)` |
| 5 | `mx.save_safetensors` not available — MLX not compiled with `MLX_BUILD_SAFETENSORS=ON` | Medium | Use Ruby `safetensors` gem for serialization |
| 6 | No `SwitchGLU` layer — Python mlx-lm uses it for efficient stacked MoE experts | Medium | Per-token expert routing loop (functional but slower) |

### Ruby ↔ MLX Coercion Issues

| # | Issue | Severity | Workaround |
|---|-------|----------|------------|
| 7 | `Float * MLX::Array` raises `TypeError` — Ruby `Float#*` can't coerce MLX arrays | High | Always put MLX array on the left: `array * scalar` |
| 8 | `Float + MLX::Array` raises `TypeError` — same coercion issue with addition | High | `array + scalar` instead of `scalar + array` |
| 9 | No unary negation `-array` | Low | `MLX::Core.negative(array)` |
| 10 | No comparison operators `>`, `<` on arrays | Low | `MLX::Core.greater()`, `MLX::Core.less()` |

### NN Module System

| # | Issue | Severity | Workaround |
|---|-------|----------|------------|
| 11 | Instance variables invisible to Module traversal — `@x = Module.new(...)` doesn't register children in `@state`, so `children`, `leaf_modules`, `parameters`, `load_weights`, and `nn.quantize` all miss them | **Critical** | Must use `self.x = Module.new(...)` (goes through `method_missing` → `@state`) for every child module in every model. Required refactoring all model files. |
| 12 | `update_modules_impl` missing Module→Hash recursion — when `nn.quantize` replaces layers, the code didn't handle `current_value=Module` + `new_value=Hash` | High | Patched `mlx-ruby/lib/mlx/nn/base.rb` to add `elsif current_value.is_a?(Module) && (new_value.is_a?(Hash) \|\| new_value.is_a?(Array))` branch for recursive descent. |

### Safetensors Gem Compatibility

| # | Issue | Severity | Workaround |
|---|-------|----------|------------|
| 13 | Dtype string format mismatch — Ruby `safetensors` gem v0.2.2 expects lowercase `"float32"` for serialization but returns uppercase `"F32"` on deserialization | Medium | `weight_utils.rb` accepts both formats: `dtype_str == "F32" \|\| dtype_str == "float32"` |

### Impact Summary

- **Issues 7, 8, 11** were the most pervasive — they affected every model file and
  many utility modules. Issue 11 alone required rewriting every constructor in every
  model architecture.
- **Issue 12** required patching mlx-ruby itself (the only upstream change).
- **Issue 1** affected any code path that creates typed arrays from Ruby values.
- **Issues 9, 10** are minor and only arise in specific model architectures.

### Recommendations for mlx-ruby Upstream

1. Fix `mx.array(values, dtype:)` to accept dtype keyword argument
2. Add `keepdims:` parameter to reduction ops (`mean`, `sum`, etc.)
3. Implement Ruby `coerce` on `MLX::Core::Array` so `Float * array` works
4. Compile with `MLX_BUILD_SAFETENSORS=ON` by default
5. Add `SwitchGLU` layer for MoE model support
6. Document the `self.x =` vs `@x =` requirement for Module children prominently

