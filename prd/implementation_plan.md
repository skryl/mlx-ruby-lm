# mlx-ruby-lm Implementation Plan

**Status:** Active
**Date:** 2026-02-25
**Reference:** [Python-Ruby Parity Checklist](2026_02_25_python_ruby_parity_checklist.md)

## Overview

This is the executable implementation plan for achieving full model architecture
parity between Python `mlx-lm` and Ruby `mlx-ruby-lm`. It is organized as a
sequence of phases, each with concrete deliverables, file-by-file implementation
targets, and — critically — a **native library gap report** checkpoint at the end
of each phase.

The gap report captures any missing or broken functionality in the underlying
native libraries (`mlx-ruby`, `mlx-onnx`) discovered during implementation.
These reports drive upstream fixes and prevent blocked downstream phases.

### Scope

- Shared model infrastructure (RoPE variants, MLA, SSM, cache, etc.)
- All ~96 missing model architectures
- HuggingFace Hub integration
- ONNX export validation for all architectures

### Out of Scope

- Training pipeline (trainer, datasets, callbacks, losses)
- Batch generation (BatchGenerator, BatchKVCache)
- Advanced quantization (AWQ, GPTQ, DWQ)
- Speculative decoding
- Tool parsers
- Evaluation harness (lm-eval)
- GGUF export

---

## Phase 1: Shared Model Infrastructure

**Objective:** Build all shared modules that downstream model phases depend on.
No model files are ported in this phase — only reusable infrastructure.

### 1A: RoPE Variants

**Python source:** `mlx-lm/mlx_lm/models/rope_utils.py` (263 lines)

**Deliverables:**
- `lib/mlx_lm/models/rope_utils.rb`
  - `SuScaledRoPE` — longrope-style scaled frequencies with mscale (used by phi3small, minicpm3)
  - `Llama3RoPE` — frequency correction with low/high wavelen smoothing (used by llama3+, mistral3)
  - `YarnRoPE` — YaRN extrapolation with linear ramp mask (used by deepseek_v2, deepseek_v3, hunyuan)
  - `initialize_rope(dims, base, traditional, scaling_config:, max_position_embeddings:)` — factory that returns the correct RoPE class based on `rope_scaling.type`

**Key MLX ops used:** `mx.fast.rope` (with custom `freqs`), `mx.arange`, `mx.clip`, `mx.where`

**Tests:**
- Parity test: each RoPE variant vs Python for identical config → `atol=1e-5`
- `initialize_rope` returns correct class for each `rope_type`: `default`, `linear`, `llama3`, `yarn`, `longrope`, `mrope`

**Exit:** All RoPE tests pass. Existing models (llama, qwen2, phi3) still pass when `rope_scaling` is absent.

### 1B: Extended Cache Types

**Python source:** `mlx-lm/mlx_lm/cache.py` (~350 lines)

**Deliverables** — extend `lib/mlx_lm/models/cache.rb`:
- `_BaseCache` — abstract base with `update_and_fetch`, `state`, `state=`
  - Retrofit existing `KVCache` and `RotatingKVCache` to inherit from it
- `ArraysCache` — stores arbitrary named array state (for SSM hidden state)
  - `update_and_fetch(**named_arrays)` → stores/returns state
- `CacheList` — heterogeneous container: layer `i` can be `KVCache`, `ArraysCache`, or nil
  - Delegates `update_and_fetch` to per-layer cache
  - Required by hybrid models (Jamba, Falcon-H1) that mix attention + SSM layers
- `QuantizedKVCache` — quantizes K/V on store, dequantizes on fetch
  - `initialize(group_size:, bits:)`
- `ChunkedKVCache` — stores K/V in fixed-size chunks to avoid growing concatenation

**Key MLX ops used:** `mx.quantize`, `mx.dequantize`, `mx.concatenate`

**Tests:**
- `ArraysCache` round-trip: store state → retrieve identical state
- `CacheList` with mixed cache types: `[KVCache, ArraysCache, KVCache]` all work
- `QuantizedKVCache` quantize → dequantize round-trip within tolerance
- Existing `make_prompt_cache` / `save_prompt_cache` / `load_prompt_cache` still work

**Exit:** All cache tests pass; existing model inference unchanged.

### 1C: MLA (Multi-head Latent Attention)

**Python source:** `mlx-lm/mlx_lm/models/mla.py` (85 lines)

**Deliverables:**
- `lib/mlx_lm/models/mla.rb`
  - `MultiLinear` — 3D weight `[num_heads, output_dims, input_dims]`, forward via `x @ weight.swapaxes(-1, -2)` or `x @ weight`
  - `QuantizedMultiLinear` — quantized variant using `mx.quantized_matmul`
  - `MultiLinear#to_quantized(group_size:, bits:, mode:)` — conversion method

**Key MLX ops used:** `mx.quantize`, `mx.quantized_matmul`, `mx.random_uniform`

**Tests:**
- `MultiLinear` forward matches Python for DeepSeek-V3 dims (`num_heads=128, input=512, output=128`)
- `QuantizedMultiLinear` matches Python
- `to_quantized` conversion produces identical quantized weights

**Exit:** MLA tests pass.

### 1D: SSM Primitives

**Python source:** `mlx-lm/mlx_lm/models/ssm.py` (252 lines)

**Deliverables:**
- `lib/mlx_lm/models/ssm.rb`
  - `compute_dt(dt, dt_bias, time_step_limit)` — softplus + clip (compiled)
  - `segsum(x, mask:)` — segmented cumulative sum with masking
  - `ssm_attn(x, A_log, B, C, D, dt, dt_bias, state:, ...)` — SSD-SSM forward (chunked attention path)
  - `ssm_update(...)` — dispatcher: uses `ssm_attn` for seq_len > 1, Metal kernel for single-step

**Key MLX ops used:** `mx.fast.metal_kernel`, `mx.compile`, `mx.tril`, `mx.cumsum`, `mx.repeat`, `mx.exp`

**Critical dependency:** `mx.fast.metal_kernel` — the SSM update kernel uses custom Metal shaders. This requires:
1. `MLX::Core::Fast.metal_kernel` exposed in mlx-ruby
2. Template parameters (`T`, constants) passed correctly
3. Grid/threadgroup dispatch working

**Tests:**
- `segsum` matches Python for random input
- `ssm_attn` forward matches Python for Mamba2 config
- `ssm_update` single-step matches Python (Metal path)
- `ssm_update` multi-step matches Python (attention path)

**Exit:** SSM tests pass.

### 1E: Gated Delta, SwitchMLP, PipelineMixin, BitLinear, Activations

**Python sources:**
- `gated_delta.py` (282 lines)
- `switch_layers.py` additions: `SwitchMLP` (35 lines), `QuantizedSwitchLinear` (90 lines)
- `pipeline.py` (31 lines)
- `bitlinear_layers.py` (158 lines)
- `activations.py` (43 lines)

**Deliverables:**

1. **`lib/mlx_lm/models/gated_delta.rb`**
   - Gated delta rule attention kernel + update functions
   - Used by: qwen3_5, qwen3_next, bailing_moe_linear
   - **Critical dependency:** `mx.fast.metal_kernel` (same as SSM)

2. **Extend `lib/mlx_lm/models/switch_layers.rb`:**
   - `SwitchMLP` — like SwitchGLU but uses `fc1` → activation → `fc2` (no gate)
   - `QuantizedSwitchLinear` — uses `mx.gather_qmm` for quantized expert dispatch
   - **Critical dependency:** `mx.gather_qmm` exposed in mlx-ruby

3. **`lib/mlx_lm/models/pipeline.rb`**
   - `PipelineMixin` — mixin for distributed pipeline parallelism
   - `pipeline_layers`, `pipeline(group)` — splits layers across ranks
   - Used by: deepseek_v2, deepseek_v3, glm4_moe, ministral3

4. **`lib/mlx_lm/models/bitlinear_layers.rb`**
   - `BitLinear` — ternary weight layer with packed uint8 storage
   - `bitnet_quantize(model, config)` — replaces `nn.Linear` layers with `BitLinear`
   - **Critical dependency:** `mx.fast.metal_kernel` (custom matmul shader)

5. **`lib/mlx_lm/models/activations.rb`**
   - `XieLU` — parameterized activation (used by step3p5)
   - `swiglu` already inlined in existing code, just needs a shared reference

**Tests:**
- `SwitchMLP` forward matches Python
- `QuantizedSwitchLinear` forward matches Python
- `PipelineMixin` layer slicing correct for various ranks/sizes
- `BitLinear` forward matches Python
- `XieLU` forward matches Python

**Exit:** All infrastructure tests pass.

### Phase 1 — Native Library Gap Report

After completing Phase 1, document:

| Category | Item | Status | Blocker? |
|----------|------|--------|----------|
| **mlx-ruby** | `mx.fast.metal_kernel` (custom Metal shaders) | ? | Blocks SSM, gated_delta, BitLinear |
| **mlx-ruby** | `mx.gather_qmm` (quantized gather matmul) | ? | Blocks QuantizedSwitchLinear |
| **mlx-ruby** | `mx.quantized_matmul` with 3D weight tensors | ? | Blocks QuantizedMultiLinear |
| **mlx-ruby** | `mx.compile` decorator (function compilation) | ? | Performance for SSM, activations |
| **mlx-ruby** | `mx.unflatten` (reshape with inferred dim) | ? | Used by switch_layers, SSM |
| **mlx-ruby** | `mx.expm1` (exp(x)-1) | ? | Used by XieLU |
| **mlx-ruby** | `nn.softplus` | ? | Used by SSM compute_dt |
| **mlx-onnx** | N/A (no new ops needed for infra) | — | — |

---

## Phase 2: Dense Transformer Models

**Objective:** Port all 29 dense transformer architectures that require only
basic RoPE, standard KV cache, and standard attention — no MoE, no SSM.

### Models

| # | File | Lines | Key Differentiator |
|---|------|-------|--------------------|
| 1 | `qwen3.py` | 223 | Q/K norm, `qk_norm` flag |
| 2 | `cohere2.py` | 217 | Sliding window, LayerNorm (not RMSNorm) |
| 3 | `gemma3_text.py` | 257 | Interleaved local/global attn, `clip_residual` |
| 4 | `phi.py` | 174 | Parallel residual (attn + MLP parallel) |
| 5 | `exaone.py` | 165 | Standard transformer |
| 6 | `exaone4.py` | 220 | Updated, Q/K norm |
| 7 | `glm.py` | 188 | GLU variant MLP |
| 8 | `glm4.py` | 181 | GLM4 standard |
| 9 | `granite.py` | 193 | `attention_multiplier`, `logits_scaling` |
| 10 | `internlm3.py` | 238 | Updated InternLM |
| 11 | `olmo.py` | 177 | AI2, no bias |
| 12 | `olmo3.py` | 236 | AI2, Q/K norm |
| 13 | `gpt2.py` | 200 | Conv1D-style weights, LayerNorm |
| 14 | `gpt_bigcode.py` | 185 | Multi-query attention |
| 15 | `minicpm.py` | 204 | Scaled residual |
| 16 | `minicpm3.py` | 251 | KV lora (MLA-like) |
| 17 | `nemotron.py` | 216 | `relu_squared`, LayerNorm1P |
| 18 | `helium.py` | 183 | Standard transformer |
| 19 | `seed_oss.py` | 184 | ByteDance standard |
| 20 | `ernie4_5.py` | 165 | Baidu standard |
| 21 | `baichuan_m1.py` | 251 | Baichuan, custom weight mapping |
| 22 | `nanochat.py` | 233 | Softcap attention |
| 23 | `telechat3.py` | 202 | Standard transformer |
| 24 | `apertus.py` | 188 | Standard transformer |
| 25 | `lille-130m.py` | 155 | Minimal model |
| 26 | `mimo.py` | 194 | Multi-input/multi-output head |
| 27 | `youtu_llm.py` | 239 | Standard transformer |
| 28 | `hunyuan_v1_dense.py` | 231 | DynamicNTK RoPE |
| 29 | `qwen.py` | 157 | Original Qwen v1 |

### VL Wrappers (trivial after text model exists)

- `pixtral.rb` → wraps llama (already done)
- `qwen2_vl.rb` → wraps qwen2 (already done)
- `qwen3_vl.rb` → wraps qwen3
- `gemma3.rb` → wraps gemma3_text

### REMAPPING Entries

- `iquestcoder` → `llama`

### Implementation Pattern

Each model file follows the same pattern:
```ruby
# lib/mlx_lm/models/{model_name}.rb
module MlxLm::Models::{ModelName}
  class ModelArgs < MlxLm::BaseModelArgs
    # fields from Python @dataclass
  end

  class Attention < MLX::NN::Module
    # GQA/MQA with RoPE, KV cache
  end

  class MLP < MLX::NN::Module
    # gate/up/down or fc1/fc2
  end

  class TransformerBlock < MLX::NN::Module
    # attention + mlp + norms
  end

  class Model < MLX::NN::Module
    # embed + blocks + norm + lm_head
    def call(inputs, cache: nil)
    def layers  # for cache creation
    def sanitize(weights)  # weight key remapping
  end
end
```

Register in `lib/mlx_lm/models.rb`:
```ruby
Models.register("model_name", Models::ModelName::Model, Models::ModelName::ModelArgs)
```

### ONNX Validation

For each model, add a `TINY_CONFIGS` entry to `test/parity/onnx_export_test.rb`.
Dense models should achieve 100% ONNX compat and full export.

### Tests

- Forward pass parity: tiny config, random weights → logits match Python `atol=1e-4`
- ONNX compat report: 100% node coverage
- ONNX full export: produces valid `.onnx` file

**Exit:** All 29 models + VL wrappers + alias pass parity and ONNX tests.

### Phase 2 — Native Library Gap Report

| Category | Item | Status | Blocker? |
|----------|------|--------|----------|
| **mlx-ruby** | `nn.LayerNorm` (non-RMSNorm) | ? | Blocks cohere2, gpt2, nemotron |
| **mlx-ruby** | `nn.LayerNorm` with `affine=False` | ? | Blocks some models |
| **mlx-ruby** | `mx.addcmul` / fused ops | ? | Performance only |
| **mlx-ruby** | `mx.clip` with array bounds | ? | Used by granite logits_scaling |
| **mlx-onnx** | Any new unsupported ops in dense models | ? | Track per-model |

---

## Phase 3: MoE Models

**Objective:** Port all 20 Mixture-of-Experts architectures. `SwitchLinear` and
`SwitchGLU` are already ported; `SwitchMLP` added in Phase 1E.

### Models

| # | File | Lines | Routing Style |
|---|------|-------|---------------|
| 1 | `qwen2_moe.py` | 238 | Shared expert + SwitchGLU |
| 2 | `qwen3_moe.py` | 259 | Updated routing + SwitchGLU |
| 3 | `olmoe.py` | 214 | SwitchGLU |
| 4 | `phimoe.py` | 210 | SwitchGLU (Phi-based) |
| 5 | `phixtral.py` | 202 | SwitchGLU (different structure) |
| 6 | `dbrx.py` | 251 | NormAttnNorm + SwitchGLU |
| 7 | `glm4_moe.py` | 403 | SwitchGLU + PipelineMixin |
| 8 | `granitemoe.py` | 235 | TopKGating + SwitchMLP |
| 9 | `exaone_moe.py` | 439 | group_expert_select + SwitchGLU |
| 10 | `llama4.py` | 325 | MoE layers + dense layers |
| 11 | `llama4_text.py` | 182 | Text backbone for llama4 |
| 12 | `hunyuan.py` | 334 | DynamicNTK RoPE + MoE |
| 13 | `ernie4_5_moe.py` | 289 | Baidu MoE |
| 14 | `afmoe.py` | 405 | Apple Foundation MoE |
| 15 | `minimax.py` | 394 | Sharded RMSNorm + MoE |
| 16 | `gpt_oss.py` | 343 | Custom topk + MoE |
| 17 | `dots1.py` | 316 | TopK router + MoE |
| 18 | `step3p5.py` | 505 | ClampedSwiGLU + MoE + XieLU |
| 19 | `Klear.py` | 263 | Standard MoE |
| 20 | `bailing_moe.py` | 401 | group_expert_select |

### Aliases & Wrappers

- `solar_open.rb` → alias for `glm4_moe`
- `phi-msft` → `phixtral` (REMAPPING)
- `minimax_m2` → `minimax` (REMAPPING)
- `llava` → `mistral3` (REMAPPING)
- `qwen3_vl_moe.rb` → wraps qwen3_moe

### ONNX Validation

MoE models should pass ONNX compat at 100% (ArgPartition + GatherMM now
supported in mlx-onnx `33d4b2e`+). Full export may crash due to data-dependent
control flow — track status per model.

### Tests

- Forward pass parity per model: `atol=1e-4`
- Expert routing: same experts selected for same input
- ONNX compat: 100% node coverage

**Exit:** All 20 MoE models + aliases pass parity tests.

### Phase 3 — Native Library Gap Report

| Category | Item | Status | Blocker? |
|----------|------|--------|----------|
| **mlx-ruby** | `mx.gather_qmm` for quantized MoE | ? | Blocks quantized MoE inference |
| **mlx-ruby** | `mx.stop_gradient` | ? | Used in training-aware MoE routing |
| **mlx-onnx** | MoE full export (data-dependent control flow) | ? | Known issue, track workarounds |

---

## Phase 4: DeepSeek-V2/V3 Family

**Objective:** Port the DeepSeek architecture family. These are the most complex
models, requiring MLA (Phase 1C), YaRN RoPE (Phase 1A), switch_layers, and
PipelineMixin (Phase 1E).

### Models

| # | File | Lines | Key Features |
|---|------|-------|-------------|
| 1 | `deepseek_v2.py` | 501 | MLA + YaRN + MoE + shared experts |
| 2 | `deepseek_v3.py` | 553 | Improved MLA + MoE + group_expert_select |
| 3 | `deepseek_v32.py` | 657 | Latest DS, Indexer attention pattern |
| 4 | `glm4_moe_lite.py` | 531 | MLA + MoE + PipelineMixin |
| 5 | `kimi_linear.py` | 611 | MLA + DeltaAttention + ShortConv1d + MoE |
| 6 | `longcat_flash.py` | 493 | MLA + MoE + TopK router |

### Aliases & Wrappers

- `glm_moe_dsa.rb` → alias for `deepseek_v32` (inherits Model)
- `kimi_k25.rb` → wraps deepseek_v3
- `kimi_vl.rb` → wraps deepseek_v3
- `kimi_k2` → `deepseek_v3` (REMAPPING)
- `joyai_llm_flash` → `deepseek_v3` (REMAPPING)

### Implementation Order

1. `deepseek_v2` first (simplest MLA usage)
2. `deepseek_v3` (adds group_expert_select)
3. `deepseek_v32` (adds Indexer)
4. Then `glm4_moe_lite`, `kimi_linear`, `longcat_flash`

### Tests

- Forward pass parity per model
- MLA attention output matches Python
- Expert routing matches Python

**Exit:** All 6 models + aliases pass parity tests.

### Phase 4 — Native Library Gap Report

| Category | Item | Status | Blocker? |
|----------|------|--------|----------|
| **mlx-ruby** | `mx.quantized_matmul` (3D weights) | ? | Blocks QuantizedMultiLinear |
| **mlx-ruby** | Any DeepSeek-specific ops | ? | Track per-model |

---

## Phase 5: SSM & Hybrid Models

**Objective:** Port state-space models and SSM/transformer hybrids. Requires
SSM primitives (Phase 1D) and extended cache (Phase 1B).

### Models

| # | File | Lines | Architecture |
|---|------|-------|-------------|
| 1 | `mamba.py` | 221 | Pure Mamba SSM |
| 2 | `mamba2.py` | 264 | Mamba-2 (SSD) |
| 3 | `jamba.py` | 385 | Mamba + Attention hybrid + MoE |
| 4 | `falcon_h1.py` | 504 | Mamba2 + Attention hybrid |
| 5 | `recurrent_gemma.py` | 452 | RGLRU recurrent (custom RNN, no SSM import) |
| 6 | `rwkv7.py` | 453 | Linear attention / RNN hybrid |
| 7 | `plamo2.py` | 478 | Mamba + Attention hybrid |
| 8 | `nemotron_h.py` | 526 | Mamba2 + Attention + MoE |
| 9 | `granitemoehybrid.py` | 559 | SSM + Attention + MoE |
| 10 | `lfm2.py` / `lfm2_moe.py` | 316/387 | ShortConv + Attention + MoE |

### Aliases & Wrappers

- `lfm2-vl.rb` → wraps lfm2
- `falcon_mamba` → `mamba` (REMAPPING)

### Critical Dependencies

- `mx.fast.metal_kernel` — SSM update kernel uses custom Metal
- `ArraysCache` — SSM state storage
- `CacheList` — mixed cache for hybrid layers

### Tests

- Forward pass parity per model
- SSM state update matches Python (single-step and multi-step)
- Hybrid model: attention layers use KVCache, SSM layers use ArraysCache

**Exit:** All 10+ models pass parity tests.

### Phase 5 — Native Library Gap Report

| Category | Item | Status | Blocker? |
|----------|------|--------|----------|
| **mlx-ruby** | `mx.fast.metal_kernel` | ? | Critical — blocks all SSM models |
| **mlx-ruby** | `mx.scan` (if used by RWKV) | ? | Check RWKV7 implementation |
| **mlx-ruby** | Custom Metal shader dispatch | ? | Template params, grid config |
| **mlx-onnx** | SSM ops (custom kernels) | ? | May need new ONNX mappings |

---

## Phase 6: Gated-Delta & Advanced Hybrid Models

**Objective:** Port models using gated-delta attention. Requires gated_delta
(Phase 1E) and extended cache (Phase 1B).

### Models

| # | File | Lines | Key Feature |
|---|------|-------|-------------|
| 1 | `qwen3_5.py` | 396 | GatedDeltaNet + CacheList |
| 2 | `qwen3_next.py` | 471 | GatedDeltaNet + MoE + CacheList |
| 3 | `bailing_moe_linear.py` | 595 | Recurrent GLA + MoE |
| 4 | `longcat_flash_ngram.py` | 214 | Ngram embeddings, wraps longcat_flash |

### Aliases

- `qwen3_5_moe.rb` → wraps qwen3_5

### Tests

- Forward pass parity per model
- Gated-delta state update matches Python

**Exit:** All 4+ models pass parity tests.

### Phase 6 — Native Library Gap Report

| Category | Item | Status | Blocker? |
|----------|------|--------|----------|
| **mlx-ruby** | `mx.fast.metal_kernel` | ? | Same as Phase 5 |
| **mlx-ruby** | Any gated-delta specific ops | ? | Track during implementation |

---

## Phase 7: Long-Tail Specialized Models

**Objective:** Port remaining models with unique architectures.

### Models

| # | File | Lines | Key Feature |
|---|------|-------|-------------|
| 1 | `afm7.py` | 390 | FusedLoRALinear, KVReuse |
| 2 | `bitnet.py` | 208 | Ternary weights (needs BitLinear from 1E) |
| 3 | `phi3small.py` | 310 | GEGELU activation, blocksparse |
| 4 | `openelm.py` | 220 | Per-layer head/dim scaling |
| 5 | `ministral3.py` | 333 | PipelineMixin, sliding window |
| 6 | `nemotron-nas.py` | 395 | Configurable block types (attn/MLP/NoOp) |
| 7 | `iquestloopcoder.py` | 286 | Loop gate projection |
| 8 | `gemma3n.py` | 613 | AltUp, LaurelBlock, PER (parameter efficient representation) |
| 9 | `smollm3.py` | 75 | NoPE (no positional encoding), extends llama |

### VL Wrappers (from earlier phases)

- `mistral3.rb` → wraps ministral3 / llama
- `kimi_vl.rb` → wraps deepseek_v3

### Tests

- Forward pass parity per model
- BitNet ternary weight handling matches Python
- ONNX compat per model

**Exit:** All 9 models pass parity tests. All REMAPPING entries complete.

### Phase 7 — Native Library Gap Report

| Category | Item | Status | Blocker? |
|----------|------|--------|----------|
| **mlx-ruby** | `mx.fast.metal_kernel` (BitLinear kernel) | ? | Blocks bitnet |
| **mlx-ruby** | `nn.GEGELU` or equivalent | ? | Check phi3small needs |
| **mlx-ruby** | Any model-specific ops | ? | Track per-model |

---

## Phase 8: HuggingFace Hub Integration

**Objective:** Load models by HF repo ID, save/shard, upload.

### 8A: Download & Caching

**Deliverables:**
- `lib/mlx_lm/hub_utils.rb`
  - `snapshot_download(repo_id, revision:, allow_patterns:, local_dir:)`
  - `hf_repo_to_path(repo_id)` → local cache path
  - `cached?(repo_id, revision:)`
  - Auth: `HF_TOKEN` env var or `~/.cache/huggingface/token`
- Update `LoadUtils.load` to accept HF repo IDs

**Implementation:** Direct HTTP to `huggingface.co/api/models/{repo_id}` and
`huggingface.co/{repo_id}/resolve/{revision}/{filename}`. Ruby `net/http`.

**Tests:**
- Download small model → loads correctly
- Cached download → no network call
- `load("mlx-community/SmolLM2-135M-Instruct-4bit")` end-to-end

### 8B: Model Saving & Sharding

**Deliverables:**
- `WeightUtils.save_safetensors(path, weights)`
- `WeightUtils.make_shards(weights, max_file_size_gb:)`
- `LoadUtils.save_model(path, model:, tokenizer:, config:)`

**Tests:**
- Save → load round-trip identical
- Sharding splits at boundary

### 8C: Upload & Cache Management

**Deliverables:**
- `HubUtils.upload_to_hub(path, repo_id:, token:)`
- `HubUtils.create_model_card(path, hf_path:)`
- `Manage` module — list/delete cached models
- CLI: `mlx_lm manage` subcommand

**Tests:**
- Upload workflow
- Cache list/delete

**Exit:** Full HF Hub integration working.

### Phase 8 — Native Library Gap Report

| Category | Item | Status | Blocker? |
|----------|------|--------|----------|
| **mlx-ruby** | `mx.save_safetensors` (writing) | ? | Blocks model saving |
| **mlx-ruby** | Any serialization gaps | ? | Track during implementation |

---

## Phase 9: ONNX Export Validation

**Objective:** ONNX compat and export tests for all ~109 model architectures.

**Prerequisite:** mlx-onnx submodule updated to `33d4b2e`+ (ArgPartition + GatherMM).

### 9A: Verify Existing Models

- Rebuild mlx-onnx
- All 13 existing models: 100% compat
- Dense models: full `.onnx` export

### 9B: Dense Model ONNX Tests (Phase 2 + 7)

- Add `TINY_CONFIGS` entries for all 38 dense models
- All should pass compat + full export

### 9C: MoE Model ONNX Tests (Phase 3 + 4 + 6)

- Add `TINY_CONFIGS` for all 30 MoE models
- Compat: 100% (ArgPartition + GatherMM supported)
- Full export: track crash status per model

### 9D: SSM Model ONNX Tests (Phase 5)

- Add `TINY_CONFIGS` for 10 SSM/hybrid models
- Document SSM-specific op gaps

### 9E: VL Wrappers & Aliases

- TINY_CONFIGS for all VL wrappers
- Verify REMAPPING works with ONNX tests

**Exit:** All ~109 architectures have ONNX compat test coverage.

### Phase 9 — Native Library Gap Report

| Category | Item | Status | Blocker? |
|----------|------|--------|----------|
| **mlx-onnx** | SSM Metal kernel ops | ? | May lack ONNX equivalents |
| **mlx-onnx** | Gated-delta Metal kernel ops | ? | May lack ONNX equivalents |
| **mlx-onnx** | BitLinear Metal kernel ops | ? | May lack ONNX equivalents |
| **mlx-onnx** | MoE data-dependent control flow | ? | Known crash, track upstream |

---

## Native Library Gap Report Template

At the end of each phase, fill in and commit this report:

```markdown
### Phase N — Native Library Gap Report (Completed YYYY-MM-DD)

**Models attempted:** X
**Models fully working:** Y
**Models blocked:** Z

#### Blockers

| Library | Missing Feature | Affected Models | Upstream Issue | Workaround |
|---------|----------------|-----------------|----------------|------------|
| mlx-ruby | ... | ... | #NNN | ... |

#### Non-blocking Issues

| Library | Issue | Impact | Notes |
|---------|-------|--------|-------|
| ... | ... | ... | ... |

#### Resolved (from previous reports)

| Library | Feature | Resolution |
|---------|---------|------------|
| ... | ... | Fixed in commit/version |
```

---

## Execution Order & Dependencies

```
Phase 1A (RoPE) ─────────────────────────────────┐
Phase 1B (Cache) ────────────────────────────┐    │
Phase 1C (MLA) ─────────────────────────┐    │    │
Phase 1D (SSM) ────────────────────┐    │    │    │
Phase 1E (GatedDelta, etc.) ──┐    │    │    │    │
                               │    │    │    │    │
                               ▼    ▼    ▼    ▼    ▼
Phase 2 (Dense) ◄──────────────────────────────────┘
Phase 3 (MoE) ◄───────────────────────────────┘
Phase 4 (DeepSeek) ◄──────────────────────────┘
Phase 5 (SSM) ◄───────────────────────────────┘
Phase 6 (GatedDelta) ◄────────────────────────┘
Phase 7 (LongTail) ◄──────────────────────────┘
Phase 8 (HF Hub) — independent, can run in parallel
Phase 9 (ONNX) — runs after each model phase
```

Phase 1 sub-phases can run in parallel. Model phases (2-7) depend on Phase 1
completion but are otherwise independent of each other. Phase 8 has no model
dependencies. Phase 9 ONNX validation runs incrementally after each model phase.

---

## Model Count Summary

| Phase | New Models | Cumulative |
|-------|-----------|------------|
| Existing | 13 | 13 |
| Phase 2 | 29 + 4 wrappers | 46 |
| Phase 3 | 20 + 4 aliases | 70 |
| Phase 4 | 6 + 5 aliases | 81 |
| Phase 5 | 10 + 2 aliases | 93 |
| Phase 6 | 4 + 1 alias | 98 |
| Phase 7 | 9 + 2 wrappers | 109 |
| **Total** | **~96 new** | **~109** |
