# Python mlx-lm → Ruby mlx-ruby-lm Parity Checklist

**Status:** Proposed
**Date:** 2026-02-25

## Context

This document provides a comprehensive class-by-class, module-by-module comparison
of the upstream Python `mlx-lm` (v0.30.7) against the Ruby `mlx-ruby-lm` (v0.30.7.1)
port. It identifies every component that has been fully mirrored, partially mirrored,
or is still missing.

The upstream Python codebase contains **~130 source files** with **~115 model
architectures**, extensive tooling (tuner, quantization, server, CLI, evaluation),
and shared infrastructure (cache, RoPE, MLA, SSM, switch layers). The Ruby port
currently implements **33 source files** covering the core inference pipeline and
**13 model architectures**.

## Goals

- Provide a single authoritative source of truth for what remains to build
- Enable prioritization of the next implementation phases
- Track progress as parity work continues

## Non-goals

- Rewriting anything already working — this is a gap analysis, not a refactor plan
- Specifying implementation details for each missing component

---

## 1. Top-Level Public API

| Python | Ruby | Status |
|--------|------|--------|
| `mlx_lm.load()` | `MlxLm::LoadUtils.load` | Done |
| `mlx_lm.generate()` | `MlxLm::Generate.generate` | Done |
| `mlx_lm.stream_generate()` | `MlxLm::Generate.stream_generate` | Done |
| `mlx_lm.batch_generate()` | — | Missing |
| `mlx_lm.convert()` | — | Missing (partial: `ConvertUtils` has dtype/param helpers) |

---

## 2. Core Modules

### 2.1 Configuration & Weight Loading

| Python Component | Ruby Equivalent | Status |
|-----------------|-----------------|--------|
| `BaseModelArgs` (base.py) | `MlxLm::BaseModelArgs` | Done |
| `create_causal_mask()` (base.py) | Inline in model files | Done (inlined) |
| `create_attention_mask()` (base.py) | Inline in model files | Done (inlined) |
| `scaled_dot_product_attention()` (base.py) | Uses `MLX::Core.fast.scaled_dot_product_attention` | Done |
| `quantized_scaled_dot_product_attention()` (base.py) | — | Missing |
| `create_ssm_mask()` (base.py) | — | Missing (needed for Mamba/Jamba) |
| Config loading (`utils.load_config`) | `MlxLm::Config.load` | Done |
| Safetensors loading (`utils.load_model`) | `MlxLm::LoadUtils.load_model` | Done |
| Weight sharding (`utils.make_shards`) | — | Missing |
| `tree_unflatten` | `MlxLm::WeightUtils.tree_unflatten` | Done |
| HF Hub download (`utils._download`) | — | Missing (requires local path) |
| `hf_repo_to_path()` | — | Missing |

### 2.2 Tokenizer

| Python Component | Ruby Equivalent | Status |
|-----------------|-----------------|--------|
| `TokenizerWrapper` | `MlxLm::TokenizerWrapper` | Done |
| `StreamingDetokenizer` (base class) | `MlxLm::StreamingDetokenizer` | Done |
| `NaiveStreamingDetokenizer` | — | Missing (single implementation in Ruby) |
| `SPMStreamingDetokenizer` | — | Missing |
| `BPEStreamingDetokenizer` | — | Missing |
| `NewlineTokenizer` | — | Missing |
| `load()` (full HF tokenizer loading) | Simplified in `LoadUtils.load_tokenizer` | Partial |
| `no_bos_or_eos()` | — | Missing |
| `_is_spm_decoder()` / `_is_bpe_decoder()` | — | Missing |
| `_infer_tool_parser()` | — | Missing |

### 2.3 Generation

| Python Component | Ruby Equivalent | Status |
|-----------------|-----------------|--------|
| `generate_step()` | `MlxLm::Generate.generate_step` | Done |
| `generate()` | `MlxLm::Generate.generate` | Done |
| `stream_generate()` | `MlxLm::Generate.stream_generate` | Done |
| `GenerationResponse` | `MlxLm::GenerationResponse` | Done |
| `speculative_generate_step()` | — | Missing |
| `batch_generate()` | — | Missing |
| `BatchGenerator` class | — | Missing |
| `BatchStats` / `BatchResponse` / `Batch` | — | Missing |
| `wired_limit()` (memory management) | — | Missing |
| `maybe_quantize_kv_cache()` | — | Missing |
| `_left_pad_prompts()` / `_right_pad_prompts()` | — | Missing |

### 2.4 Sampling

| Python Component | Ruby Equivalent | Status |
|-----------------|-----------------|--------|
| `make_sampler()` | `MlxLm::SampleUtils.make_sampler` | Done |
| `make_logits_processors()` | `MlxLm::SampleUtils.make_logits_processors` | Done |
| `apply_top_k()` | `MlxLm::SampleUtils.apply_top_k` | Done |
| `apply_top_p()` | `MlxLm::SampleUtils.apply_top_p` | Done |
| `apply_min_p()` | `MlxLm::SampleUtils.apply_min_p` | Done |
| `categorical_sampling()` | `MlxLm::SampleUtils.categorical_sampling` | Done |
| `make_repetition_penalty()` | `MlxLm::SampleUtils.make_repetition_penalty` | Done |
| `apply_xtc()` (XTC sampling) | — | Missing |

### 2.5 KV Cache

| Python Component | Ruby Equivalent | Status |
|-----------------|-----------------|--------|
| `_BaseCache` | — | Missing (no shared base class) |
| `ConcatenateKVCache` | — | Missing |
| `KVCache` | `MlxLm::KVCache` | Done |
| `RotatingKVCache` | `MlxLm::RotatingKVCache` | Done |
| `QuantizedKVCache` | — | Missing |
| `ArraysCache` (for Mamba/SSM) | — | Missing |
| `ChunkedKVCache` | — | Missing |
| `CacheList` | — | Missing (Ruby uses plain arrays) |
| `BatchKVCache` | — | Missing |
| `BatchRotatingKVCache` | — | Missing |
| `make_prompt_cache()` | `MlxLm::Cache.make_prompt_cache` | Done |
| `save_prompt_cache()` | `MlxLm::Cache.save_prompt_cache` | Done |
| `load_prompt_cache()` | `MlxLm::Cache.load_prompt_cache` | Done |
| `can_trim_prompt_cache()` | — | Missing |
| `trim_prompt_cache()` | — | Missing |
| `dynamic_roll()` | — | Missing |

---

## 3. Model Architectures

### 3.1 Implemented (13 architectures)

| # | Python Model | Ruby Module | Registry Key | Notes |
|---|-------------|-------------|--------------|-------|
| 1 | `llama.py` | `MlxLm::Models::Llama` | `"llama"` | Done — also handles `"mistral"` via alias |
| 2 | `gemma.py` | `MlxLm::Models::Gemma` | `"gemma"` | Done |
| 3 | `gemma2.py` | `MlxLm::Models::Gemma2` | `"gemma2"` | Done |
| 4 | `qwen2.py` | `MlxLm::Models::Qwen2` | `"qwen2"` | Done |
| 5 | `phi3.py` | `MlxLm::Models::Phi3` | `"phi3"` | Done |
| 6 | `starcoder2.py` | `MlxLm::Models::Starcoder2` | `"starcoder2"` | Done |
| 7 | `stablelm.py` | `MlxLm::Models::StableLM` | `"stablelm"` | Done |
| 8 | `cohere.py` | `MlxLm::Models::Cohere` | `"cohere"` | Done |
| 9 | `olmo2.py` | `MlxLm::Models::OLMo2` | `"olmo2"` | Done |
| 10 | `gpt_neox.py` | `MlxLm::Models::GPTNeoX` | `"gpt_neox"` | Done |
| 11 | `mixtral.py` | `MlxLm::Models::Mixtral` | `"mixtral"` | Done (MoE via SwitchGLU) |
| 12 | `deepseek.py` | `MlxLm::Models::DeepSeek` | `"deepseek"` | Done (MoE + shared experts) |
| 13 | `internlm2.py` | `MlxLm::Models::InternLM2` | `"internlm2"` | Done |

### 3.2 Missing Model Architectures (100+ models)

#### Tier 1 — High Priority (popular, widely used)

| # | Python File | Model | Key Feature | Depends On |
|---|------------|-------|-------------|------------|
| 14 | `qwen3.py` | Qwen3 | Latest Qwen | — |
| 15 | `qwen3_moe.py` | Qwen3-MoE | MoE variant | switch_layers |
| 16 | `qwen3_5.py` | Qwen3.5 | Latest Qwen | — |
| 17 | `qwen3_5_moe.py` | Qwen3.5-MoE | MoE variant | switch_layers |
| 18 | `gemma3.py` | Gemma3 | Interleaved local/global attn | — |
| 19 | `gemma3_text.py` | Gemma3-Text | Text backbone | — |
| 20 | `llama4.py` | Llama4 | Meta's latest | — |
| 21 | `llama4_text.py` | Llama4-Text | Text backbone | — |
| 22 | `deepseek_v2.py` | DeepSeek-V2 | MLA attention | mla.py |
| 23 | `deepseek_v3.py` | DeepSeek-V3 | Improved MLA + MoE | mla.py |
| 24 | `deepseek_v32.py` | DeepSeek-V3.2 | Latest DeepSeek | mla.py |
| 25 | `mistral3.py` | Mistral3 | Latest Mistral | — |
| 26 | `phi.py` | Phi (original) | MS Phi | — |
| 27 | `cohere2.py` | Cohere2 | Updated Cohere | — |

#### Tier 2 — Medium Priority (significant user base)

| # | Python File | Model | Key Feature | Depends On |
|---|------------|-------|-------------|------------|
| 28 | `granite.py` | Granite | IBM model | — |
| 29 | `granitemoe.py` | GraniteMoE | IBM MoE model | switch_layers |
| 30 | `granitemoehybrid.py` | GraniteMoE-Hybrid | Hybrid MoE+SSM | ssm.py, switch_layers |
| 31 | `olmo.py` | OLMo | AI2 original | — |
| 32 | `olmo3.py` | OLMo3 | AI2 latest | — |
| 33 | `olmoe.py` | OLMoE | AI2 MoE | switch_layers |
| 34 | `exaone.py` | EXAONE | LG AI Research | — |
| 35 | `exaone4.py` | EXAONE4 | Updated | — |
| 36 | `exaone_moe.py` | EXAONE-MoE | MoE variant | switch_layers |
| 37 | `internlm3.py` | InternLM3 | Updated InternLM | — |
| 38 | `glm.py` | GLM | Zhipu AI | — |
| 39 | `glm4.py` | GLM4 | Zhipu AI latest | — |
| 40 | `glm4_moe.py` | GLM4-MoE | MoE variant | switch_layers |
| 41 | `glm4_moe_lite.py` | GLM4-MoE-Lite | Lightweight MoE | switch_layers |
| 42 | `glm_moe_dsa.py` | GLM-MoE-DSA | DSA attention MoE | switch_layers |
| 43 | `minimax.py` | MiniMax | MiniMax model | — |
| 44 | `minicpm.py` | MiniCPM | OpenBMB | — |
| 45 | `minicpm3.py` | MiniCPM3 | OpenBMB latest | — |
| 46 | `smollm3.py` | SmolLM3 | HuggingFace | — |
| 47 | `gpt2.py` | GPT-2 | OpenAI classic | — |
| 48 | `gpt_bigcode.py` | GPT-BigCode | StarCoder1 base | — |

#### Tier 3 — Specialized / SSM / Hybrid

| # | Python File | Model | Key Feature | Depends On |
|---|------------|-------|-------------|------------|
| 49 | `mamba.py` | Mamba | SSM (state-space) | ssm.py |
| 50 | `mamba2.py` | Mamba2 | Improved SSM | ssm.py |
| 51 | `jamba.py` | Jamba | SSM + Transformer hybrid | ssm.py |
| 52 | `falcon_h1.py` | Falcon-H1 | Hybrid Mamba+Attn | ssm.py |
| 53 | `gated_delta.py` | Gated Delta | Delta-net style | ssm.py |
| 54 | `nemotron_h.py` | Nemotron-H | NVIDIA hybrid | ssm.py |
| 55 | `recurrent_gemma.py` | RecurrentGemma | Google recurrent | — |
| 56 | `rwkv7.py` | RWKV7 | Linear attention / RNN | — |

#### Tier 4 — Long Tail

| # | Python File | Model | Key Feature | Depends On |
|---|------------|-------|-------------|------------|
| 57 | `afm7.py` | AFM7 | Apple Foundation Model | — |
| 58 | `afmoe.py` | AFMoE | Apple MoE | switch_layers |
| 59 | `apertus.py` | Apertus | — | — |
| 60 | `baichuan_m1.py` | Baichuan-M1 | Baichuan | — |
| 61 | `bailing_moe.py` | Bailing-MoE | Bailing | bailing_moe_linear.py |
| 62 | `bailing_moe_linear.py` | Bailing-MoE-Linear | Shared layer | — |
| 63 | `bitnet.py` | BitNet | Ternary weights | bitlinear_layers.py |
| 64 | `dbrx.py` | DBRX | Databricks MoE | switch_layers |
| 65 | `dots1.py` | Dots1 | — | — |
| 66 | `ernie4_5.py` | ERNIE 4.5 | Baidu | — |
| 67 | `ernie4_5_moe.py` | ERNIE 4.5 MoE | Baidu MoE | switch_layers |
| 68 | `gemma3n.py` | Gemma3n | Nano variant | — |
| 69 | `gpt_oss.py` | GPT-OSS | — | — |
| 70 | `helium.py` | Helium | — | — |
| 71 | `hunyuan.py` | Hunyuan | Tencent | — |
| 72 | `hunyuan_v1_dense.py` | Hunyuan-V1-Dense | Tencent dense | — |
| 73 | `iquestloopcoder.py` | iQuestLoopCoder | — | — |
| 74 | `Klear.py` | Klear | — | — |
| 75 | `kimi_k25.py` | Kimi-K25 | Moonshot | kimi_linear.py |
| 76 | `kimi_linear.py` | Kimi-Linear | Shared layer | — |
| 77 | `kimi_vl.py` | Kimi-VL | Moonshot VL | — |
| 78 | `lfm2.py` | LFM2 | Liquid Foundation | — |
| 79 | `lfm2-vl.py` | LFM2-VL | Liquid VL | — |
| 80 | `lfm2_moe.py` | LFM2-MoE | Liquid MoE | switch_layers |
| 81 | `lille-130m.py` | Lille-130M | Small model | — |
| 82 | `longcat_flash.py` | LongCat-Flash | — | — |
| 83 | `longcat_flash_ngram.py` | LongCat-Flash-Ngram | — | — |
| 84 | `mimo.py` | MIMO | Multi-in/multi-out | — |
| 85 | `mimo_v2_flash.py` | MIMO-V2-Flash | — | — |
| 86 | `ministral3.py` | Ministral3 | Small Mistral | — |
| 87 | `nanochat.py` | NanoChat | — | — |
| 88 | `nemotron.py` | Nemotron | NVIDIA | — |
| 89 | `nemotron-nas.py` | Nemotron-NAS | NVIDIA NAS | — |
| 90 | `openelm.py` | OpenELM | Apple | — |
| 91 | `phi3small.py` | Phi3-Small | MS small variant | — |
| 92 | `phimoe.py` | PhiMoE | MS MoE | switch_layers |
| 93 | `phixtral.py` | Phixtral | Community MoE | switch_layers |
| 94 | `pixtral.py` | Pixtral | Mistral VL | — |
| 95 | `plamo.py` | PLaMo | Preferred Networks | — |
| 96 | `plamo2.py` | PLaMo2 | Preferred Networks | — |
| 97 | `qwen.py` | Qwen (v1) | Original Qwen | — |
| 98 | `qwen2_moe.py` | Qwen2-MoE | MoE variant | switch_layers |
| 99 | `qwen2_vl.py` | Qwen2-VL | Vision-language | — |
| 100 | `qwen3_next.py` | Qwen3-Next | Experimental | — |
| 101 | `qwen3_vl.py` | Qwen3-VL | Vision-language | — |
| 102 | `qwen3_vl_moe.py` | Qwen3-VL-MoE | VL + MoE | switch_layers |
| 103 | `seed_oss.py` | Seed-OSS | ByteDance | — |
| 104 | `solar_open.py` | Solar-Open | Upstage | — |
| 105 | `step3p5.py` | Step3.5 | — | — |
| 106 | `telechat3.py` | Telechat3 | — | — |
| 107 | `youtu_llm.py` | Youtu-LLM | — | — |

---

## 4. Shared Model Infrastructure

| Python Component | Ruby Equivalent | Status |
|-----------------|-----------------|--------|
| `rope_utils.py` — `SuScaledRoPE` | — | Missing |
| `rope_utils.py` — `Llama3RoPE` | — | Missing |
| `rope_utils.py` — `YarnRoPE` | — | Missing |
| `rope_utils.py` — `initialize_rope()` | Inline in model files | Partial (basic RoPE only) |
| `switch_layers.py` — `SwitchLinear` | `MlxLm::Models::SwitchLayers::SwitchLinear` | Done |
| `switch_layers.py` — `SwitchGLU` | `MlxLm::Models::SwitchLayers::SwitchGLU` | Done |
| `switch_layers.py` — `SwitchMLP` | — | Missing |
| `switch_layers.py` — `QuantizedSwitchLinear` | — | Missing |
| `switch_layers.py` — `SwiGLU` | — | Missing (standalone version) |
| `activations.py` — `swiglu()` | Inline in MLP classes | Done (inlined) |
| `activations.py` — `XieLU` | — | Missing |
| `mla.py` — `MultiLinear` | — | Missing (needed for DeepSeek-V2/V3) |
| `mla.py` — `QuantizedMultiLinear` | — | Missing |
| `ssm.py` — `ssm_update()` / `ssm_attn()` | — | Missing (needed for Mamba/Jamba) |
| `ssm.py` — `compute_dt()` / `segsum()` | — | Missing |
| `bitlinear_layers.py` — `BitLinear` | — | Missing (needed for BitNet) |
| `pipeline.py` — `PipelineMixin` | — | Missing |

---

## 5. Quantization

| Python Component | Ruby Equivalent | Status |
|-----------------|-----------------|--------|
| `utils.quantize_model()` | `MlxLm::Quantize.quantize_model` | Done |
| `utils.dequantize_model()` | `MlxLm::Quantize.dequantize_model` | Done |
| `utils.compute_bits_per_weight()` | `MlxLm::Quantize.bits_per_weight` | Done |
| `convert.convert()` (full pipeline) | — | Missing |
| `convert.mixed_quant_predicate_builder()` | — | Missing |
| `quant/awq.py` — `awq_quantize()` | — | Missing |
| `quant/awq.py` — `AWQConfig` / `ScaleConfig` | — | Missing |
| `quant/gptq.py` — `gptq_quantize()` | — | Missing |
| `quant/dwq.py` — `dwq_quantize()` | — | Missing |
| `quant/dynamic_quant.py` — `estimate_sensitivities()` | — | Missing |

---

## 6. Tuner / Fine-Tuning

| Python Component | Ruby Equivalent | Status |
|-----------------|-----------------|--------|
| `tuner/lora.py` — `LoRALinear` | `MlxLm::Tuner::LoRALinear` | Done |
| `tuner/lora.py` — `LoRAEmbedding` | `MlxLm::Tuner::LoRAEmbedding` | Done |
| `tuner/lora.py` — `LoRASwitchLinear` | — | Missing |
| `tuner/dora.py` — `DoRALinear` | — | Missing |
| `tuner/dora.py` — `DoRAEmbedding` | — | Missing |
| `tuner/trainer.py` — `TrainingArgs` | — | Missing |
| `tuner/trainer.py` — `train()` | — | Missing |
| `tuner/trainer.py` — `evaluate()` | — | Missing |
| `tuner/trainer.py` — `default_loss()` | — | Missing |
| `tuner/trainer.py` — `iterate_batches()` | — | Missing |
| `tuner/trainer.py` — `grad_checkpoint()` | — | Missing |
| `tuner/datasets.py` — `TextDataset` | — | Missing |
| `tuner/datasets.py` — `ChatDataset` | — | Missing |
| `tuner/datasets.py` — `CompletionsDataset` | — | Missing |
| `tuner/datasets.py` — `ConcatenatedDataset` | — | Missing |
| `tuner/datasets.py` — `CacheDataset` | — | Missing |
| `tuner/datasets.py` — `load_dataset()` | — | Missing |
| `tuner/callbacks.py` — `TrainingCallback` | — | Missing |
| `tuner/callbacks.py` — `WandBCallback` | — | Missing |
| `tuner/utils.py` — `linear_to_lora_layers()` | `MlxLm::Tuner.apply_lora_layers` | Done |
| `tuner/utils.py` — `load_adapters()` | — | Missing |
| `tuner/utils.py` — `remove_lora_layers()` | — | Missing |
| `tuner/utils.py` — `print_trainable_parameters()` | — | Missing |
| `tuner/utils.py` — `build_schedule()` | — | Missing |
| `tuner/losses.py` — `kl_div_loss()` | — | Missing |
| `tuner/losses.py` — `js_div_loss()` | — | Missing |

---

## 7. CLI & Server

| Python Component | Ruby Equivalent | Status |
|-----------------|-----------------|--------|
| `cli.py` — `main()` dispatcher | `MlxLm::CLI.run` | Partial (3 commands only) |
| CLI: `generate` | `MlxLm::CLI.run_generate` | Done |
| CLI: `chat` | `MlxLm::CLI.run_chat` | Done |
| CLI: `server` | `MlxLm::CLI.run_server` | Done |
| CLI: `convert` | — | Missing |
| CLI: `lora` | — | Missing |
| CLI: `fuse` | — | Missing |
| CLI: `benchmark` | — | Missing (module exists, no CLI binding) |
| CLI: `evaluate` | — | Missing |
| CLI: `manage` | — | Missing |
| `server.py` — `APIHandler` (full HTTP handler) | `MlxLm::Server.start` (WEBrick) | Partial |
| `server.py` — `ModelProvider` | — | Missing |
| `server.py` — `ResponseGenerator` | — | Missing |
| `server.py` — `LRUPromptCache` | — | Missing |
| `server.py` — `/v1/completions` endpoint | — | Missing |
| `server.py` — `/v1/chat/completions` endpoint | `MlxLm::Server` | Done |
| `server.py` — `/v1/models` endpoint | `MlxLm::Server` | Done |
| `server.py` — Stop condition handling | — | Missing |
| `server.py` — Tool call support | — | Missing |

---

## 8. Tool Parsers

| Python Component | Ruby Equivalent | Status |
|-----------------|-----------------|--------|
| `tool_parsers/json_tools.py` | — | Missing |
| `tool_parsers/pythonic.py` | — | Missing |
| `tool_parsers/mistral.py` | — | Missing |
| `tool_parsers/function_gemma.py` | — | Missing |
| `tool_parsers/glm47.py` | — | Missing |
| `tool_parsers/kimi_k2.py` | — | Missing |
| `tool_parsers/longcat.py` | — | Missing |
| `tool_parsers/minimax_m2.py` | — | Missing |
| `tool_parsers/qwen3_coder.py` | — | Missing |

Entire subsystem missing — 0/9 parsers implemented.

---

## 9. Chat Templates

| Python Component | Ruby Equivalent | Status |
|-----------------|-----------------|--------|
| `chat_templates/deepseek_v32.py` | — | Missing |
| Basic ChatML formatting | `MlxLm::ChatTemplate` | Done (default + chatml only) |

---

## 10. Standalone Tools & Utilities

| Python Component | Ruby Equivalent | Status |
|-----------------|-----------------|--------|
| `benchmark.py` | `MlxLm::Benchmark` | Done (module, no CLI) |
| `perplexity.py` | `MlxLm::Perplexity` | Done |
| `cache_prompt.py` | — | Missing (CLI for prompt caching) |
| `chat.py` | Inline in `CLI.run_chat` | Done (inlined) |
| `evaluate.py` — `MLXLM` (lm-eval harness) | — | Missing |
| `fuse.py` — adapter fusion pipeline | — | Missing |
| `gguf.py` — GGUF export | — | Missing |
| `lora.py` — LoRA training entry point | — | Missing |
| `manage.py` — cache management | — | Missing |
| `share.py` — model sharing server | — | Missing |
| `upload.py` — HF Hub upload | — | Missing |

---

## 11. Summary Scorecard

### By Category

| Category | Implemented | Total | Coverage |
|----------|------------|-------|----------|
| Top-level API | 3 | 5 | 60% |
| Config & Weights | 5 | 12 | 42% |
| Tokenizer | 3 | 11 | 27% |
| Generation | 3 | 11 | 27% |
| Sampling | 7 | 8 | 88% |
| KV Cache | 5 | 16 | 31% |
| Model Architectures | 13 | 107 | 12% |
| Shared Infrastructure | 3 | 17 | 18% |
| Quantization | 3 | 10 | 30% |
| Tuner / Fine-Tuning | 3 | 26 | 12% |
| CLI Commands | 3 | 10 | 30% |
| Server | 2 | 8 | 25% |
| Tool Parsers | 0 | 9 | 0% |
| Chat Templates | 1 | 2 | 50% |
| Standalone Tools | 2 | 11 | 18% |

### Overall

| Metric | Count |
|--------|-------|
| **Components fully implemented** | ~56 |
| **Components partially implemented** | ~5 |
| **Components missing** | ~100+ |
| **Model architectures implemented** | 13 / 107 (12%) |
| **Core inference pipeline** | Functional for 13 architectures |

### What Works Today

The Ruby port has a **complete inference pipeline** for 13 model architectures:
- Load model + tokenizer from local path
- Generate text (streaming and non-streaming)
- All major sampling strategies (top-k, top-p, min-p, temperature, repetition penalty)
- KV cache (simple + rotating)
- Basic quantization (quantize/dequantize model layers)
- LoRA layer application (LoRALinear, LoRAEmbedding)
- OpenAI-compatible chat server (basic)
- Perplexity evaluation
- Benchmarking

### Biggest Gaps (by impact)

1. **94 missing model architectures** — particularly Qwen3, Gemma3, Llama4, DeepSeek-V2/V3
2. **Training pipeline** — no trainer, datasets, loss functions, or callbacks
3. **Batch generation** — no BatchGenerator or batch inference
4. **HuggingFace Hub integration** — no download, upload, or cache management
5. **Advanced quantization** — no AWQ, GPTQ, DWQ, or mixed quantization
6. **Speculative decoding** — no draft model support
7. **Tool calling** — no tool parsers for function calling
8. **Shared model infrastructure** — missing RoPE variants, MLA, SSM primitives
9. **Evaluation harness** — no lm-eval integration
10. **GGUF export** — no GGUF support

---

## 12. Phased Implementation Plan

### Scope

This plan covers four areas:
1. Shared model infrastructure (RoPE variants, MLA, SSM, extended cache, etc.)
2. All missing model architectures (~94 models, grouped by dependency/complexity)
3. HuggingFace Hub integration (download, cache, upload)
4. ONNX export support for all model architectures

**Out of scope:** Training pipeline, batch generation, advanced quantization (AWQ/GPTQ/DWQ),
speculative decoding, tool parsers, evaluation harness, GGUF export.

### Dependency Map

Before models can be ported, they need their shared infrastructure. Here's what
each infrastructure module unlocks:

| Infrastructure | Models Unlocked (not yet ported) |
|---------------|----------------------------------|
| `rope_utils` (SuScaledRoPE, Llama3RoPE, YarnRoPE, initialize_rope) | 43 models use it (most already work via basic RoPE; advanced scaling needed for: gemma3_text, exaone, exaone4, ministral3, minicpm, minicpm3, deepseek_v3, deepseek_v32, etc.) |
| `mla` (MultiLinear, QuantizedMultiLinear) | deepseek_v3, deepseek_v32, glm4_moe_lite, kimi_linear, longcat_flash |
| `ssm` (ssm_update, ssm_attn, segsum) | mamba2, falcon_h1, granitemoehybrid, nemotron_h, plamo2 |
| `switch_layers` additions (SwitchMLP, QuantizedSwitchLinear) | 33 MoE models (SwitchLinear/SwitchGLU already ported) |
| `gated_delta` (gated_delta_kernel, gated_delta_update) | kimi_linear, qwen3_5, qwen3_next |
| `bitlinear_layers` (BitLinear) | bitnet |
| Extended cache (ArraysCache, CacheList, ChunkedKVCache) | mamba, mamba2, jamba, falcon_h1, lfm2, lfm2_moe, rwkv7, recurrent_gemma, plamo2, qwen3_5, qwen3_next, etc. |
| `pipeline` (PipelineMixin) | deepseek_v2, deepseek_v3, glm4_moe, glm4_moe_lite, ministral3 |

### Model Classification

**VL Wrappers** (thin ~50-line files delegating to a text model — port as trivial
wrappers once the text model exists):
- gemma3 → gemma3_text
- mistral3 → ministral3 / llama
- pixtral → llama
- lfm2-vl → lfm2
- qwen2_vl → qwen2
- qwen3_vl → qwen3
- qwen3_vl_moe → qwen3_moe (or qwen3_next)
- kimi_vl → deepseek_v3

**Pure Aliases** (re-export Model from another module):
- solar_open → glm4_moe
- glm_moe_dsa → deepseek_v32
- qwen3_5_moe → qwen3_5
- longcat_flash_ngram → longcat_flash
- kimi_k25 → deepseek_v3

**Remapped in MODEL_REMAPPING** (no new code needed, just add to Ruby REMAPPING):
- mistral → llama (already done)
- llava → mistral3
- phi-msft → phixtral
- falcon_mamba → mamba
- joyai_llm_flash → deepseek_v3
- kimi_k2 → deepseek_v3
- qwen2_5_vl → qwen2_vl
- minimax_m2 → minimax
- iquestcoder → llama

---

### Phase 1: Shared Model Infrastructure

**Objective:** Build all shared modules needed by downstream model phases.

#### Phase 1A: RoPE Variants & Base Utilities

**Red:**
- Test `SuScaledRoPE` produces identical embeddings to Python for Phi-3 long-context config
- Test `Llama3RoPE` matches Python for Llama-3 frequency correction
- Test `YarnRoPE` matches Python for DeepSeek-V2 YaRN config
- Test `initialize_rope()` factory returns correct class for each `rope_scaling` type

**Green:**
- [ ] `lib/mlx_lm/models/rope_utils.rb` — `SuScaledRoPE`, `Llama3RoPE`, `YarnRoPE`, `initialize_rope()`
- [ ] Update existing model files (llama, qwen2, phi3) to use `initialize_rope()` when `rope_scaling` is present

**Exit:** All RoPE variant tests pass; existing model tests still pass.

#### Phase 1B: Extended Cache Types

**Red:**
- Test `ArraysCache` stores/retrieves arbitrary array state (for SSM)
- Test `CacheList` composes heterogeneous cache types (e.g., KVCache + ArraysCache)
- Test `ChunkedKVCache` chunked storage matches Python behavior
- Test `QuantizedKVCache` quantize/dequantize round-trip

**Green:**
- [ ] `_BaseCache` base class in `cache.rb`
- [ ] `ArraysCache` — arbitrary array state cache
- [ ] `CacheList` — heterogeneous cache container
- [ ] `ChunkedKVCache` — chunked KV storage
- [ ] `QuantizedKVCache` — quantized KV cache
- [ ] `can_trim_prompt_cache()` and `trim_prompt_cache()`

**Exit:** All cache type tests pass; existing cache tests still pass.

#### Phase 1C: MLA (Multi-head Latent Attention)

**Red:**
- Test `MultiLinear` forward pass matches Python for DeepSeek-V3 dimensions
- Test `QuantizedMultiLinear` matches Python

**Green:**
- [ ] `lib/mlx_lm/models/mla.rb` — `MultiLinear`, `QuantizedMultiLinear`

**Exit:** MLA layer tests pass.

#### Phase 1D: SSM Primitives

**Red:**
- Test `ssm_update()` state transition matches Python for Mamba config
- Test `ssm_attn()` matches Python for Mamba2 config
- Test `segsum()` matches Python

**Green:**
- [ ] `lib/mlx_lm/models/ssm.rb` — `compute_dt`, `segsum`, `ssm_attn`, `ssm_update`

**Exit:** SSM primitive tests pass.

#### Phase 1E: Gated Delta & Remaining Infrastructure

**Red:**
- Test `gated_delta_update` matches Python
- Test `SwitchMLP` forward pass matches Python
- Test `QuantizedSwitchLinear` matches Python
- Test `PipelineMixin` splits layers correctly
- Test `BitLinear` forward pass matches Python

**Green:**
- [ ] `lib/mlx_lm/models/gated_delta.rb` — gated delta rule ops
- [ ] Add `SwitchMLP`, `QuantizedSwitchLinear` to `switch_layers.rb`
- [ ] `lib/mlx_lm/models/pipeline.rb` — `PipelineMixin`
- [ ] `lib/mlx_lm/models/bitlinear_layers.rb` — `BitLinear`, `bitnet_quantize`

**Exit:** All infrastructure tests pass.

---

### Phase 2: Dense Transformer Models (No Special Infrastructure)

**Objective:** Port all simple dense transformer models that need only basic
RoPE (or no RoPE), standard KV cache, and standard attention.

Models (29 architectures):

| # | Model | Lines | Notes |
|---|-------|-------|-------|
| 1 | `qwen3` | 223 | Latest Qwen, Q/K norm |
| 2 | `cohere2` | 217 | Updated Cohere |
| 3 | `gemma3_text` | 257 | Interleaved local/global attention, clip_residual |
| 4 | `phi` | 174 | Original Phi (parallel residual) |
| 5 | `exaone` | 165 | LG AI Research |
| 6 | `exaone4` | 220 | Updated EXAONE |
| 7 | `glm` | 188 | Zhipu AI |
| 8 | `glm4` | 181 | Zhipu AI latest |
| 9 | `granite` | 193 | IBM (attention_multiplier, logit_scale) |
| 10 | `internlm3` | 238 | Updated InternLM |
| 11 | `olmo` | 177 | AI2 original |
| 12 | `olmo3` | 236 | AI2 latest (Q/K norm) |
| 13 | `gpt2` | 200 | OpenAI classic |
| 14 | `gpt_bigcode` | 185 | StarCoder1 base |
| 15 | `minicpm` | 204 | OpenBMB |
| 16 | `minicpm3` | 251 | OpenBMB latest (MLA-like kv_lora) |
| 17 | `nemotron` | 216 | NVIDIA (relu_squared, LayerNorm1P) |
| 18 | `helium` | 183 | Standard transformer |
| 19 | `seed_oss` | 184 | ByteDance |
| 20 | `ernie4_5` | 165 | Baidu |
| 21 | `baichuan_m1` | 251 | Baichuan |
| 22 | `nanochat` | 233 | Softcap attention |
| 23 | `telechat3` | 202 | Standard transformer |
| 24 | `apertus` | 188 | Standard transformer |
| 25 | `lille-130m` | 155 | Small model |
| 26 | `mimo` | 194 | Multi-in/multi-out |
| 27 | `youtu_llm` | 239 | Standard transformer |
| 28 | `hunyuan_v1_dense` | 231 | DynamicNTK RoPE |
| 29 | `qwen` | 157 | Original Qwen v1 |

Plus VL wrappers that become trivially portable:
- `pixtral` → wraps llama (already done)
- `qwen2_vl` → wraps qwen2 (already done)
- `qwen3_vl` → wraps qwen3 (after qwen3 is ported)
- `gemma3` → wraps gemma3_text (after gemma3_text is ported)

Plus aliases (just add to REMAPPING):
- `iquestcoder` → llama

**Red:** For each model, test forward pass with tiny config → logits match Python `atol=1e-4`.

**Green:**
- [ ] Port each model file following the standard pattern
- [ ] Register in `Models::REGISTRY`
- [ ] Add `sanitize` weight remapping where needed
- [ ] Add VL wrappers and REMAPPING entries

**Exit:** All 29+ models pass forward-pass parity tests.

---

### Phase 3: MoE Models (Depend on switch_layers)

**Objective:** Port all Mixture-of-Experts models. `SwitchLinear` and `SwitchGLU`
are already ported; this phase adds models using them, plus `SwitchMLP` where needed.

Models (20 architectures):

| # | Model | Lines | Notes |
|---|-------|-------|-------|
| 1 | `qwen2_moe` | 238 | Shared expert + routed experts |
| 2 | `qwen3_moe` | 259 | Updated MoE routing |
| 3 | `olmoe` | 214 | AI2 MoE |
| 4 | `phimoe` | 210 | MS MoE (Phi-based) |
| 5 | `phixtral` | 202 | Community MoE (Phi-based, different structure) |
| 6 | `dbrx` | 251 | Databricks MoE (NormAttnNorm pattern) |
| 7 | `glm4_moe` | 403 | Zhipu MoE + PipelineMixin |
| 8 | `granitemoe` | 235 | IBM MoE (TopKGating) |
| 9 | `exaone_moe` | 439 | LG MoE (group_expert_select) |
| 10 | `llama4` / `llama4_text` | 325/182 | Meta Llama 4 with MoE layers |
| 11 | `hunyuan` | 334 | Tencent MoE (DynamicNTK RoPE) |
| 12 | `ernie4_5_moe` | 289 | Baidu MoE |
| 13 | `afmoe` | 405 | Apple Foundation MoE |
| 14 | `minimax` | 394 | MiniMax (sharded RMSNorm) |
| 15 | `gpt_oss` | 343 | MoE with custom topk |
| 16 | `dots1` | 316 | MoE with TopK router |
| 17 | `step3p5` | 505 | ClampedSwiGLU + MoE |
| 18 | `Klear` | 263 | MoE |
| 19 | `bailing_moe` | 401 | Custom group_expert_select |
| 20 | `mimo_v2_flash` | 384 | MoE with group_expert_select |

Plus VL wrappers: `qwen3_vl_moe`

Plus aliases:
- `solar_open` → glm4_moe
- `phi-msft` → phixtral (REMAPPING)
- `minimax_m2` → minimax (REMAPPING)
- `llava` → mistral3 (REMAPPING)

**Red:** For each MoE model, test forward pass with tiny config → logits match Python.

**Green:**
- [ ] Add `SwitchMLP` to `switch_layers.rb` if needed
- [ ] Port each model file
- [ ] Register in `Models::REGISTRY`
- [ ] Add REMAPPING entries

**Exit:** All 20+ MoE models pass forward-pass parity tests.

---

### Phase 4: DeepSeek-V2/V3 Family (Depend on MLA + switch_layers)

**Objective:** Port the DeepSeek-V2/V3 architecture family, which requires MLA
(Multi-head Latent Attention), YaRN RoPE, and switch_layers.

Models (6 architectures):

| # | Model | Lines | Notes |
|---|-------|-------|-------|
| 1 | `deepseek_v2` | 501 | MLA + YaRN RoPE + MoE |
| 2 | `deepseek_v3` | 553 | Improved MLA + MoE + group_expert_select |
| 3 | `deepseek_v32` | 657 | Latest DeepSeek, Indexer pattern |
| 4 | `glm4_moe_lite` | 531 | Uses MLA + MoE + PipelineMixin |
| 5 | `kimi_linear` | 611 | MLA + DeltaAttention + ShortConv1d + MoE |
| 6 | `longcat_flash` | 493 | MLA + MoE + TopK router |

Plus aliases/wrappers:
- `glm_moe_dsa` → deepseek_v32
- `kimi_k25` → wraps deepseek_v3
- `kimi_vl` → wraps deepseek_v3
- `kimi_k2` → deepseek_v3 (REMAPPING)
- `joyai_llm_flash` → deepseek_v3 (REMAPPING)

**Red:** Forward pass parity tests for each model.

**Green:**
- [ ] Port each model file (deepseek_v2, deepseek_v3, deepseek_v32 first — they're the foundation)
- [ ] Port glm4_moe_lite, kimi_linear, longcat_flash
- [ ] Add aliases, wrappers, REMAPPING entries

**Exit:** All 6+ models pass forward-pass parity tests.

---

### Phase 5: SSM & Hybrid Models (Depend on SSM + Extended Cache)

**Objective:** Port state-space models and hybrid SSM/transformer architectures.
These require `ssm.rb`, `ArraysCache`, and `CacheList`.

Models (10 architectures):

| # | Model | Lines | Notes |
|---|-------|-------|-------|
| 1 | `mamba` | 221 | Pure SSM |
| 2 | `mamba2` | 264 | Improved SSM |
| 3 | `jamba` | 385 | SSM + Transformer hybrid + MoE |
| 4 | `falcon_h1` | 504 | Hybrid Mamba2 + Attention |
| 5 | `recurrent_gemma` | 452 | RGLRU recurrent (no SSM import, custom RNN) |
| 6 | `rwkv7` | 453 | Linear attention / RNN hybrid |
| 7 | `plamo2` | 478 | Hybrid Mamba + Attention |
| 8 | `nemotron_h` | 526 | NVIDIA hybrid (Mamba2 + Attention + MoE) |
| 9 | `granitemoehybrid` | 559 | IBM hybrid (SSM + Attention + MoE) |
| 10 | `lfm2` / `lfm2_moe` | 316/387 | Liquid Foundation (ShortConv + Attention + MoE) |

Plus wrappers:
- `lfm2-vl` → wraps lfm2
- `falcon_mamba` → mamba (REMAPPING)

**Red:** Forward pass parity tests; SSM state update parity tests.

**Green:**
- [ ] Port each model file
- [ ] Ensure `ArraysCache` and `CacheList` work for SSM state
- [ ] Register in `Models::REGISTRY`

**Exit:** All 10+ models pass forward-pass and state update parity tests.

---

### Phase 6: Gated-Delta & Advanced Hybrid Models

**Objective:** Port models requiring gated-delta attention primitives.

Models (4 architectures):

| # | Model | Lines | Notes |
|---|-------|-------|-------|
| 1 | `qwen3_5` | 396 | GatedDeltaNet + CacheList |
| 2 | `qwen3_next` | 471 | GatedDeltaNet + MoE + CacheList |
| 3 | `bailing_moe_linear` | 595 | Recurrent GLA + MoE |
| 4 | `longcat_flash_ngram` | 214 | Ngram embeddings, wraps longcat_flash |

Plus wrappers:
- `qwen3_5_moe` → wraps qwen3_5

**Red:** Forward pass parity tests; gated-delta state update tests.

**Green:**
- [ ] Port each model file
- [ ] Register in `Models::REGISTRY`

**Exit:** All 4+ models pass parity tests.

---

### Phase 7: Remaining Long-Tail Models

**Objective:** Port remaining specialized models that don't fit neatly into
earlier phases.

Models (9 architectures):

| # | Model | Lines | Notes |
|---|-------|-------|-------|
| 1 | `afm7` | 390 | Apple Foundation (FusedLoRALinear, KVReuse) |
| 2 | `bitnet` | 208 | Ternary weights (needs bitlinear_layers) |
| 3 | `phi3small` | 310 | GEGELU activation |
| 4 | `openelm` | 220 | Per-layer head/dim scaling |
| 5 | `ministral3` | 333 | Mistral small (PipelineMixin) |
| 6 | `nemotron-nas` | 395 | NVIDIA NAS (configurable block types) |
| 7 | `iquestloopcoder` | 286 | Loop gate projection |
| 8 | `gemma3n` | 613 | Gemma Nano (AltUp, LaurelBlock) |
| 9 | `smollm3` | 75 | NoPE (no positional encoding), extends llama |

Plus VL wrappers from earlier phases:
- `mistral3` → wraps ministral3 / llama
- `kimi_vl` → wraps deepseek_v3

**Red:** Forward pass parity tests.

**Green:**
- [ ] Port each model file
- [ ] Register in `Models::REGISTRY`
- [ ] Add final REMAPPING entries

**Exit:** All remaining models pass parity tests.

---

### Phase 8: HuggingFace Hub Integration

**Objective:** Enable loading models directly from HuggingFace Hub repo IDs
(e.g., `MlxLm::LoadUtils.load("mlx-community/Llama-3.2-1B-Instruct-4bit")`),
cache management, and model upload.

#### Phase 8A: Hub Download & Caching

**Red:**
- Test `hub_download("mlx-community/SmolLM2-135M-Instruct-4bit")` downloads to local cache
- Test cached download returns immediately without network call
- Test `load("mlx-community/SmolLM2-135M-Instruct-4bit")` works end-to-end

**Green:**
- [ ] `lib/mlx_lm/hub_utils.rb` — `HubUtils` module:
  - `snapshot_download(repo_id, revision:, allow_patterns:, local_dir:)` — download via HF Hub HTTP API
  - `hf_repo_to_path(repo_id)` — resolve repo to local cache path (`~/.cache/huggingface/hub/`)
  - `cached?(repo_id, revision:)` — check if already cached
  - Token auth via `HF_TOKEN` env var or `~/.cache/huggingface/token`
  - File-level caching with lock files and etag-based invalidation
- [ ] Update `LoadUtils.load` to accept HF repo IDs (call `_download` → local path → existing load logic)
- [ ] Support `allow_patterns` for selective download (e.g., only `*.safetensors`, `config.json`, `tokenizer.json`)

**Implementation approach:** Direct HTTP API calls to `https://huggingface.co/api/models/{repo_id}`
and file downloads from `https://huggingface.co/{repo_id}/resolve/{revision}/{filename}`.
Use Ruby `net/http` or the `down` gem. No Python dependency required.

**Exit:** Can load any quantized MLX model from HF Hub by repo ID.

#### Phase 8B: Model Saving & Sharding

**Red:**
- Test `save_model` writes valid safetensors + config.json
- Test sharded saving splits at 5GB boundary
- Test saved model loads back identically

**Green:**
- [ ] `MlxLm::WeightUtils.save_safetensors(path, weights)` — write safetensors file
- [ ] `MlxLm::WeightUtils.make_shards(weights, max_file_size_gb:)` — split large weight dicts
- [ ] `MlxLm::LoadUtils.save_model(path, model:, tokenizer:, config:)` — save full model
- [ ] `MlxLm::LoadUtils.save_config(config, path)` — write config.json

**Exit:** Round-trip save/load produces identical model.

#### Phase 8C: Hub Upload & Cache Management

**Red:**
- Test `upload_to_hub` creates repo and pushes files
- Test `manage` lists cached models
- Test `manage` deletes cached models

**Green:**
- [ ] `MlxLm::HubUtils.upload_to_hub(path, repo_id:, token:)` — upload via HF API
- [ ] `MlxLm::HubUtils.create_model_card(path, hf_path:)` — generate README.md
- [ ] `MlxLm::Manage` module — list/delete cached models
- [ ] CLI: `mlx_lm manage` subcommand

**Exit:** Full upload workflow works; cache management lists and deletes correctly.

---

### Phase 9: ONNX Export Support

**Objective:** Ensure every model architecture can be exported to ONNX format via
`MLX::ONNX.export_onnx`. This enables deployment to ONNX Runtime and other ONNX-compatible
inference engines.

**Prerequisites:**
- mlx-onnx submodule updated to `33d4b2e` or later (adds `ArgPartition` and `GatherMM` ops)
- The two previously missing ops (`ArgPartition`, `GatherMM`) are used by all MoE models
  for top-k expert selection and batched per-expert matrix multiplication respectively

**Background:**
The existing ONNX export test infrastructure (`test/parity/onnx_export_test.rb`) runs each
model in an isolated subprocess, performing:
1. **Compatibility report** — node-by-node probe that checks if each MLX op can be lowered
   to an ONNX equivalent (via `MLX::ONNX.export_onnx_compatibility_report`)
2. **Full export** — traces the model and writes a `.onnx` file
   (via `MLX::ONNX.export_onnx`)

Each model needs a `TINY_CONFIGS` entry with minimal dimensions for fast tracing.

**Known issue:** MoE models (mixtral, deepseek, etc.) crash during full ONNX export tracing
because `tolist` forces data-dependent control flow (per-token expert routing). The compat
report runs successfully because it probes ops individually. This requires either upstream
mlx-onnx support for data-dependent control flow or a tracing workaround.

#### Phase 9A: Update Submodule & Verify Existing Models

**Red:**
- Test updated mlx-onnx builds successfully with ArgPartition + GatherMM support
- Test all 13 existing models pass ONNX compat report at 100% node coverage
- Test dense models (llama, gemma, qwen2, etc.) pass full ONNX export

**Green:**
- [ ] Update `mlx-ruby/submodules/mlx-onnx` to latest (`33d4b2e`+)
- [ ] Rebuild mlx-onnx native extension
- [ ] Verify all 13 existing model compat tests pass (mixtral/deepseek should now show 100% node support)
- [ ] Verify dense models produce valid `.onnx` files

**Exit:** All existing model ONNX compat tests pass at 100%; dense models fully export.

#### Phase 9B: ONNX Compat Tests for New Dense Models (Phases 2 + 7)

**Red:**
- Test each newly ported dense model passes ONNX compat report at 100%
- Test each newly ported dense model produces valid `.onnx` file

**Green:**
- [ ] Add `TINY_CONFIGS` entries for all Phase 2 dense models (29 models):
  qwen3, cohere2, gemma3_text, phi, exaone, exaone4, glm, glm4, granite, internlm3,
  olmo, olmo3, gpt2, gpt_bigcode, minicpm, minicpm3, nemotron, helium, seed_oss,
  ernie4_5, baichuan_m1, nanochat, telechat3, apertus, lille-130m, mimo, youtu_llm,
  hunyuan_v1_dense, qwen
- [ ] Add `TINY_CONFIGS` entries for Phase 7 long-tail dense models (9 models):
  afm7, bitnet, phi3small, openelm, ministral3, nemotron-nas, iquestloopcoder,
  gemma3n, smollm3
- [ ] Verify all pass compat + full export

**Exit:** All dense model architectures pass ONNX export tests.

#### Phase 9C: ONNX Compat Tests for MoE Models (Phases 3 + 4 + 6)

**Red:**
- Test each MoE model passes ONNX compat report at 100% node coverage
- Test MoE full export (expected: crash due to data-dependent control flow; track upstream fix)

**Green:**
- [ ] Add `TINY_CONFIGS` entries for all Phase 3 MoE models (20 models):
  qwen2_moe, qwen3_moe, olmoe, phimoe, phixtral, dbrx, glm4_moe, granitemoe,
  exaone_moe, llama4/llama4_text, hunyuan, ernie4_5_moe, afmoe, minimax, gpt_oss,
  dots1, step3p5, Klear, bailing_moe, mimo_v2_flash
- [ ] Add `TINY_CONFIGS` entries for Phase 4 DeepSeek family (6 models):
  deepseek_v2, deepseek_v3, deepseek_v32, glm4_moe_lite, kimi_linear, longcat_flash
- [ ] Add `TINY_CONFIGS` entries for Phase 6 gated-delta models (4 models):
  qwen3_5, qwen3_next, bailing_moe_linear, longcat_flash_ngram
- [ ] All MoE models should pass compat report at 100% (ArgPartition + GatherMM now supported)
- [ ] Track MoE full export crash status — document workarounds or upstream fixes

**Exit:** All MoE model architectures pass ONNX compat at 100%. Full export tracked.

#### Phase 9D: ONNX Compat Tests for SSM & Hybrid Models (Phase 5)

**Red:**
- Test each SSM/hybrid model passes ONNX compat report
- Identify any SSM-specific ops that lack ONNX mappings

**Green:**
- [ ] Add `TINY_CONFIGS` entries for Phase 5 SSM models (10 models):
  mamba, mamba2, jamba, falcon_h1, recurrent_gemma, rwkv7, plamo2, nemotron_h,
  granitemoehybrid, lfm2/lfm2_moe
- [ ] Identify and document any SSM-specific ops missing from mlx-onnx
- [ ] File upstream issues for any missing SSM op mappings

**Exit:** SSM model ONNX compat status fully documented; achievable models pass.

#### Phase 9E: VL Wrapper & Alias Coverage

**Red:**
- Test VL wrapper models pass ONNX compat (text-only path)
- Test aliased models resolve correctly for ONNX export

**Green:**
- [ ] Add `TINY_CONFIGS` entries for VL wrappers: pixtral, qwen2_vl, qwen3_vl,
  gemma3, mistral3, lfm2-vl, kimi_vl, qwen3_vl_moe
- [ ] Verify MODEL_REMAPPING aliases work with ONNX export test infrastructure
- [ ] Ensure all ~109 model architectures have ONNX compat test coverage

**Exit:** Complete ONNX test coverage for all registered model architectures.

---

### Execution Checklist

- [ ] Phase 1A: RoPE variants complete
- [ ] Phase 1B: Extended cache types complete
- [ ] Phase 1C: MLA complete
- [ ] Phase 1D: SSM primitives complete
- [ ] Phase 1E: Gated delta, SwitchMLP, PipelineMixin, BitLinear complete
- [ ] Phase 2: Dense transformer models (29 models) complete
- [ ] Phase 3: MoE models (20 models) complete
- [ ] Phase 4: DeepSeek-V2/V3 family (6 models) complete
- [ ] Phase 5: SSM & hybrid models (10 models) complete
- [ ] Phase 6: Gated-delta & advanced hybrid (4 models) complete
- [ ] Phase 7: Long-tail models (9 models) complete
- [ ] Phase 8A: HF Hub download & caching complete
- [ ] Phase 8B: Model saving & sharding complete
- [ ] Phase 8C: Hub upload & cache management complete
- [ ] Phase 9A: mlx-onnx submodule updated, existing models verified
- [ ] Phase 9B: ONNX tests for dense models (38 models) complete
- [ ] Phase 9C: ONNX tests for MoE models (30 models) complete
- [ ] Phase 9D: ONNX tests for SSM & hybrid models (10 models) complete
- [ ] Phase 9E: ONNX tests for VL wrappers & aliases complete
- [ ] All model REMAPPING entries added
- [ ] All VL wrapper models added
- [ ] Integration / regression checks complete
- [ ] PRD status updated to Completed

### Model Count Summary

| Phase | New Models | Cumulative | Notes |
|-------|-----------|------------|-------|
| Existing | 13 | 13 | llama, gemma, gemma2, qwen2, phi3, starcoder2, stablelm, cohere, olmo2, gpt_neox, mixtral, deepseek, internlm2 |
| Phase 2 | 29 + wrappers | 42+ | Dense transformers (no special infra) |
| Phase 3 | 20 + aliases | 62+ | MoE models |
| Phase 4 | 6 + aliases | 68+ | DeepSeek-V2/V3 + MLA models |
| Phase 5 | 10 + wrappers | 78+ | SSM & hybrid models |
| Phase 6 | 4 + wrappers | 82+ | Gated-delta models |
| Phase 7 | 9 + wrappers | 91+ | Long-tail specialized |
| Remappings | ~10 | 101+ | MODEL_REMAPPING aliases |
| VL Wrappers | ~8 | 109+ | Thin VL delegation wrappers |
| Phase 9 | — | 109+ | ONNX export tests for all architectures |
| **Total** | **~96 new** | **~109** | Full upstream coverage + ONNX export |
