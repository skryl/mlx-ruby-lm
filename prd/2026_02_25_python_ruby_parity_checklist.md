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

## 12. Implementation Plan

The detailed phased implementation plan is maintained in a separate document:
**[Implementation Plan](implementation_plan.md)**

The plan covers 9 phases:
1. **Shared Infrastructure** — RoPE variants, extended cache, MLA, SSM, gated delta, SwitchMLP, PipelineMixin, BitLinear
2. **Dense Transformers** — 29 models
3. **MoE Models** — 20 models
4. **DeepSeek-V2/V3 Family** — 6 models
5. **SSM & Hybrid Models** — 10 models
6. **Gated-Delta Models** — 4 models
7. **Long-Tail Models** — 9 models
8. **HuggingFace Hub Integration** — download, save, upload
9. **ONNX Export Validation** — all ~109 architectures

Each phase includes a **native library gap report** checkpoint that documents any
missing functionality in `mlx-ruby` or `mlx-onnx` discovered during implementation.
