# Product Requirements Document: mlx-ruby-lm

## 1. Product Overview

**mlx-ruby-lm** is a Ruby gem that provides a complete port of the Python `mlx-lm`
package for large language model inference and fine-tuning, built on top of the
`mlx-ruby` gem. It targets 100% functional parity with the Python implementation
while being idiomatically Ruby.

## 2. Goals

- Provide Ruby developers with a native LLM inference library on MLX
- 100% functional parity with Python mlx-lm v0.30.7
- Idiomatic Ruby API (snake_case, blocks, keyword arguments, Enumerable)
- Every feature validated by parity tests comparing Ruby ↔ Python output

## 3. Non-Goals

- GUI or web frontend
- Custom model training from scratch (only LoRA fine-tuning)
- Support for non-MLX backends

## 4. Technical Architecture

- **Runtime:** Ruby >= 3.3
- **Test framework:** Minitest
- **ML backend:** mlx-ruby gem (wraps MLX C++ via native extension)
- **Tokenizer:** `tokenizers` gem (HuggingFace Rust bindings)
- **Weights:** `safetensors` gem for loading/saving
- **Namespace:** `MlxLm` module

## 5. Implementation Phases

See `prd/conversion_plan.md` for the complete 12-phase plan with parity tests.

## 6. Success Criteria

- All 76 parity tests pass (comparing Ruby output to Python output)
- `MlxLm.load` + `MlxLm.generate` produces identical output to Python
- All 100+ model architectures supported
- OpenAI-compatible HTTP server works identically
