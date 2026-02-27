# Python `mlx-lm` -> Ruby `mlx-ruby-lm` Parity PRD (Execution Revision)

**Status:** Completed (Architecture + Class-Level Parity Closure Achieved)
**Date:** 2026-02-27
**Supersedes:** prior draft from 2026-02-25
**References:**
- [Python-Ruby Parity Checklist](2026_02_25_python_ruby_parity_checklist.md)
- [Parity Inventory Snapshot](../test/reports/python_ruby_parity_inventory_snapshot.json)

## 1. Purpose

Maintain execution-grade parity for inference-focused model coverage between Python
`mlx-lm` and Ruby `mlx-ruby-lm`, with reproducible gates and artifacted validation.

This revision updates the plan to match current repo reality:

- inventory parity is complete at architecture-key level
- class-level parity now has an explicit closure plan (`768` classes total)
- inventory generation/checking is task-based (not script-based)
- ONNX reporting is task-based with Markdown/JSON artifacts
- default ONNX full-export behavior is gated to avoid unnecessary heavy runs

## 2. Baseline Snapshot (Frozen)

Source: `test/reports/python_ruby_parity_inventory_snapshot.json`
Regenerate with: `bundle exec rake parity:inventory`
Validate with: `bundle exec rake parity:inventory_check`

- Python model files in `mlx-lm/mlx_lm/models`: **116**
- Python shared infra files: **10**
- Python architecture files: **106**
- Ruby model files in `lib/mlx_lm/models`: **115**
- Ruby shared infra files: **2**
- Ruby architecture files: **113**
- Ruby registered architecture keys: **106**
- Current architecture key gap: **0**
- Python classes discovered (`mlx-lm/mlx_lm/**/*.py`): **768**
- Ruby class parity status: **768 Implemented / 0 Partial / 0 Missing**

## 3. Scope

### In Scope

- Maintain architecture-key parity between upstream Python model set and Ruby registry
- Execute class-level parity closure for remaining Partial/Missing classes
- Keep governance gates reproducible and green
- Run ONNX compatibility coverage and produce artifacted reports
- Track unsupported ONNX ops with exact invocation details

### Out of Scope

- Speculative decoding R&D paths
- Performance tuning and benchmark optimization work

## 4. Hard Gates (Must Pass)

### G1: Inventory Freeze Gate

**Requirement:** `test/reports/python_ruby_parity_inventory_snapshot.json` is current.

- Regenerate snapshot: `bundle exec rake parity:inventory`
- Check freshness: `bundle exec rake parity:inventory_check`
- CI/test gate: `test/parity/governance_parity_gates_test.rb`

### G2: ONNX Submodule Minimum Commit Gate

**Requirement:** `mlx-ruby/submodules/mlx-onnx` includes required lowering support.

- Minimum commit: `33d4b2eed2aa342f0836298dda60b6c5eb011b0f`
- Validation method: `git merge-base --is-ancestor <min_sha> HEAD`
- CI/test gate: `test/parity/governance_parity_gates_test.rb`

### G3: ONNX Reporting Gate

**Requirement:** compat report generation is reproducible and artifacted.

- Task: `bundle exec rake onnx:report`
- Artifacts:
  - `test/reports/onnx_compat_test_output.txt`
  - `test/reports/onnx_compat_full_report.json`
  - `test/reports/onnx_compat_full_report.md`
  - `test/reports/onnx_compat_missing_ops_invocations.csv`

### G4: Class Parity Checklist Gate

**Requirement:** class-level parity checklist is current and reviewed before merges
that affect parity-sensitive files.

- Source of truth: `prd/2026_02_25_python_ruby_parity_checklist.md`
- Required fields: Python class, status, Ruby reference, notes
- Change policy: PRs changing model/server/generation/tuner code must update the checklist

## 5. Public API / Workflow Contract Updates

### Inventory workflow contract

Inventory generation must run via task class, not ad hoc scripts:

1. `tasks/parity_inventory_task.rb` is the implementation unit
2. `rake parity:inventory` writes snapshot output
3. `rake parity:inventory_check` enforces freshness and fails on drift

### ONNX validation artifact contract

Compat reporting must produce machine-readable and human-readable outputs:

1. full per-model compat JSON payload
2. markdown summary report
3. unsupported-op union and per-model incidence
4. unsupported-op invocation list (indexed)

## 6. Execution Plan (Post-Parity)

## Phase A: Governance Stability

**Objective:** Keep parity gates deterministic and low-maintenance.

**Exit Criteria:**
- inventory and submodule governance tests are green
- inventory task outputs remain stable

## Phase B: ONNX Compat Matrix Quality

**Objective:** maximize compat coverage and minimize unsupported ops.

**Exit Criteria:**
- compat suite runs for all ONNX model tests
- missing-op list is explicit with model + invocation context
- report artifacts regenerate cleanly through `rake onnx:report`

## Phase C: ONNX Export Reliability

**Objective:** reduce hangs and classify failures predictably.

**Exit Criteria:**
- subprocess timeout behavior is enforced
- full export remains opt-in by default
- failing exports are categorized in report outputs

## Phase D: Core Runtime API Class Closure

**Objective:** close high-impact runtime API gaps that currently block feature parity.

**Target areas:**
- `generate.py`: `GenerationResponse`, `BatchStats`, `BatchResponse`, `Batch`, `BatchGenerator`, `Response`
- `tokenizer_utils.py`: `NaiveStreamingDetokenizer`, `SPMStreamingDetokenizer`,
  `BPEStreamingDetokenizer`, `NewlineTokenizer`
- `models/cache.py`: `_BaseCache` alignment, `ConcatenateKVCache`,
  `BatchKVCache`, `BatchRotatingKVCache`

**Exit Criteria:**
- targeted classes move from `Partial` to `Implemented`
- batch generation and detokenizer behavior have parity tests
- cache-family parity tests cover batch cache variants

## Phase E: Server Surface Completion

**Objective:** complete Python server class parity for request/response and runtime paths.

**Target areas (`server.py`):**
- `StopCondition`
- `LRUPromptCache`, `CacheEntry`, `SearchResult`
- `ModelDescription`, `SamplingArguments`, `LogitsProcessorArguments`,
  `GenerationArguments`, `CompletionRequest`, `GenerationContext`, `Response`, `TimeBudget`
- `ModelProvider`, `ResponseGenerator`, `APIHandler`

**Exit Criteria:**
- `/v1/chat/completions`, `/v1/completions`, and `/v1/models` behavior reaches parity
- stop/tool and streaming behaviors are covered by parity tests
- `server.py` class inventory has zero `Partial` entries

## Phase F: Model Internal Class Closure (High-Delta Files)

**Objective:** close class-level internal gaps in architecture files that are currently only partially mirrored.

**Wave F1 (SSM/Hybrid focus):**
- `models/falcon_h1.py`, `models/jamba.py`, `models/nemotron_h.py`,
  `models/plamo2.py`, `models/qwen3_next.py`, `models/rwkv7.py`, `models/kimi_linear.py`

**Wave F2 (MoE/DeepSeek focus):**
- `models/afm7.py`, `models/bailing_moe_linear.py`, `models/deepseek_v2.py`,
  `models/deepseek_v3.py`, `models/deepseek_v32.py`, `models/granitemoehybrid.py`,
  `models/glm4_moe_lite.py`, `models/longcat_flash.py`, `models/lfm2.py`, `models/minimax.py`

**Wave F3 (Naming/alias cleanup):**
- close remaining class-name drift where structure exists but names differ
  (for example `DBRX` vs `DbrxModel`, `CohereModel` vs `Cohere2Model`)

**Exit Criteria:**
- high-delta model files above have no unresolved structural class gaps
- class parity `Partial` count is reduced materially from baseline (`221`)
- model parity and ONNX compat tests remain green for touched models

## Phase G: Missing Utility Modules

**Objective:** implement currently missing non-model utility classes.

**Target areas (`Missing` classes):**
- `evaluate.py`: `MLXLM`
- `share.py`: `DirectoryEntry`
- `gguf.py`: `TokenType`, `GGMLFileType`, `HfVocab`
- `quant/awq.py`, `quant/gptq.py`: `ScaleConfig`, `AWQConfig`, `Catcher`

**Exit Criteria:**
- utility class set above is implemented or explicitly superseded with documented mapping
- missing-class count is reduced to only tuner/training classes

## Phase H: Tuner and Training Stack Parity

**Objective:** complete the training-side classes currently missing in Ruby.

**Target areas:**
- `tuner/trainer.py`: `TrainingArgs`
- `tuner/callbacks.py`: `TrainingCallback`, `WandBCallback`, `SwanLabCallback`
- `tuner/datasets.py`: `TextDataset`, `ChatDataset`, `CompletionsDataset`,
  `ConcatenatedDataset`, `CacheDataset`
- `tuner/dora.py`: `DoRALinear`, `DoRAEmbedding`

**Exit Criteria:**
- all tuner/training classes above have Ruby implementations
- training smoke tests cover LoRA + DoRA paths and dataset adapters
- `Missing` class count reaches `0`

## Phase I: Final Parity Closure and Maintenance

**Objective:** make class-level parity drift-resistant after closure.

**Exit Criteria:**
- class inventory remains at `Missing = 0`
- remaining `Partial` entries are either closed or explicitly accepted as intentional aliases
- parity checklist updates are enforced in PR review flow

## 7. Test Strategy

### Governance tests

- inventory snapshot freshness gate
- `mlx-onnx` minimum commit gate
- class parity checklist freshness/review gate

### ONNX tests

- compat-report path for full model set
- optional full-export path gated by env var

### Class parity tests

- per-phase class closure tests for touched modules
- API parity tests for generation/tokenizer/cache/server surfaces
- model parity tests for high-delta architecture files (Phase F waves)
- tuner smoke tests for callbacks/datasets/DoRA/trainer paths (Phase H)

### Execution policy

Every report refresh should produce committed artifacts when used for parity review:
- markdown summary
- full JSON payload
- missing-op invocation CSV

## 8. Execution Status (Current)

- [x] Architecture-key parity achieved (`106 / 106`)
- [x] Inventory task class implemented: `tasks/parity_inventory_task.rb`
- [x] Rake inventory tasks active: `parity:inventory`, `parity:inventory_check`
- [x] Governance parity gates active in parity test suite
- [x] ONNX report task active: `onnx:report`
- [x] ONNX report includes Markdown + JSON + invocation CSV artifacts
- [x] Full class-level checklist published (`768` classes tracked)
- [x] Class parity closure backlog active and closed (`0 Partial / 0 Missing`)
- [x] Phase D-I implementation completed in code

## 9. Success Definition

This parity program is successful when:

- architecture-key parity remains at zero gap against upstream inventory,
- class-level parity reaches `Missing = 0`,
- remaining `Partial` entries are zero or explicitly justified aliases,
- governance gates fail fast on drift,
- ONNX compat reports are reproducible and actionable,
- unresolved ONNX gaps are tracked by model, op, and invocation.
