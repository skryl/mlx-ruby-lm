# mlx-ruby-lm Documentation Restructure PRD (Agent Tranches)

**Status:** Active  
**Date:** 2026-02-27  
**Owner:** Documentation/Developer Experience

## 1. Purpose

Restructure project documentation so the top-level README is a concise onboarding entrypoint, while all detailed functionality documentation lives in dedicated `docs/*` pages.

This PRD defines an agent-tranche execution plan that is decision-complete and safe to parallelize.

## 2. Background and Problem

Current docs are concentrated in `README.md` with mixed depth:

- CLI, API inventory, and full model registry are all in one file.
- The README is too dense for first-time users.
- Detailed references are not separated by concern.

Required target state:

- README covers only installation, CLI usage, and high-level API/model usage examples.
- README has a top-level index.
- Detailed API and model docs are moved into `docs/*`.

## 3. Goals

1. Create a complete `docs/` set for current gem functionality.
2. Move README detail sections into dedicated docs pages.
3. Preserve factual accuracy against current runtime behavior.
4. Keep edits documentation-only (no runtime behavior changes).

## 4. Non-Goals

1. No feature implementation or runtime refactors.
2. No gem packaging behavior changes.
3. No doc site generator or publishing pipeline changes.

## 5. Scope

### In Scope

- README refactor and top-level index.
- New docs pages under `docs/` with cross-links.
- API coverage for public modules/classes and advanced internals.
- Full model registry page (single page, full list).
- Explicit behavior caveats where implementation is partial.

### Out of Scope

- Updating historical PRDs.
- Introducing new CLI flags or changing CLI semantics.
- Changing tests unless needed for docs link/format validation (not expected).

## 6. Documentation Source of Truth

All docs content must be derived from current code and test contracts in:

- `lib/mlx_lm.rb`
- `lib/mlx_lm/*.rb`
- `lib/mlx_lm/models/*.rb`
- `test/parity/*_test.rb` (behavioral contracts and caveats)
- `exe/mlx_lm` and CLI runtime output (`mlx_lm ... --help`)

## 7. Tranche Plan

## Tranche 1: README and Core Docs (Launch First)

**Objective:** Deliver high-traffic docs and move README detail sections out.

**Owned Files:**

- `README.md`
- `docs/index.md`
- `docs/installation.md`
- `docs/cli.md`
- `docs/ruby-apis.md`
- `docs/models.md`

**Required Changes:**

1. Add top-level `Index` section to README near the top.
2. Remove README sections equivalent to detailed `Ruby APIs` and `Included models` inventories.
3. Keep README focused on:
   - Installation
   - CLI usage
   - High-level API usage examples
   - High-level model usage examples
4. Add and link core docs pages listed above.
5. In `docs/cli.md`, document current behavior exactly, including caveats for parser-only/unapplied flags.
6. In `docs/models.md`, include full sorted 106 model keys and remapping notes.

**Acceptance Gates:**

- `docs/` exists and contains all tranche-owned files.
- README has top-level index and no detailed API/model inventory sections.
- README links to docs pages resolve.
- `docs/models.md` registry list matches runtime registry count and keys.

## Tranche 2: Runtime and Serving Deep Dives

**Objective:** Cover runtime workflows beyond the core landing docs.

**Owned Files:**

- `docs/loading-tokenization.md`
- `docs/generation-sampling.md`
- `docs/server.md`

**Required Changes:**

1. Document model loading/config/weights/tokenizer flow.
2. Document generation surfaces (`generate_step`, `stream_generate`, `generate`) and sampling processors.
3. Document server schema/streaming behavior and caveats.

**Acceptance Gates:**

- All new pages linked from `docs/index.md`.
- Examples match current method signatures and request/response structures.

## Tranche 3: Advanced and Internal Surfaces

**Objective:** Document advanced modules and lower-level internals.

**Owned Files:**

- `docs/quantization.md`
- `docs/lora.md`
- `docs/cache.md`
- `docs/evaluation-conversion.md`
- `docs/model-internals.md`

**Required Changes:**

1. Quantization/dequantization behavior and model-size/bits utilities.
2. LoRA adapters and layer-application flow.
3. Cache class hierarchy and prompt cache serialization helpers.
4. Perplexity/benchmark/conversion utilities.
5. Internal model utility modules (RoPE, SSM, gated delta, switch layers, MLA, pipeline mixin) marked as advanced/non-stable.

**Acceptance Gates:**

- Terminology and method names match runtime exports.
- Advanced/internal stability caveat appears on `docs/model-internals.md`.

## Tranche 4: QA and Consistency Sweep

**Objective:** Final quality pass for navigability and correctness.

**Owned Files:**

- All `docs/*.md`
- `README.md`

**Required Changes:**

1. Verify all internal links.
2. Normalize heading style and code examples.
3. Ensure cross-links are two-click maximum from README/docs index.
4. Validate caveats are present where implementation differs from implied flag surface.

**Acceptance Gates:**

- No broken local markdown links.
- No conflicting descriptions across docs pages.
- README remains concise and aligned with goals.

## 8. Agent Execution Contract

For each tranche worker:

1. Worker is file-owner for listed files only.
2. Worker must ignore unrelated repository changes.
3. Worker must include a completion summary with:
   - Files changed
   - Key decisions/caveats
   - Any blockers
4. Worker must not change runtime code for this PRD unless explicitly requested.

## 9. Risks and Mitigations

1. **Risk:** Docs overstate behavior versus implementation.  
   **Mitigation:** Require code-backed statements and explicit caveats.
2. **Risk:** README drifts back into reference-manual size.  
   **Mitigation:** Keep README to onboarding-only sections.
3. **Risk:** Model registry list goes stale.  
   **Mitigation:** Derive from `MlxLm::Models::REGISTRY.keys.sort` at writing time.

## 10. Success Definition

This initiative is complete when:

1. README matches target structure and scope.
2. `docs/` contains complete dedicated references for APIs and models.
3. All docs pages are cross-linked and navigable.
4. Documentation reflects current behavior, including caveats.
