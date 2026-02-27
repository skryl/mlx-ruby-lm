# Model Internals (Advanced, Non-Stable)

[Back to Documentation Index](index.md)

`MlxLm::Models` internals documented here are advanced implementation surfaces. They are not guaranteed stable across releases and may change without compatibility guarantees.

## RoPE Internals

Source: `lib/mlx_lm/models/rope_utils.rb`

### Implementations

- `MlxLm::Models::SuScaledRoPE`
- `MlxLm::Models::Llama3RoPE`
- `MlxLm::Models::YarnRoPE`

### Factory

```ruby
MlxLm::Models.initialize_rope(
  dims,
  base,
  traditional,
  scaling_config = nil,
  max_position_embeddings: nil
)
```

Dispatch behavior:

- `"default"` / `"linear"` -> `MLX::NN::RoPE`
- `"llama3"` -> `Llama3RoPE`
- `"yarn"`, `"deepseek_yarn"`, `"telechat3-yarn"` -> `YarnRoPE`
- `"longrope"` -> `SuScaledRoPE`
- `"mrope"` -> currently returns `MLX::NN::RoPE` after validating `mrope_section.length == 3`
- unknown type -> `ArgumentError`

Compatibility alias module:

- `MlxLm::Models::RoPEUtils` re-exports these classes and forwards `initialize_rope`.

Caveat:

- `SuScaledRoPE` initializer accepts `short_factor`/`short_mscale`, but the current implementation computes `_freqs`/`_scale` from long-scaling inputs.

## SSM Internals

Source: `lib/mlx_lm/models/ssm.rb`

Module: `MlxLm::Models::SSM`

Key functions:

- `compute_dt(dt, dt_bias, time_step_limit = [0.001, 100.0])`
- `segsum(x, mask: nil)`
- `ssm_attn(...)`
- `ssm_update(...)`
- `ssm_update_kernel(...)`

Current status:

- `ssm_attn` is the baseline explicit recurrence implementation.
- `ssm_update` usually dispatches to `ssm_attn`; kernel path is only considered for specific single-step GPU/Metal conditions.
- `ssm_update_kernel` currently raises `NotImplementedError`.
- `lengths:` path in `ssm_attn` is not implemented and raises `NotImplementedError`.

## Gated Delta Internals

Source: `lib/mlx_lm/models/gated_delta.rb`

Module: `MlxLm::Models::GatedDelta`

Key functions:

- `compute_g(a_log, a, dt_bias)`
- `gated_delta_ops(q, k, v, g, beta, state = nil, mask = nil)`
- `gated_delta_update(q, k, v, a, b, a_log, dt_bias, state = nil, mask = nil, use_kernel: true)`

Current status:

- `gated_delta_kernel` currently delegates to `gated_delta_ops` (no specialized Metal kernel implementation yet).
- `gated_delta_update` checks Metal/GPU availability and only then considers kernel path.
- Masked positions keep prior state for masked entries.

## Switch Layers Internals

Source: `lib/mlx_lm/models/switch_layers.rb`

Module: `MlxLm::Models::SwitchLayers`

Primary helpers and layers:

- `gather_sort(x, indices)` / `scatter_unsort(x, inv_order, shape = nil)`
- `SwitchLinear`
- `QuantizedSwitchLinear`
- `SwitchGLU`
- `SwitchMLP`

Key details:

- `SwitchLinear#to_quantized(...)` returns `QuantizedSwitchLinear`.
- `quantize_input: true` is explicitly unsupported and raises `ArgumentError`.
- `SwitchGLU`/`SwitchMLP` use a sorted fast path when `indices.size >= 64`.

## MLA Internals

Source: `lib/mlx_lm/models/mla.rb`

Module: `MlxLm::Models::MLA`

Classes:

- `MultiLinear`
- `QuantizedMultiLinear`

Key details:

- `MultiLinear#to_quantized(...)` returns `QuantizedMultiLinear`.
- `quantize_input: true` is explicitly unsupported and raises `ArgumentError`.
- Quantized path uses `MLX::Core.quantized_matmul`.

## Pipeline Partitioning Internals

Source: `lib/mlx_lm/models/pipeline.rb`

Mixin: `MlxLm::Models::PipelineMixin`

Fields:

- `pipeline_rank`
- `pipeline_size`
- `start_idx`
- `end_idx`

Methods:

- `pipeline_layers`
- `pipeline(group)`

Behavior:

- Splits layers in reverse rank order: rank `0` receives the last layer block.
- Mutates `self.layers` in place.
- Replaces non-local prefix with `nil` to preserve layer numbering for checkpoint compatibility.

## Caveats

- These modules are implementation internals and should be treated as non-stable APIs.
- Several code paths are intentionally partial right now (`SSM` kernel path, `SSM` length-aware path, `GatedDelta` kernel specialization, constrained `mrope` handling).

[Back to Documentation Index](index.md)
