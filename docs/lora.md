# LoRA

[Back to Documentation Index](index.md)

This page documents the current LoRA surfaces in `lib/mlx_lm/tuner/lora.rb`.

## Module and Defaults

- Module: `MlxLm::Tuner`
- Default target keys (`DEFAULT_LORA_KEYS`):
  - `self_attn.q_proj`
  - `self_attn.k_proj`
  - `self_attn.v_proj`

## `LoRALinear`

Class: `MlxLm::Tuner::LoRALinear < MLX::NN::Module`

Constructor path:

```ruby
lora = MlxLm::Tuner::LoRALinear.from_base(
  linear,
  r: 8,
  dropout: 0.0,
  scale: 20.0
)
```

Behavior:

- Wraps a base linear layer and adds LoRA matrices:
  - `lora_a`: random uniform `[input_dims, r]`
  - `lora_b`: zero-initialized `[r, output_dims]`
- Forward:
  - `linear(x) + scale * (dropout(x) @ lora_a @ lora_b)`
- `from_base` accepts both `MLX::NN::Linear` and `MLX::NN::QuantizedLinear`.
- Because `lora_b` starts at zeros, initial output matches base output.

### `LoRALinear#fuse`

```ruby
fused_linear = lora.fuse(dequantize: false)
```

Behavior:

- Returns a new `MLX::NN::Linear`.
- Fuses with: `W' = W + scale * (lora_a @ lora_b)^T`.
- Copies base bias when present.
- If `dequantize: true` and base is quantized, it first dequantizes via `MlxLm::Quantize.linear_from_quantized`.

## `LoRAEmbedding`

Class: `MlxLm::Tuner::LoRAEmbedding < MLX::NN::Module`

Constructor path:

```ruby
lora_embed = MlxLm::Tuner::LoRAEmbedding.from_base(
  embedding,
  r: 8,
  dropout: 0.0,
  scale: 20.0
)
```

Behavior:

- Adds LoRA matrices:
  - `lora_a`: random uniform `[num_embeddings, r]`
  - `lora_b`: zero-initialized `[r, dims]`
- `call(x)`:
  - base embedding lookup plus LoRA delta from `take(lora_a, x, 0) @ lora_b`
- `as_linear(x)` is also implemented.

### `LoRAEmbedding#fuse`

```ruby
fused_embedding = lora_embed.fuse(dequantize: false)
```

Behavior:

- Returns a new `MLX::NN::Embedding` with fused weights.
- If `dequantize: true` and base is quantized, it first dequantizes via `MlxLm::Quantize.embedding_from_quantized`.

## `apply_lora_layers`

```ruby
MlxLm::Tuner.apply_lora_layers(model, num_layers: nil, config: {})
```

Config keys (string or symbol):

- `rank` (default `8`)
- `scale` (default `20.0`)
- `dropout` (default `0.0`)
- `keys` (default `DEFAULT_LORA_KEYS`)

Behavior:

- Requires `model.layers`.
- Targets `layers.last(num_layers)` (all layers if `num_layers` is nil).
- Recursively walks each target layer via `mod.state`.
- Replaces matching `MLX::NN::Linear` with `LoRALinear`.
- Replaces matching `MLX::NN::Embedding` with `LoRAEmbedding`.
- Key match rule is permissive: `full_key.end_with?(k) || full_key.include?(k)`.
- Mutates model layers in place.

## Caveats

- `apply_lora_layers` only checks `MLX::NN::Linear` and `MLX::NN::Embedding`; quantized linear/embedding layers are not matched by this helper.
- Key matching uses substring inclusion (`include?`), so broad key strings can match more modules than intended.
- `fuse` returns dense (`Linear`/`Embedding`) modules, not quantized modules.
- When the wrapped base layer is quantized, `fuse(dequantize: true)` is the safe path before adding LoRA deltas to weights.

[Back to Documentation Index](index.md)
