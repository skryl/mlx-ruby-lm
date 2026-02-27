# Quantization

[Back to Documentation Index](index.md)

This page covers the current `MlxLm::Quantize` behavior in `lib/mlx_lm/quantize.rb`.

## `MlxLm::Quantize.quantize_model`

```ruby
MlxLm::Quantize.quantize_model(model, group_size: 64, bits: 4, weights: nil)
# => { "group_size" => 64, "bits" => 4 }
```

Behavior:

- Uses `MLX::NN.quantize` internally.
- Quantizes modules where `mod.respond_to?(:to_quantized)`.
- If `weights:` is provided, quantization is filtered to paths where `weights.key?("#{path}.scales")` is true.
- Mutates `model` in place and returns a quantization config hash.

Current usage in load path:

- `MlxLm::LoadUtils.load_model` checks `config["quantization"]`.
- If present, it calls `quantize_model` before `model.load_weights(...)`.

## `MlxLm::Quantize.dequantize_model`

```ruby
MlxLm::Quantize.dequantize_model(model)
# => model
```

Behavior:

- Recursively walks module instance variables.
- Replaces:
  - `MLX::NN::QuantizedLinear` -> `MLX::NN::Linear`
  - `MLX::NN::QuantizedEmbedding` -> `MLX::NN::Embedding`
- Also traverses arrays of submodules.
- Mutates and returns the same model object.

Conversion helpers used by dequantization:

- `MlxLm::Quantize.linear_from_quantized(qlinear)`
- `MlxLm::Quantize.embedding_from_quantized(qembed)`

Both call `MLX::Core.dequantize(...)` with each layer's stored `group_size`/`bits`.

## `MlxLm::Quantize.bits_per_weight`

```ruby
MlxLm::Quantize.bits_per_weight(model)
# => Float
```

Behavior:

- Iterates `model.named_modules`.
- Counts parameters only from:
  - `MLX::NN::Linear` (assumed 32 bits)
  - `MLX::NN::QuantizedLinear` (assumed 4 bits)
- Returns `total_bits / total_params` as a float.

## Related Size Utility

- `MlxLm::ConvertUtils.model_size_bytes(model)` (documented in [Evaluation and Conversion Utilities](evaluation-conversion.md)) estimates total parameter bytes by dtype.

## Caveats

- `weights:` filtering depends on exact module path names matching `"<path>.scales"` keys in your loaded weight map.
- `dequantize_model` only handles quantized linear/embedding modules; other quantized module types are not dequantized here.
- `bits_per_weight` is approximate:
  - hardcodes quantized linear cost as 4 bits (does not read per-layer `bits`)
  - ignores embeddings and other parameterized modules
  - ignores quantization metadata overhead (scales/biases packing details)

[Back to Documentation Index](index.md)
