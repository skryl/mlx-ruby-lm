# Loading and Tokenization

[Back to Documentation Index](index.md)

This page documents the current runtime loading and tokenizer flow implemented in `lib/mlx_lm/load_utils.rb`, `lib/mlx_lm/config.rb`, `lib/mlx_lm/weight_utils.rb`, and `lib/mlx_lm/tokenizer_utils.rb`.

## Primary APIs

```ruby
MlxLm::LoadUtils.load(model_path, tokenizer_config: nil) # => [model, tokenizer]
MlxLm::LoadUtils.load_model(model_path)                  # => [model, config]
MlxLm::LoadUtils.load_tokenizer(model_path)              # => TokenizerWrapper

MlxLm::Config.load(model_path)                           # => Hash
MlxLm::WeightUtils.load_sharded_safetensors(directory)   # => Hash<String, MLX::Core::Array>

MlxLm::TokenizerWrapper.new(path_or_tokenizer, eos_token: nil, eos_token_id: nil)
MlxLm::StreamingDetokenizer.new(tokenizer_wrapper)
```

## Model Loading Flow

`MlxLm::LoadUtils.load` does:

1. `load_model(model_path)`
2. `load_tokenizer(model_path)`
3. Returns `[model, tokenizer]`

`load_model(model_path)` flow:

1. Loads config via `MlxLm::Config.load(model_path)`.
2. Resolves classes via `MlxLm::Models.get_classes(config)`.
3. Builds args via `args_class.from_dict(config)` (unknown keys are ignored; defaults are applied from `BaseModelArgs` fields).
4. Instantiates model with `model_class.new(model_args)`.
5. Loads weights via `MlxLm::WeightUtils.load_sharded_safetensors(model_path)`.
6. If model responds to `sanitize`, applies `weights = model.sanitize(weights)`.
7. If `config["quantization"]` exists, calls:
   `MlxLm::Quantize.quantize_model(model, group_size:, bits:, weights:)`
   with defaults `group_size = 64`, `bits = 4` when missing.
8. Loads weights with `model.load_weights(weights, strict: false)`.
9. Returns `[model, config]`.

## Config Behavior

`MlxLm::Config.load(model_path)`:

- Requires `config.json`.
- Optionally reads `generation_config.json`.
- If `generation_config.json` is present and has `eos_token_id`, it overwrites `config["eos_token_id"]`.
- If `generation_config.json` is invalid JSON, it is ignored (falls back to `{}`).
- Preserves all other config fields as-is.

## Weights Behavior

`MlxLm::WeightUtils.load_sharded_safetensors(directory)`:

- Loads `model*.safetensors` files in sorted order.
- Raises if no matching safetensor file exists.
- Merges all tensors into one hash keyed by tensor name.

`MlxLm::WeightUtils.load_safetensors(path)` converts serialized tensor blobs to `MLX::Core::Array` values. It has explicit unpack logic for `F32`, `F16`, `BF16`, `I32`, `I64`, `U8`; unrecognized dtypes fall back to float32 unpacking.

## Tokenizer Loading Flow

`MlxLm::LoadUtils.load_tokenizer(model_path)`:

1. Requires `tokenizer.json` in the model directory.
2. Creates a raw tokenizer via `Tokenizers::Tokenizer.from_file`.
3. Reads `tokenizer_config.json` (if present) and extracts `eos_token` (string or `{"content": ...}` form).
4. Reads `config.json` (if present) and extracts `eos_token_id`.
5. Normalizes `eos_token_id` to an array and passes it as override.
6. Returns `TokenizerWrapper.new(tokenizer, eos_token:, eos_token_id:)`.

## TokenizerWrapper and StreamingDetokenizer

`TokenizerWrapper` supports:

- `encode(text, add_special_tokens: true) -> Array<Integer>`
- `decode(ids, skip_special_tokens: false) -> String`
- `eos_token`, `eos_token_id`, `eos_token_ids`
- `bos_token`, `bos_token_id`
- `vocab_size`, `id_to_token`, `token_to_id`
- `detokenizer` (memoized `StreamingDetokenizer`)
- `has_chat_template`

EOS ID resolution order in `TokenizerWrapper#eos_token_id`:

1. First element of `eos_token_id` override (if provided).
2. ID looked up from `tokenizer_config.json` `eos_token`.
3. ID looked up from explicit `eos_token` override.
4. `nil` if none resolve.

`TokenizerWrapper#eos_token_ids` returns a `Set` containing the override IDs (if any) plus the single resolved `eos_token_id`.

`StreamingDetokenizer` incrementally emits text via:

- `add_token(token_id)` (returns the newly added segment)
- `finalize` (returns remaining segment)
- `text` (full accumulated text)
- `reset`

It decodes the full token buffer each step and returns only the new suffix (`current_text[@prev_text.length..]`).

## Caveats

- `load(model_path, tokenizer_config: nil)` accepts `tokenizer_config:` but does not use it.
- Loading is local-directory based; there is no built-in Hugging Face download/resolve path in `LoadUtils`.
- `load` returns `[model, tokenizer]` only; use `load_model` if you also need the merged config hash.
- `model.load_weights(..., strict: false)` allows partial/non-strict weight loading.
- `TokenizerWrapper.new(path)` reads `tokenizer.json` + `tokenizer_config.json`, but does not read `config.json` for `eos_token_id` unless you use `LoadUtils.load_tokenizer`.

[Back to Documentation Index](index.md)
