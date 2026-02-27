# mlx-ruby-lm

Ruby LLM inference toolkit built on the `mlx` gem.

## Included tools

### CLI

Executable: `mlx_lm`

Commands:

- `mlx_lm generate`
- `mlx_lm chat`
- `mlx_lm server`

Example:

```bash
mlx_lm generate --model /path/to/model --prompt "Hello"
```

### Ruby APIs

- `MlxLm::LoadUtils`: load model weights/config/tokenizer from model directory.
- `MlxLm::Generate`: token generation (`generate`, `stream_generate`, `generate_step`).
- `MlxLm::SampleUtils`: samplers and logits processors (`top_p`, `top_k`, repetition penalty).
- `MlxLm::ChatTemplate`: default/chatml prompt formatting.
- `MlxLm::Server`: OpenAI-compatible chat completion server (`/v1/models`, `/v1/chat/completions`).
- `MlxLm::Quantize`: model quantization/dequantization helpers.
- `MlxLm::Perplexity`: perplexity/log-likelihood helpers.
- `MlxLm::Benchmark`: simple generation throughput and model stats helpers.
- `MlxLm::Tuner`: LoRA adapters (`LoRALinear`, `LoRAEmbedding`, `apply_lora_layers`).
- `MlxLm::ConvertUtils`: dtype conversion and parameter/size utilities.

Minimal usage:

```ruby
require "mlx"
require "mlx_lm"

model, tokenizer = MlxLm::LoadUtils.load("/path/to/model")
text = MlxLm::Generate.generate(model, tokenizer, "Hello", max_tokens: 64)
puts text
```

## Included models

Current registry includes 106 `model_type` values.

Families covered include:

- Llama/Gemma/Qwen/Phi
- Mistral/Mixtral/Granite/Cohere
- DeepSeek/GLM/InternLM/Kimi
- Mamba/RWKV/Recurrent Gemma
- MoE variants (for example `*_moe`, `mixtral`, `jamba`, `granitemoe*`)
- Vision-language variants (for example `qwen*_vl`, `kimi_vl`, `pixtral`, `lfm2-vl`)

Registered `model_type` values:

```text
Klear
afm7
afmoe
apertus
baichuan_m1
bailing_moe
bailing_moe_linear
bitnet
cohere
cohere2
dbrx
deepseek
deepseek_v2
deepseek_v3
deepseek_v32
dots1
ernie4_5
ernie4_5_moe
exaone
exaone4
exaone_moe
falcon_h1
gemma
gemma2
gemma3
gemma3_text
gemma3n
glm
glm4
glm4_moe
glm4_moe_lite
glm_moe_dsa
gpt2
gpt_bigcode
gpt_neox
gpt_oss
granite
granitemoe
granitemoehybrid
helium
hunyuan
hunyuan_v1_dense
internlm2
internlm3
iquestloopcoder
jamba
kimi_k25
kimi_linear
kimi_vl
lfm2
lfm2-vl
lfm2_moe
lille-130m
llama
llama4
llama4_text
longcat_flash
longcat_flash_ngram
mamba
mamba2
mimo
mimo_v2_flash
minicpm
minicpm3
minimax
ministral3
mistral3
mixtral
nanochat
nemotron
nemotron-nas
nemotron_h
olmo
olmo2
olmo3
olmoe
openelm
phi
phi3
phi3small
phimoe
phixtral
pixtral
plamo
plamo2
qwen
qwen2
qwen2_moe
qwen2_vl
qwen3
qwen3_5
qwen3_5_moe
qwen3_moe
qwen3_next
qwen3_vl
qwen3_vl_moe
recurrent_gemma
rwkv7
seed_oss
smollm3
solar_open
stablelm
starcoder2
step3p5
telechat3
youtu_llm
```

To inspect the current registry from Ruby:

```ruby
require "mlx_lm"
puts MlxLm::Models::REGISTRY.keys.sort
```
