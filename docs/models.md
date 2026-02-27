# Models

[Back to Documentation Index](index.md)

## Registry Source

Model loading resolves architecture keys through:

- `MlxLm::Models::REGISTRY`
- `MlxLm::Models::REMAPPING`
- `MlxLm::Models.get_classes(config)`

`LoadUtils.load_model` reads `config["model_type"]`, applies remapping, then loads the
registered `[ModelClass, ModelArgsClass]`.

## Remapping Notes

Current remapping entries (`MlxLm::Models::REMAPPING`):

- `mistral` -> `llama`
- `falcon_mamba` -> `mamba`

If a config uses a remapped key, the canonical key above is used for registry lookup.

## Runtime Registry Snapshot

Generated from:

```bash
bundle exec ruby -Ilib -e 'require "mlx"; require "mlx_lm"; keys = MlxLm::Models::REGISTRY.keys.sort; puts "COUNT=#{keys.size}"; puts keys'
```

Current count: `106`

## Full Sorted Registry Keys

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

## Related Docs

- [Documentation Index](index.md)
- [CLI Usage](cli.md)
- [Ruby APIs](ruby-apis.md)
