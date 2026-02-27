# ONNX Compat Report

- Generated at: `2026-02-27T04:59:36Z`
- Command: `'bundle' 'exec' 'rake' 'test' 'TEST=test/onnx/*_test.rb' 'TESTOPTS=--name=/test_onnx_compat_report/'`
- Exit status: `0`
- Models: `105`
- Models with missing ops: `0`
- Unsupported op union size: `0`

## Test Summary

| Runs | Assertions | Failures | Errors | Skips |
| --- | --- | --- | --- | --- |
| 106 | 315 | 0 | 0 | 1 |

## Per-Model Coverage

| Model | Status | Supported/Total | Coverage % | Missing Ops |
| --- | --- | --- | --- | --- |
| Klear | PASS | 392/392 | 100.0 |  |
| afm7 | PASS | 245/245 | 100.0 |  |
| afmoe | PASS | 245/245 | 100.0 |  |
| apertus | PASS | 311/311 | 100.0 |  |
| baichuan_m1 | PASS | 307/307 | 100.0 |  |
| bailing_moe | PASS | 284/284 | 100.0 |  |
| bailing_moe_linear | PASS | 284/284 | 100.0 |  |
| bitnet | PASS | 574/574 | 100.0 |  |
| cohere | PASS | 267/267 | 100.0 |  |
| cohere2 | PASS | 261/261 | 100.0 |  |
| dbrx | PASS | 21/21 | 100.0 |  |
| deepseek | PASS | 288/288 | 100.0 |  |
| deepseek_v2 | PASS | 288/288 | 100.0 |  |
| deepseek_v3 | PASS | 288/288 | 100.0 |  |
| deepseek_v32 | PASS | 288/288 | 100.0 |  |
| dots1 | PASS | 331/331 | 100.0 |  |
| ernie4_5 | PASS | 267/267 | 100.0 |  |
| ernie4_5_moe | PASS | 267/267 | 100.0 |  |
| exaone | PASS | 255/255 | 100.0 |  |
| exaone4 | PASS | 293/293 | 100.0 |  |
| exaone_moe | PASS | 292/292 | 100.0 |  |
| falcon_h1 | PASS | 318/318 | 100.0 |  |
| gemma | PASS | 266/266 | 100.0 |  |
| gemma2 | PASS | 353/353 | 100.0 |  |
| gemma3 | PASS | 389/389 | 100.0 |  |
| gemma3_text | PASS | 389/389 | 100.0 |  |
| gemma3n | PASS | 353/353 | 100.0 |  |
| glm | PASS | 263/263 | 100.0 |  |
| glm4 | PASS | 307/307 | 100.0 |  |
| glm4_moe | PASS | 369/369 | 100.0 |  |
| glm4_moe_lite | PASS | 296/296 | 100.0 |  |
| glm_moe_dsa | PASS | 288/288 | 100.0 |  |
| gpt2 | PASS | 204/204 | 100.0 |  |
| gpt_bigcode | PASS | 197/197 | 100.0 |  |
| gpt_neox | PASS | 264/264 | 100.0 |  |
| gpt_oss | PASS | 326/326 | 100.0 |  |
| granite | PASS | 264/264 | 100.0 |  |
| granitemoe | PASS | 264/264 | 100.0 |  |
| granitemoehybrid | PASS | 318/318 | 100.0 |  |
| helium | PASS | 267/267 | 100.0 |  |
| hunyuan | PASS | 288/288 | 100.0 |  |
| hunyuan_v1_dense | PASS | 289/289 | 100.0 |  |
| internlm2 | PASS | 245/245 | 100.0 |  |
| internlm3 | PASS | 255/255 | 100.0 |  |
| iquestloopcoder | PASS | 546/546 | 100.0 |  |
| jamba | PASS | 318/318 | 100.0 |  |
| kimi_k25 | PASS | 288/288 | 100.0 |  |
| kimi_linear | PASS | 284/284 | 100.0 |  |
| kimi_vl | PASS | 288/288 | 100.0 |  |
| lfm2 | PASS | 293/293 | 100.0 |  |
| lfm2-vl | PASS | 293/293 | 100.0 |  |
| lfm2_moe | PASS | 401/401 | 100.0 |  |
| lille-130m | PASS | 269/269 | 100.0 |  |
| llama | PASS | 255/255 | 100.0 |  |
| llama4 | PASS | 350/350 | 100.0 |  |
| llama4_text | PASS | 297/297 | 100.0 |  |
| longcat_flash | PASS | 296/296 | 100.0 |  |
| longcat_flash_ngram | PASS | 296/296 | 100.0 |  |
| mamba | PASS | 224/224 | 100.0 |  |
| mamba2 | PASS | 351/351 | 100.0 |  |
| mimo | PASS | 261/261 | 100.0 |  |
| mimo_v2_flash | PASS | 310/310 | 100.0 |  |
| minicpm | PASS | 264/264 | 100.0 |  |
| minicpm3 | PASS | 335/335 | 100.0 |  |
| minimax | PASS | 345/345 | 100.0 |  |
| ministral3 | PASS | 529/529 | 100.0 |  |
| mistral3 | PASS | 267/267 | 100.0 |  |
| mixtral | PASS | 292/292 | 100.0 |  |
| nanochat | PASS | 297/297 | 100.0 |  |
| nemotron | PASS | 288/288 | 100.0 |  |
| nemotron-nas | PASS | 159/159 | 100.0 |  |
| nemotron_h | PASS | 318/318 | 100.0 |  |
| olmo | PASS | 251/251 | 100.0 |  |
| olmo2 | PASS | 293/293 | 100.0 |  |
| olmo3 | PASS | 585/585 | 100.0 |  |
| olmoe | PASS | 293/293 | 100.0 |  |
| openelm | PASS | 1501/1501 | 100.0 |  |
| phi | PASS | 279/279 | 100.0 |  |
| phi3 | PASS | 243/243 | 100.0 |  |
| phi3small | PASS | 303/303 | 100.0 |  |
| phimoe | PASS | 329/329 | 100.0 |  |
| pixtral | PASS | 267/267 | 100.0 |  |
| plamo | PASS | 247/247 | 100.0 |  |
| plamo2 | PASS | 318/318 | 100.0 |  |
| qwen | PASS | 247/247 | 100.0 |  |
| qwen2 | PASS | 261/261 | 100.0 |  |
| qwen2_moe | PASS | 350/350 | 100.0 |  |
| qwen2_vl | PASS | 261/261 | 100.0 |  |
| qwen3 | PASS | 293/293 | 100.0 |  |
| qwen3_5 | PASS | 293/293 | 100.0 |  |
| qwen3_5_moe | PASS | 293/293 | 100.0 |  |
| qwen3_moe | PASS | 336/336 | 100.0 |  |
| qwen3_next | PASS | 284/284 | 100.0 |  |
| qwen3_vl | PASS | 293/293 | 100.0 |  |
| qwen3_vl_moe | PASS | 336/336 | 100.0 |  |
| recurrent_gemma | PASS | 463/463 | 100.0 |  |
| rwkv7 | PASS | 312/312 | 100.0 |  |
| seed_oss | PASS | 255/255 | 100.0 |  |
| smollm3 | PASS | 255/255 | 100.0 |  |
| solar_open | PASS | 288/288 | 100.0 |  |
| stablelm | PASS | 289/289 | 100.0 |  |
| starcoder2 | PASS | 298/298 | 100.0 |  |
| step3p5 | PASS | 580/580 | 100.0 |  |
| telechat3 | PASS | 255/255 | 100.0 |  |
| youtu_llm | PASS | 331/331 | 100.0 |  |

## Unsupported Ops Union

none

## Unsupported Node Invocations

none
