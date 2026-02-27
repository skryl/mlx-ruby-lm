# Python `mlx-lm` -> Ruby `mlx-ruby-lm` Class Parity Checklist

**Status:** Active
**Date:** 2026-02-26
**Scope:** Full Python class inventory from `mlx-lm/mlx_lm/**/*.py`
**Ruby surface:** `lib/mlx_lm/**/*.rb`

## Summary

| Metric | Count |
|---|---:|
| Python classes discovered | 768 |
| Implemented | 527 |
| Partial | 221 |
| Missing | 20 |

## Status Rules

- `Implemented`: class is present in the expected Ruby file (or uniquely implemented in a different Ruby file with a note).
- `Partial`: Ruby file exists for that Python module, but the class is absent/renamed/merged.
- `Missing`: no Ruby class/file counterpart found for that Python class.

## Full Class Inventory

| Python File | Line | Python Class | Ruby Status | Ruby Reference | Notes |
|---|---:|---|---|---|---|
| evaluate.py | 72 | MLXLM | Missing | - |  |
| generate.py | 266 | GenerationResponse | Partial | generate.rb | Ruby file exists but defines no classes |
| generate.py | 804 | BatchStats | Partial | generate.rb | Ruby file exists but defines no classes |
| generate.py | 828 | BatchResponse | Partial | generate.rb | Ruby file exists but defines no classes |
| generate.py | 843 | Batch | Partial | generate.rb | Ruby file exists but defines no classes |
| generate.py | 930 | BatchGenerator | Partial | generate.rb | Ruby file exists but defines no classes |
| generate.py | 932 | Response | Partial | generate.rb | Ruby file exists but defines no classes |
| gguf.py | 10 | TokenType | Missing | - |  |
| gguf.py | 19 | GGMLFileType | Missing | - |  |
| gguf.py | 24 | HfVocab | Missing | - |  |
| models/Klear.py | 15 | ModelArgs | Implemented | models/klear.rb |  |
| models/Klear.py | 36 | KlearAttention | Implemented | models/klear.rb |  |
| models/Klear.py | 110 | KlearMLP | Implemented | models/klear.rb |  |
| models/Klear.py | 121 | KlearSparseMoeBlock | Implemented | models/klear.rb |  |
| models/Klear.py | 156 | KlearDecoderLayer | Implemented | models/klear.rb |  |
| models/Klear.py | 186 | KlearModel | Implemented | models/klear.rb |  |
| models/Klear.py | 214 | Model | Implemented | models/klear.rb |  |
| models/activations.py | 25 | XieLU | Implemented | models/activations.rb |  |
| models/afm7.py | 19 | ModelArgs | Implemented | models/afm7.rb |  |
| models/afm7.py | 32 | FusedLoRALinear | Partial | models/afm7.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/afm7.py | 96 | FusedQuantizedLinear | Partial | models/afm7.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/afm7.py | 123 | FusedLinear | Partial | models/afm7.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/afm7.py | 165 | Attention | Partial | models/afm7.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/afm7.py | 226 | KVReuseAttention | Partial | models/afm7.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/afm7.py | 266 | MLP | Partial | models/afm7.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/afm7.py | 283 | TransformerBlock | Partial | models/afm7.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/afm7.py | 306 | KVReuseTransformerBlock | Partial | models/afm7.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/afm7.py | 330 | AFMModel | Partial | models/afm7.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/afm7.py | 369 | Model | Implemented | models/afm7.rb |  |
| models/afmoe.py | 18 | ModelArgs | Implemented | models/afmoe.rb |  |
| models/afmoe.py | 48 | Attention | Implemented | models/afmoe.rb |  |
| models/afmoe.py | 137 | MLP | Implemented | models/afmoe.rb |  |
| models/afmoe.py | 156 | MoERouter | Implemented | models/afmoe.rb |  |
| models/afmoe.py | 167 | AfmoeMoE | Implemented | models/afmoe.rb |  |
| models/afmoe.py | 242 | DecoderLayer | Implemented | models/afmoe.rb |  |
| models/afmoe.py | 278 | AfmoeModel | Implemented | models/afmoe.rb |  |
| models/afmoe.py | 332 | Model | Implemented | models/afmoe.rb |  |
| models/apertus.py | 16 | ModelArgs | Implemented | models/apertus.rb |  |
| models/apertus.py | 36 | ApertusMLP | Implemented | models/apertus.rb |  |
| models/apertus.py | 51 | ApertusAttention | Implemented | models/apertus.rb |  |
| models/apertus.py | 117 | ApertusDecoderLayer | Implemented | models/apertus.rb |  |
| models/apertus.py | 137 | ApertusModel | Implemented | models/apertus.rb |  |
| models/apertus.py | 164 | Model | Implemented | models/apertus.rb |  |
| models/baichuan_m1.py | 15 | ModelArgs | Implemented | models/baichuan_m1.rb |  |
| models/baichuan_m1.py | 33 | Attention | Implemented | models/baichuan_m1.rb |  |
| models/baichuan_m1.py | 130 | MLP | Implemented | models/baichuan_m1.rb |  |
| models/baichuan_m1.py | 147 | DecoderLayer | Implemented | models/baichuan_m1.rb |  |
| models/baichuan_m1.py | 166 | BaichuanModel | Implemented | models/baichuan_m1.rb |  |
| models/baichuan_m1.py | 212 | Model | Implemented | models/baichuan_m1.rb |  |
| models/bailing_moe.py | 17 | ModelArgs | Implemented | models/bailing_moe.rb |  |
| models/bailing_moe.py | 60 | BailingMoeMLP | Implemented | models/bailing_moe.rb |  |
| models/bailing_moe.py | 83 | BailingMoeAttention | Implemented | models/bailing_moe.rb |  |
| models/bailing_moe.py | 202 | BailingMoeGate | Implemented | models/bailing_moe.rb |  |
| models/bailing_moe.py | 234 | BailingMoeSparseMoeBlock | Implemented | models/bailing_moe.rb |  |
| models/bailing_moe.py | 267 | BailingMoeDecoderLayer | Implemented | models/bailing_moe.rb |  |
| models/bailing_moe.py | 296 | BailingMoeModel | Implemented | models/bailing_moe.rb |  |
| models/bailing_moe.py | 324 | Model | Implemented | models/bailing_moe.rb |  |
| models/bailing_moe_linear.py | 23 | ModelArgs | Implemented | models/bailing_moe_linear.rb |  |
| models/bailing_moe_linear.py | 100 | GroupRMSNorm | Partial | models/bailing_moe_linear.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/bailing_moe_linear.py | 114 | MLP | Partial | models/bailing_moe_linear.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/bailing_moe_linear.py | 137 | Attention | Partial | models/bailing_moe_linear.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/bailing_moe_linear.py | 213 | LinearAttention | Partial | models/bailing_moe_linear.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/bailing_moe_linear.py | 368 | Gate | Partial | models/bailing_moe_linear.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/bailing_moe_linear.py | 400 | SparseMoeBlock | Partial | models/bailing_moe_linear.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/bailing_moe_linear.py | 433 | DecoderLayer | Partial | models/bailing_moe_linear.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/bailing_moe_linear.py | 475 | LanguageModel | Partial | models/bailing_moe_linear.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/bailing_moe_linear.py | 509 | Model | Implemented | models/bailing_moe_linear.rb |  |
| models/base.py | 12 | BaseModelArgs | Implemented | model_args.rb | Implemented in different Ruby file |
| models/bitlinear_layers.py | 92 | BitLinear | Implemented | models/bitlinear_layers.rb |  |
| models/bitnet.py | 16 | ModelArgs | Implemented | models/bitnet.rb |  |
| models/bitnet.py | 35 | Attention | Implemented | models/bitnet.rb |  |
| models/bitnet.py | 96 | MLP | Implemented | models/bitnet.rb |  |
| models/bitnet.py | 120 | TransformerBlock | Implemented | models/bitnet.rb |  |
| models/bitnet.py | 146 | LlamaModel | Partial | models/bitnet.rb | Ruby file exists; classes differ: ModelArgs, Attention, MLP, TransformerBlock, BitnetModel, Model |
| models/bitnet.py | 176 | Model | Implemented | models/bitnet.rb |  |
| models/cache.py | 125 | _BaseCache | Partial | models/cache.rb | Ruby file exists; classes differ: BaseCache, KVCache, RotatingKVCache, QuantizedKVCache, ArraysCache, ChunkedKVCache, CacheList |
| models/cache.py | 176 | ConcatenateKVCache | Partial | models/cache.rb | Ruby file exists; classes differ: BaseCache, KVCache, RotatingKVCache, QuantizedKVCache, ArraysCache, ChunkedKVCache, CacheList |
| models/cache.py | 230 | QuantizedKVCache | Implemented | models/cache.rb |  |
| models/cache.py | 323 | KVCache | Implemented | models/cache.rb |  |
| models/cache.py | 408 | RotatingKVCache | Implemented | models/cache.rb |  |
| models/cache.py | 592 | ArraysCache | Implemented | models/cache.rb |  |
| models/cache.py | 682 | ChunkedKVCache | Implemented | models/cache.rb |  |
| models/cache.py | 765 | CacheList | Implemented | models/cache.rb |  |
| models/cache.py | 863 | BatchKVCache | Partial | models/cache.rb | Ruby file exists; classes differ: BaseCache, KVCache, RotatingKVCache, QuantizedKVCache, ArraysCache, ChunkedKVCache, CacheList |
| models/cache.py | 1058 | BatchRotatingKVCache | Partial | models/cache.rb | Ruby file exists; classes differ: BaseCache, KVCache, RotatingKVCache, QuantizedKVCache, ArraysCache, ChunkedKVCache, CacheList |
| models/cohere.py | 14 | ModelArgs | Implemented | models/cohere.rb |  |
| models/cohere.py | 30 | LayerNorm2D | Partial | models/cohere.rb | Ruby file exists; classes differ: ModelArgs, Attention, MLP, TransformerBlock, CohereModel, Model |
| models/cohere.py | 41 | Attention | Implemented | models/cohere.rb |  |
| models/cohere.py | 105 | MLP | Implemented | models/cohere.rb |  |
| models/cohere.py | 116 | TransformerBlock | Implemented | models/cohere.rb |  |
| models/cohere.py | 141 | CohereModel | Implemented | models/cohere.rb |  |
| models/cohere.py | 174 | Model | Implemented | models/cohere.rb |  |
| models/cohere2.py | 15 | ModelArgs | Implemented | models/cohere2.rb |  |
| models/cohere2.py | 33 | Attention | Implemented | models/cohere2.rb |  |
| models/cohere2.py | 102 | MLP | Implemented | models/cohere2.rb |  |
| models/cohere2.py | 113 | TransformerBlock | Implemented | models/cohere2.rb |  |
| models/cohere2.py | 140 | CohereModel | Partial | models/cohere2.rb | Ruby file exists; classes differ: ModelArgs, Attention, MLP, TransformerBlock, Cohere2Model, Model |
| models/cohere2.py | 184 | Model | Implemented | models/cohere2.rb |  |
| models/dbrx.py | 15 | ModelArgs | Implemented | models/dbrx.rb |  |
| models/dbrx.py | 25 | Attention | Implemented | models/dbrx.rb |  |
| models/dbrx.py | 85 | NormAttnNorm | Implemented | models/dbrx.rb |  |
| models/dbrx.py | 103 | MLP | Implemented | models/dbrx.rb |  |
| models/dbrx.py | 116 | Router | Implemented | models/dbrx.rb |  |
| models/dbrx.py | 125 | SparseMoeBlock | Implemented | models/dbrx.rb |  |
| models/dbrx.py | 172 | DecoderLayer | Implemented | models/dbrx.rb |  |
| models/dbrx.py | 189 | DBRX | Partial | models/dbrx.rb | Ruby file exists; classes differ: ModelArgs, Attention, NormAttnNorm, MLP, Router, SparseMoeBlock, DecoderLayer, DbrxModel, Model |
| models/dbrx.py | 215 | Model | Implemented | models/dbrx.rb |  |
| models/deepseek.py | 13 | ModelArgs | Implemented | models/deepseek.rb |  |
| models/deepseek.py | 34 | DeepseekAttention | Partial | models/deepseek.rb | Ruby file exists; classes differ: ModelArgs, Attention, DeepseekMLP, MoEGate, DeepseekMoE, DecoderLayer, DeepseekModel, Model |
| models/deepseek.py | 108 | DeepseekMLP | Implemented | models/deepseek.rb |  |
| models/deepseek.py | 127 | MoEGate | Implemented | models/deepseek.rb |  |
| models/deepseek.py | 144 | DeepseekMoE | Implemented | models/deepseek.rb |  |
| models/deepseek.py | 169 | DeepseekDecoderLayer | Partial | models/deepseek.rb | Ruby file exists; classes differ: ModelArgs, Attention, DeepseekMLP, MoEGate, DeepseekMoE, DecoderLayer, DeepseekModel, Model |
| models/deepseek.py | 200 | DeepseekModel | Implemented | models/deepseek.rb |  |
| models/deepseek.py | 228 | Model | Implemented | models/deepseek.rb |  |
| models/deepseek_v2.py | 18 | ModelArgs | Implemented | models/deepseek_v2.rb |  |
| models/deepseek_v2.py | 82 | DeepseekV2YarnRotaryEmbedding | Partial | models/deepseek_v2.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/deepseek_v2.py | 129 | DeepseekV2Attention | Partial | models/deepseek_v2.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/deepseek_v2.py | 248 | DeepseekV2MLP | Partial | models/deepseek_v2.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/deepseek_v2.py | 268 | MoEGate | Partial | models/deepseek_v2.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/deepseek_v2.py | 304 | DeepseekV2MoE | Partial | models/deepseek_v2.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/deepseek_v2.py | 338 | DeepseekV2DecoderLayer | Partial | models/deepseek_v2.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/deepseek_v2.py | 369 | DeepseekV2Model | Partial | models/deepseek_v2.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/deepseek_v2.py | 414 | Model | Implemented | models/deepseek_v2.rb |  |
| models/deepseek_v3.py | 21 | ModelArgs | Implemented | models/deepseek_v3.rb |  |
| models/deepseek_v3.py | 53 | DeepseekV3Attention | Partial | models/deepseek_v3.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/deepseek_v3.py | 172 | DeepseekV3MLP | Partial | models/deepseek_v3.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/deepseek_v3.py | 227 | MoEGate | Partial | models/deepseek_v3.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/deepseek_v3.py | 253 | DeepseekV3MoE | Partial | models/deepseek_v3.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/deepseek_v3.py | 289 | DeepseekV3DecoderLayer | Partial | models/deepseek_v3.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/deepseek_v3.py | 319 | DeepseekV3Model | Partial | models/deepseek_v3.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/deepseek_v3.py | 364 | Model | Implemented | models/deepseek_v3.rb |  |
| models/deepseek_v32.py | 20 | ModelArgs | Implemented | models/deepseek_v32.rb |  |
| models/deepseek_v32.py | 55 | Indexer | Partial | models/deepseek_v32.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/deepseek_v32.py | 120 | DeepseekV32Attention | Partial | models/deepseek_v32.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/deepseek_v32.py | 265 | DeepseekV32MLP | Partial | models/deepseek_v32.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/deepseek_v32.py | 320 | MoEGate | Partial | models/deepseek_v32.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/deepseek_v32.py | 346 | DeepseekV32MoE | Partial | models/deepseek_v32.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/deepseek_v32.py | 382 | DeepseekV32DecoderLayer | Partial | models/deepseek_v32.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/deepseek_v32.py | 412 | DeepseekV32Model | Partial | models/deepseek_v32.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/deepseek_v32.py | 481 | Model | Implemented | models/deepseek_v32.rb |  |
| models/dots1.py | 17 | ModelArgs | Implemented | models/dots1.rb |  |
| models/dots1.py | 45 | Dots1Attention | Implemented | models/dots1.rb |  |
| models/dots1.py | 138 | Dots1TopkRouter | Implemented | models/dots1.rb |  |
| models/dots1.py | 162 | Dots1MLP | Implemented | models/dots1.rb |  |
| models/dots1.py | 187 | Dots1MoE | Implemented | models/dots1.rb |  |
| models/dots1.py | 216 | Dots1DecoderLayer | Implemented | models/dots1.rb |  |
| models/dots1.py | 243 | Dots1Model | Implemented | models/dots1.rb |  |
| models/dots1.py | 271 | Model | Implemented | models/dots1.rb |  |
| models/ernie4_5.py | 15 | ModelArgs | Implemented | models/ernie4_5.rb |  |
| models/ernie4_5.py | 31 | Attention | Implemented | models/ernie4_5.rb |  |
| models/ernie4_5.py | 83 | MLP | Implemented | models/ernie4_5.rb |  |
| models/ernie4_5.py | 94 | DecoderLayer | Implemented | models/ernie4_5.rb |  |
| models/ernie4_5.py | 117 | Ernie45Model | Implemented | models/ernie4_5.rb |  |
| models/ernie4_5.py | 142 | Model | Implemented | models/ernie4_5.rb |  |
| models/ernie4_5_moe.py | 16 | ModelArgs | Implemented | models/ernie4_5_moe.rb |  |
| models/ernie4_5_moe.py | 42 | Attention | Partial | models/ernie4_5_moe.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/ernie4_5_moe.py | 94 | Ernie4_5_MLP | Partial | models/ernie4_5_moe.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/ernie4_5_moe.py | 105 | Ernie4_5_MoeMLP | Partial | models/ernie4_5_moe.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/ernie4_5_moe.py | 163 | Ernie4_5_DecoderLayer | Partial | models/ernie4_5_moe.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/ernie4_5_moe.py | 211 | Ernie45Model | Partial | models/ernie4_5_moe.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/ernie4_5_moe.py | 238 | Model | Implemented | models/ernie4_5_moe.rb |  |
| models/exaone.py | 15 | ModelArgs | Implemented | models/exaone.rb |  |
| models/exaone.py | 34 | AttentionModule | Implemented | models/exaone.rb |  |
| models/exaone.py | 79 | Attention | Implemented | models/exaone.rb |  |
| models/exaone.py | 85 | MLP | Implemented | models/exaone.rb |  |
| models/exaone.py | 98 | TransformerBlock | Implemented | models/exaone.rb |  |
| models/exaone.py | 117 | ExaoneModel | Implemented | models/exaone.rb |  |
| models/exaone.py | 142 | Model | Implemented | models/exaone.rb |  |
| models/exaone4.py | 16 | ModelArgs | Implemented | models/exaone4.rb |  |
| models/exaone4.py | 34 | Attention | Implemented | models/exaone4.rb |  |
| models/exaone4.py | 98 | MLP | Implemented | models/exaone4.rb |  |
| models/exaone4.py | 109 | TransformerBlock | Implemented | models/exaone4.rb |  |
| models/exaone4.py | 137 | ExaoneModel | Implemented | models/exaone4.rb |  |
| models/exaone4.py | 187 | Model | Implemented | models/exaone4.rb |  |
| models/exaone_moe.py | 18 | ModelArgs | Implemented | models/exaone_moe.rb |  |
| models/exaone_moe.py | 88 | MoEGate | Implemented | models/exaone_moe.rb |  |
| models/exaone_moe.py | 113 | MLP | Implemented | models/exaone_moe.rb |  |
| models/exaone_moe.py | 126 | MoE | Implemented | models/exaone_moe.rb |  |
| models/exaone_moe.py | 164 | Attention | Implemented | models/exaone_moe.rb |  |
| models/exaone_moe.py | 237 | DecoderLayer | Implemented | models/exaone_moe.rb |  |
| models/exaone_moe.py | 262 | ExaoneMoEModel | Implemented | models/exaone_moe.rb | Name variant in Ruby: ExaoneMoeModel |
| models/exaone_moe.py | 307 | Model | Implemented | models/exaone_moe.rb |  |
| models/falcon_h1.py | 22 | ModelArgs | Implemented | models/falcon_h1.rb |  |
| models/falcon_h1.py | 76 | FalconH1RMSNormGated | Partial | models/falcon_h1.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/falcon_h1.py | 116 | FalconH1Attention | Partial | models/falcon_h1.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/falcon_h1.py | 179 | FalconH1Mixer | Partial | models/falcon_h1.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/falcon_h1.py | 339 | FalconH1MLP | Partial | models/falcon_h1.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/falcon_h1.py | 357 | FalconH1DecoderLayer | Partial | models/falcon_h1.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/falcon_h1.py | 402 | FalconH1Model | Partial | models/falcon_h1.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/falcon_h1.py | 441 | Model | Implemented | models/falcon_h1.rb |  |
| models/gemma.py | 13 | ModelArgs | Implemented | models/gemma.rb |  |
| models/gemma.py | 27 | RMSNorm | Partial | models/gemma.rb | Ruby file exists; classes differ: ModelArgs, Attention, MLP, TransformerBlock, GemmaModel, Model |
| models/gemma.py | 37 | Attention | Implemented | models/gemma.rb |  |
| models/gemma.py | 90 | MLP | Implemented | models/gemma.rb |  |
| models/gemma.py | 101 | TransformerBlock | Implemented | models/gemma.rb |  |
| models/gemma.py | 125 | GemmaModel | Implemented | models/gemma.rb |  |
| models/gemma.py | 157 | Model | Implemented | models/gemma.rb |  |
| models/gemma2.py | 13 | ModelArgs | Implemented | models/gemma2.rb |  |
| models/gemma2.py | 30 | RMSNorm | Partial | models/gemma2.rb | Ruby file exists; classes differ: ModelArgs, Gemma2RMSNorm, Attention, MLP, TransformerBlock, Gemma2Model, Model |
| models/gemma2.py | 40 | Attention | Implemented | models/gemma2.rb |  |
| models/gemma2.py | 111 | MLP | Implemented | models/gemma2.rb |  |
| models/gemma2.py | 122 | TransformerBlock | Implemented | models/gemma2.rb |  |
| models/gemma2.py | 152 | GemmaModel | Partial | models/gemma2.rb | Ruby file exists; classes differ: ModelArgs, Gemma2RMSNorm, Attention, MLP, TransformerBlock, Gemma2Model, Model |
| models/gemma2.py | 184 | Model | Implemented | models/gemma2.rb |  |
| models/gemma3.py | 15 | ModelArgs | Implemented | models/gemma3.rb |  |
| models/gemma3.py | 30 | Model | Implemented | models/gemma3.rb |  |
| models/gemma3_text.py | 16 | ModelArgs | Implemented | models/gemma3_text.rb |  |
| models/gemma3_text.py | 35 | Attention | Implemented | models/gemma3_text.rb |  |
| models/gemma3_text.py | 104 | RMSNorm | Implemented | models/gemma3_text.rb |  |
| models/gemma3_text.py | 114 | MLP | Implemented | models/gemma3_text.rb |  |
| models/gemma3_text.py | 135 | TransformerBlock | Implemented | models/gemma3_text.rb |  |
| models/gemma3_text.py | 164 | Gemma3Model | Implemented | models/gemma3_text.rb |  |
| models/gemma3_text.py | 215 | Model | Implemented | models/gemma3_text.rb |  |
| models/gemma3n.py | 17 | TextConfig | Partial | models/gemma3n.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/gemma3n.py | 46 | ModelArgs | Implemented | models/gemma3n.rb |  |
| models/gemma3n.py | 51 | RMSNoScale | Partial | models/gemma3n.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/gemma3n.py | 60 | Gemma3nLaurelBlock | Partial | models/gemma3n.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/gemma3n.py | 85 | Gemma3nAttention | Partial | models/gemma3n.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/gemma3n.py | 171 | MLP | Partial | models/gemma3n.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/gemma3n.py | 204 | Gemma3nAltUp | Partial | models/gemma3n.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/gemma3n.py | 283 | Gemma3nDecoderLayer | Partial | models/gemma3n.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/gemma3n.py | 379 | LanguageModel | Partial | models/gemma3n.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/gemma3n.py | 568 | Gemma3n | Partial | models/gemma3n.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/gemma3n.py | 587 | Model | Implemented | models/gemma3n.rb |  |
| models/glm.py | 15 | ModelArgs | Implemented | models/glm.rb |  |
| models/glm.py | 31 | GLMAttention | Partial | models/glm.rb | Ruby file exists; classes differ: ModelArgs, Attention, MLP, TransformerBlock, GLMModel, Model |
| models/glm.py | 95 | GLMMLP | Partial | models/glm.rb | Ruby file exists; classes differ: ModelArgs, Attention, MLP, TransformerBlock, GLMModel, Model |
| models/glm.py | 109 | GLMBlock | Partial | models/glm.rb | Ruby file exists; classes differ: ModelArgs, Attention, MLP, TransformerBlock, GLMModel, Model |
| models/glm.py | 132 | GLMModel | Implemented | models/glm.rb |  |
| models/glm.py | 157 | Model | Implemented | models/glm.rb |  |
| models/glm4.py | 14 | ModelArgs | Implemented | models/glm4.rb |  |
| models/glm4.py | 31 | Glm4MLP | Implemented | models/glm4.rb | Name variant in Ruby: GLM4MLP |
| models/glm4.py | 45 | Glm4Attention | Implemented | models/glm4.rb | Name variant in Ruby: GLM4Attention |
| models/glm4.py | 107 | Glm4DecoderLayer | Implemented | models/glm4.rb | Name variant in Ruby: GLM4DecoderLayer |
| models/glm4.py | 136 | Glm4Model | Implemented | models/glm4.rb | Name variant in Ruby: GLM4Model |
| models/glm4.py | 163 | Model | Implemented | models/glm4.rb |  |
| models/glm4_moe.py | 19 | ModelArgs | Implemented | models/glm4_moe.rb |  |
| models/glm4_moe.py | 49 | Attention | Implemented | models/glm4_moe.rb |  |
| models/glm4_moe.py | 111 | MLP | Implemented | models/glm4_moe.rb |  |
| models/glm4_moe.py | 166 | MoEGate | Implemented | models/glm4_moe.rb |  |
| models/glm4_moe.py | 192 | MoE | Implemented | models/glm4_moe.rb |  |
| models/glm4_moe.py | 228 | DecoderLayer | Implemented | models/glm4_moe.rb |  |
| models/glm4_moe.py | 257 | LanguageModel | Implemented | models/glm4_moe.rb |  |
| models/glm4_moe.py | 301 | Model | Implemented | models/glm4_moe.rb |  |
| models/glm4_moe_lite.py | 20 | ModelArgs | Implemented | models/glm4_moe_lite.rb |  |
| models/glm4_moe_lite.py | 57 | Glm4MoeLiteAttention | Partial | models/glm4_moe_lite.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/glm4_moe_lite.py | 177 | Glm4MoeLiteMLP | Partial | models/glm4_moe_lite.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/glm4_moe_lite.py | 231 | MoEGate | Partial | models/glm4_moe_lite.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/glm4_moe_lite.py | 257 | Glm4MoeLiteMoE | Partial | models/glm4_moe_lite.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/glm4_moe_lite.py | 293 | Glm4MoeLiteDecoderLayer | Partial | models/glm4_moe_lite.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/glm4_moe_lite.py | 320 | Glm4MoeLiteModel | Partial | models/glm4_moe_lite.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/glm4_moe_lite.py | 365 | Model | Implemented | models/glm4_moe_lite.rb |  |
| models/glm_moe_dsa.py | 11 | ModelArgs | Implemented | models/glm_moe_dsa.rb |  |
| models/glm_moe_dsa.py | 51 | Model | Implemented | models/glm_moe_dsa.rb |  |
| models/gpt2.py | 13 | ModelArgs | Implemented | models/gpt2.rb |  |
| models/gpt2.py | 29 | Attention | Implemented | models/gpt2.rb |  |
| models/gpt2.py | 71 | MLP | Implemented | models/gpt2.rb |  |
| models/gpt2.py | 83 | TransformerBlock | Implemented | models/gpt2.rb |  |
| models/gpt2.py | 111 | GPT2Model | Implemented | models/gpt2.rb |  |
| models/gpt2.py | 154 | Model | Implemented | models/gpt2.rb |  |
| models/gpt_bigcode.py | 14 | ModelArgs | Implemented | models/gpt_bigcode.rb |  |
| models/gpt_bigcode.py | 34 | Attention | Implemented | models/gpt_bigcode.rb |  |
| models/gpt_bigcode.py | 84 | MLP | Implemented | models/gpt_bigcode.rb |  |
| models/gpt_bigcode.py | 102 | TransformerBlock | Implemented | models/gpt_bigcode.rb |  |
| models/gpt_bigcode.py | 126 | GPTBigCodeModel | Implemented | models/gpt_bigcode.rb |  |
| models/gpt_bigcode.py | 162 | Model | Implemented | models/gpt_bigcode.rb |  |
| models/gpt_neox.py | 16 | ModelArgs | Implemented | models/gpt_neox.rb |  |
| models/gpt_neox.py | 34 | Attention | Implemented | models/gpt_neox.rb |  |
| models/gpt_neox.py | 90 | MLP | Implemented | models/gpt_neox.rb |  |
| models/gpt_neox.py | 103 | TransformerBlock | Implemented | models/gpt_neox.rb |  |
| models/gpt_neox.py | 142 | GPTNeoXModel | Implemented | models/gpt_neox.rb |  |
| models/gpt_neox.py | 178 | Model | Implemented | models/gpt_neox.rb |  |
| models/gpt_oss.py | 19 | ModelArgs | Implemented | models/gpt_oss.rb |  |
| models/gpt_oss.py | 62 | SwiGLU | Partial | models/gpt_oss.rb | Ruby file exists; classes differ: ModelArgs, AttentionBlock, MLPBlock, TransformerBlock, GptOssMoeModel, Model |
| models/gpt_oss.py | 70 | AttentionBlock | Implemented | models/gpt_oss.rb |  |
| models/gpt_oss.py | 130 | MLPBlock | Implemented | models/gpt_oss.rb |  |
| models/gpt_oss.py | 169 | TransformerBlock | Implemented | models/gpt_oss.rb |  |
| models/gpt_oss.py | 192 | GptOssMoeModel | Implemented | models/gpt_oss.rb |  |
| models/gpt_oss.py | 232 | Model | Implemented | models/gpt_oss.rb |  |
| models/granite.py | 15 | ModelArgs | Implemented | models/granite.rb |  |
| models/granite.py | 36 | Attention | Implemented | models/granite.rb |  |
| models/granite.py | 92 | MLP | Implemented | models/granite.rb |  |
| models/granite.py | 111 | TransformerBlock | Implemented | models/granite.rb |  |
| models/granite.py | 137 | GraniteModel | Implemented | models/granite.rb |  |
| models/granite.py | 169 | Model | Implemented | models/granite.rb |  |
| models/granitemoe.py | 15 | ModelArgs | Implemented | models/granitemoe.rb |  |
| models/granitemoe.py | 37 | GraniteMoeAttention | Partial | models/granitemoe.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/granitemoe.py | 92 | GraniteMoeTopKGating | Partial | models/granitemoe.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/granitemoe.py | 110 | GraniteMoeMoE | Partial | models/granitemoe.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/granitemoe.py | 131 | GraniteMoeDecoderLayer | Partial | models/granitemoe.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/granitemoe.py | 155 | GraniteMoEModel | Partial | models/granitemoe.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/granitemoe.py | 184 | Model | Implemented | models/granitemoe.rb |  |
| models/granitemoehybrid.py | 23 | ModelArgs | Implemented | models/granitemoehybrid.rb |  |
| models/granitemoehybrid.py | 71 | GraniteMoeHybridRMSNormGated | Partial | models/granitemoehybrid.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/granitemoehybrid.py | 83 | GraniteMoeHybridMamba2Mixer | Partial | models/granitemoehybrid.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/granitemoehybrid.py | 226 | GraniteMoeHybridAttention | Partial | models/granitemoehybrid.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/granitemoehybrid.py | 290 | GraniteMoeHybridTopKGating | Partial | models/granitemoehybrid.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/granitemoehybrid.py | 308 | GraniteMoeHybridMoE | Partial | models/granitemoehybrid.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/granitemoehybrid.py | 329 | GraniteMoeHybridSharedMLP | Partial | models/granitemoehybrid.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/granitemoehybrid.py | 344 | GraniteMoeHybridMLP | Partial | models/granitemoehybrid.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/granitemoehybrid.py | 359 | GraniteMoeHybridLayer | Partial | models/granitemoehybrid.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/granitemoehybrid.py | 420 | GraniteMoeHybridModel | Partial | models/granitemoehybrid.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/granitemoehybrid.py | 467 | Model | Implemented | models/granitemoehybrid.rb |  |
| models/helium.py | 14 | ModelArgs | Implemented | models/helium.rb |  |
| models/helium.py | 31 | HeliumAttention | Partial | models/helium.rb | Ruby file exists; classes differ: ModelArgs, Attention, MLP, DecoderLayer, HeliumModel, Model |
| models/helium.py | 79 | HeliumMLP | Partial | models/helium.rb | Ruby file exists; classes differ: ModelArgs, Attention, MLP, DecoderLayer, HeliumModel, Model |
| models/helium.py | 99 | HeliumDecoderLayer | Partial | models/helium.rb | Ruby file exists; classes differ: ModelArgs, Attention, MLP, DecoderLayer, HeliumModel, Model |
| models/helium.py | 124 | HeliumModel | Implemented | models/helium.rb |  |
| models/helium.py | 155 | Model | Implemented | models/helium.rb |  |
| models/hunyuan.py | 15 | ModelArgs | Implemented | models/hunyuan.rb |  |
| models/hunyuan.py | 51 | DynamicNTKAlphaRoPE | Implemented | models/hunyuan.rb |  |
| models/hunyuan.py | 75 | Attention | Implemented | models/hunyuan.rb |  |
| models/hunyuan.py | 144 | MLP | Implemented | models/hunyuan.rb |  |
| models/hunyuan.py | 155 | Gate | Implemented | models/hunyuan.rb |  |
| models/hunyuan.py | 164 | MoeBlock | Implemented | models/hunyuan.rb |  |
| models/hunyuan.py | 210 | DecoderLayer | Implemented | models/hunyuan.rb |  |
| models/hunyuan.py | 242 | HunYuanModel | Implemented | models/hunyuan.rb |  |
| models/hunyuan.py | 279 | Model | Implemented | models/hunyuan.rb |  |
| models/hunyuan_v1_dense.py | 14 | ModelArgs | Implemented | models/hunyuan_v1_dense.rb |  |
| models/hunyuan_v1_dense.py | 38 | DynamicNTKAlphaRoPE | Implemented | models/hunyuan_v1_dense.rb |  |
| models/hunyuan_v1_dense.py | 62 | Attention | Implemented | models/hunyuan_v1_dense.rb |  |
| models/hunyuan_v1_dense.py | 136 | MLP | Implemented | models/hunyuan_v1_dense.rb |  |
| models/hunyuan_v1_dense.py | 151 | TransformerBlock | Implemented | models/hunyuan_v1_dense.rb |  |
| models/hunyuan_v1_dense.py | 177 | HunyuanV1DenseModel | Implemented | models/hunyuan_v1_dense.rb |  |
| models/hunyuan_v1_dense.py | 204 | Model | Implemented | models/hunyuan_v1_dense.rb |  |
| models/internlm2.py | 14 | ModelArgs | Implemented | models/internlm2.rb |  |
| models/internlm2.py | 45 | DynamicNTKScalingRoPE | Partial | models/internlm2.rb | Ruby file exists; classes differ: ModelArgs, Attention, MLP, TransformerBlock, InternLM2Model, Model |
| models/internlm2.py | 85 | Attention | Implemented | models/internlm2.rb |  |
| models/internlm2.py | 152 | MLP | Implemented | models/internlm2.rb |  |
| models/internlm2.py | 163 | TransformerBlock | Implemented | models/internlm2.rb |  |
| models/internlm2.py | 184 | InternLM2Model | Implemented | models/internlm2.rb |  |
| models/internlm2.py | 211 | Model | Implemented | models/internlm2.rb |  |
| models/internlm3.py | 14 | ModelArgs | Implemented | models/internlm3.rb |  |
| models/internlm3.py | 46 | DynamicNTKScalingRoPE | Implemented | models/internlm3.rb |  |
| models/internlm3.py | 86 | Attention | Implemented | models/internlm3.rb |  |
| models/internlm3.py | 150 | MLP | Implemented | models/internlm3.rb |  |
| models/internlm3.py | 161 | TransformerBlock | Implemented | models/internlm3.rb |  |
| models/internlm3.py | 184 | InternLM2Model | Partial | models/internlm3.rb | Ruby file exists; classes differ: ModelArgs, DynamicNTKScalingRoPE, Attention, MLP, TransformerBlock, InternLM3Model, Model |
| models/internlm3.py | 211 | Model | Implemented | models/internlm3.rb |  |
| models/iquestloopcoder.py | 36 | ModelArgs | Implemented | models/iquestloopcoder.rb |  |
| models/iquestloopcoder.py | 56 | LoopGateProjection | Implemented | models/iquestloopcoder.rb |  |
| models/iquestloopcoder.py | 68 | Attention | Implemented | models/iquestloopcoder.rb |  |
| models/iquestloopcoder.py | 117 | MLP | Implemented | models/iquestloopcoder.rb |  |
| models/iquestloopcoder.py | 130 | TransformerBlock | Implemented | models/iquestloopcoder.rb |  |
| models/iquestloopcoder.py | 141 | IQuestLoopCoderModel | Implemented | models/iquestloopcoder.rb |  |
| models/iquestloopcoder.py | 219 | Model | Implemented | models/iquestloopcoder.rb |  |
| models/jamba.py | 22 | ModelArgs | Implemented | models/jamba.rb |  |
| models/jamba.py | 61 | JambaMLP | Partial | models/jamba.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/jamba.py | 72 | JambaAttention | Partial | models/jamba.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/jamba.py | 127 | JambaMambaMixer | Partial | models/jamba.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/jamba.py | 229 | JambaSparseMoeBlock | Partial | models/jamba.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/jamba.py | 250 | JambaDecoderLayer | Partial | models/jamba.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/jamba.py | 284 | JambaModel | Partial | models/jamba.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/jamba.py | 317 | Model | Implemented | models/jamba.rb |  |
| models/kimi_k25.py | 17 | ModelArgs | Implemented | models/kimi_k25.rb |  |
| models/kimi_k25.py | 26 | LanguageModel | Partial | models/kimi_k25.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/kimi_k25.py | 42 | Model | Implemented | models/kimi_k25.rb |  |
| models/kimi_linear.py | 23 | ModelArgs | Implemented | models/kimi_linear.rb |  |
| models/kimi_linear.py | 57 | KimiMLP | Partial | models/kimi_linear.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/kimi_linear.py | 120 | KimiSparseMoE | Partial | models/kimi_linear.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/kimi_linear.py | 158 | KimiMLAAttention | Partial | models/kimi_linear.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/kimi_linear.py | 235 | ShortConv1d | Partial | models/kimi_linear.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/kimi_linear.py | 275 | KimiDeltaAttention | Partial | models/kimi_linear.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/kimi_linear.py | 388 | KimiDecoderLayer | Partial | models/kimi_linear.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/kimi_linear.py | 426 | KimiLinearModel | Partial | models/kimi_linear.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/kimi_linear.py | 458 | Model | Implemented | models/kimi_linear.rb |  |
| models/kimi_vl.py | 14 | TextArgs | Partial | models/kimi_vl.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/kimi_vl.py | 46 | ModelArgs | Implemented | models/kimi_vl.rb |  |
| models/kimi_vl.py | 54 | LanguageModel | Partial | models/kimi_vl.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/kimi_vl.py | 70 | Model | Implemented | models/kimi_vl.rb |  |
| models/lfm2-vl.py | 15 | ModelArgs | Implemented | models/lfm2_vl.rb |  |
| models/lfm2-vl.py | 23 | Model | Implemented | models/lfm2_vl.rb |  |
| models/lfm2.py | 19 | ModelArgs | Implemented | models/lfm2.rb |  |
| models/lfm2.py | 53 | Attention | Partial | models/lfm2.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/lfm2.py | 112 | ShortConv | Partial | models/lfm2.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/lfm2.py | 173 | MLP | Partial | models/lfm2.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/lfm2.py | 197 | Lfm2DecoderLayer | Partial | models/lfm2.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/lfm2.py | 237 | Lfm2Model | Partial | models/lfm2.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/lfm2.py | 282 | Model | Implemented | models/lfm2.rb |  |
| models/lfm2_moe.py | 20 | ModelArgs | Implemented | models/lfm2_moe.rb |  |
| models/lfm2_moe.py | 54 | Attention | Implemented | models/lfm2_moe.rb |  |
| models/lfm2_moe.py | 113 | ShortConv | Implemented | models/lfm2_moe.rb |  |
| models/lfm2_moe.py | 174 | MLP | Implemented | models/lfm2_moe.rb |  |
| models/lfm2_moe.py | 189 | Lfm2MoeSparseMoeBlock | Partial | models/lfm2_moe.rb | Ruby file exists; classes differ: ModelArgs, Attention, ShortConv, MLP, SparseMoeBlock, DecoderLayer, Lfm2MoeModel, Model |
| models/lfm2_moe.py | 229 | Lfm2DecoderLayer | Partial | models/lfm2_moe.rb | Ruby file exists; classes differ: ModelArgs, Attention, ShortConv, MLP, SparseMoeBlock, DecoderLayer, Lfm2MoeModel, Model |
| models/lfm2_moe.py | 270 | Lfm2Model | Partial | models/lfm2_moe.rb | Ruby file exists; classes differ: ModelArgs, Attention, ShortConv, MLP, SparseMoeBlock, DecoderLayer, Lfm2MoeModel, Model |
| models/lfm2_moe.py | 315 | Model | Implemented | models/lfm2_moe.rb |  |
| models/lille-130m.py | 14 | ModelArgs | Implemented | models/lille_130m.rb |  |
| models/lille-130m.py | 27 | Lille130mAttention | Implemented | models/lille_130m.rb |  |
| models/lille-130m.py | 79 | Lille130mMLP | Implemented | models/lille_130m.rb |  |
| models/lille-130m.py | 94 | Lille130Block | Implemented | models/lille_130m.rb |  |
| models/lille-130m.py | 111 | Lille130 | Implemented | models/lille_130m.rb |  |
| models/lille-130m.py | 136 | Model | Implemented | models/lille_130m.rb |  |
| models/llama.py | 17 | ModelArgs | Implemented | models/llama.rb |  |
| models/llama.py | 45 | Attention | Implemented | models/llama.rb |  |
| models/llama.py | 105 | MLP | Implemented | models/llama.rb |  |
| models/llama.py | 124 | TransformerBlock | Implemented | models/llama.rb |  |
| models/llama.py | 151 | LlamaModel | Implemented | models/llama.rb |  |
| models/llama.py | 200 | Model | Implemented | models/llama.rb |  |
| models/llama4.py | 17 | TextArgs | Implemented | models/llama4.rb |  |
| models/llama4.py | 43 | ModelArgs | Implemented | models/llama4.rb |  |
| models/llama4.py | 51 | Attention | Implemented | models/llama4.rb |  |
| models/llama4.py | 137 | MLP | Implemented | models/llama4.rb |  |
| models/llama4.py | 152 | MoE | Implemented | models/llama4.rb |  |
| models/llama4.py | 175 | TransformerBlock | Implemented | models/llama4.rb |  |
| models/llama4.py | 208 | LlamaModel | Implemented | models/llama4.rb |  |
| models/llama4.py | 258 | LanguageModel | Implemented | models/llama4.rb |  |
| models/llama4.py | 277 | Model | Implemented | models/llama4.rb |  |
| models/llama4_text.py | 14 | ModelArgs | Implemented | models/llama4_text.rb |  |
| models/llama4_text.py | 31 | Attention | Implemented | models/llama4_text.rb |  |
| models/llama4_text.py | 91 | MLP | Implemented | models/llama4_text.rb |  |
| models/llama4_text.py | 102 | TransformerBlock | Implemented | models/llama4_text.rb |  |
| models/llama4_text.py | 129 | LanguageModel | Implemented | models/llama4_text.rb |  |
| models/llama4_text.py | 158 | Model | Implemented | models/llama4_text.rb |  |
| models/longcat_flash.py | 18 | ModelArgs | Implemented | models/longcat_flash.rb |  |
| models/longcat_flash.py | 48 | LongcatFlashMLA | Partial | models/longcat_flash.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/longcat_flash.py | 182 | LongcatFlashMLP | Partial | models/longcat_flash.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/longcat_flash.py | 195 | LongcatFlashTopkRouter | Partial | models/longcat_flash.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/longcat_flash.py | 231 | LongcatFlashMoE | Partial | models/longcat_flash.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/longcat_flash.py | 280 | LongcatFlashDecoderLayer | Partial | models/longcat_flash.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/longcat_flash.py | 329 | LongcatFlashModel | Partial | models/longcat_flash.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/longcat_flash.py | 355 | Model | Implemented | models/longcat_flash.rb |  |
| models/longcat_flash_ngram.py | 16 | ModelArgs | Implemented | models/longcat_flash_ngram.rb |  |
| models/longcat_flash_ngram.py | 48 | NgramEmbedding | Partial | models/longcat_flash_ngram.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/longcat_flash_ngram.py | 146 | LongcatFlashNgramModel | Partial | models/longcat_flash_ngram.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/longcat_flash_ngram.py | 172 | Model | Implemented | models/longcat_flash_ngram.rb |  |
| models/mamba.py | 15 | ModelArgs | Implemented | models/mamba.rb |  |
| models/mamba.py | 54 | MambaBlock | Implemented | models/mamba.rb |  |
| models/mamba.py | 163 | ResidualBlock | Implemented | models/mamba.rb |  |
| models/mamba.py | 173 | Mamba | Partial | models/mamba.rb | Ruby file exists; classes differ: ModelArgs, MambaBlock, ResidualBlock, MambaModel, Model |
| models/mamba.py | 189 | Model | Implemented | models/mamba.rb |  |
| models/mamba2.py | 17 | ModelArgs | Implemented | models/mamba2.rb |  |
| models/mamba2.py | 44 | MambaRMSNormGated | Implemented | models/mamba2.rb |  |
| models/mamba2.py | 56 | Mamba2Block | Implemented | models/mamba2.rb |  |
| models/mamba2.py | 196 | ResidualBlock | Implemented | models/mamba2.rb |  |
| models/mamba2.py | 209 | Mamba2 | Partial | models/mamba2.rb | Ruby file exists; classes differ: ModelArgs, MambaRMSNormGated, Mamba2Block, ResidualBlock, Mamba2Model, Model |
| models/mamba2.py | 232 | Model | Implemented | models/mamba2.rb |  |
| models/mimo.py | 15 | ModelArgs | Implemented | models/mimo.rb |  |
| models/mimo.py | 32 | Attention | Implemented | models/mimo.rb |  |
| models/mimo.py | 86 | MLP | Implemented | models/mimo.rb |  |
| models/mimo.py | 97 | TransformerBlock | Implemented | models/mimo.rb |  |
| models/mimo.py | 123 | MiMoModel | Implemented | models/mimo.rb | Name variant in Ruby: MimoModel |
| models/mimo.py | 158 | Model | Implemented | models/mimo.rb |  |
| models/mimo_v2_flash.py | 18 | ModelArgs | Implemented | models/mimo_v2_flash.rb |  |
| models/mimo_v2_flash.py | 54 | Attention | Implemented | models/mimo_v2_flash.rb |  |
| models/mimo_v2_flash.py | 127 | MLP | Implemented | models/mimo_v2_flash.rb |  |
| models/mimo_v2_flash.py | 182 | MoEGate | Implemented | models/mimo_v2_flash.rb |  |
| models/mimo_v2_flash.py | 212 | MoE | Implemented | models/mimo_v2_flash.rb |  |
| models/mimo_v2_flash.py | 240 | DecoderLayer | Implemented | models/mimo_v2_flash.rb |  |
| models/mimo_v2_flash.py | 265 | LanguageModel | Implemented | models/mimo_v2_flash.rb |  |
| models/mimo_v2_flash.py | 305 | Model | Implemented | models/mimo_v2_flash.rb |  |
| models/minicpm.py | 15 | ModelArgs | Implemented | models/minicpm.rb |  |
| models/minicpm.py | 34 | MLP | Implemented | models/minicpm.rb |  |
| models/minicpm.py | 45 | Attention | Implemented | models/minicpm.rb |  |
| models/minicpm.py | 114 | DecoderLayer | Implemented | models/minicpm.rb |  |
| models/minicpm.py | 144 | MiniCPMModel | Implemented | models/minicpm.rb |  |
| models/minicpm.py | 173 | Model | Implemented | models/minicpm.rb |  |
| models/minicpm3.py | 15 | ModelArgs | Implemented | models/minicpm3.rb |  |
| models/minicpm3.py | 39 | Attention | Implemented | models/minicpm3.rb |  |
| models/minicpm3.py | 152 | MLP | Implemented | models/minicpm3.rb |  |
| models/minicpm3.py | 163 | DecoderLayer | Implemented | models/minicpm3.rb |  |
| models/minicpm3.py | 193 | MiniCPM3Model | Implemented | models/minicpm3.rb |  |
| models/minicpm3.py | 224 | Model | Implemented | models/minicpm3.rb |  |
| models/minimax.py | 16 | ModelArgs | Implemented | models/minimax.rb |  |
| models/minimax.py | 58 | ShardedRMSNorm | Partial | models/minimax.rb | Ruby file exists; classes differ: ModelArgs, Attention, SparseMoeBlock, DecoderLayer, MiniMaxModel, Model |
| models/minimax.py | 86 | MiniMaxAttention | Partial | models/minimax.rb | Ruby file exists; classes differ: ModelArgs, Attention, SparseMoeBlock, DecoderLayer, MiniMaxModel, Model |
| models/minimax.py | 162 | MiniMaxSparseMoeBlock | Partial | models/minimax.rb | Ruby file exists; classes differ: ModelArgs, Attention, SparseMoeBlock, DecoderLayer, MiniMaxModel, Model |
| models/minimax.py | 200 | MiniMaxDecoderLayer | Partial | models/minimax.rb | Ruby file exists; classes differ: ModelArgs, Attention, SparseMoeBlock, DecoderLayer, MiniMaxModel, Model |
| models/minimax.py | 224 | MiniMaxModel | Implemented | models/minimax.rb |  |
| models/minimax.py | 254 | Model | Implemented | models/minimax.rb |  |
| models/ministral3.py | 18 | ModelArgs | Implemented | models/ministral3.rb |  |
| models/ministral3.py | 55 | Attention | Implemented | models/ministral3.rb |  |
| models/ministral3.py | 114 | MLP | Implemented | models/ministral3.rb |  |
| models/ministral3.py | 128 | TransformerBlock | Implemented | models/ministral3.rb |  |
| models/ministral3.py | 156 | LanguageModel | Implemented | models/ministral3.rb |  |
| models/ministral3.py | 245 | Model | Implemented | models/ministral3.rb |  |
| models/mistral3.py | 15 | ModelArgs | Implemented | models/mistral3.rb |  |
| models/mistral3.py | 24 | Model | Implemented | models/mistral3.rb |  |
| models/mixtral.py | 14 | ModelArgs | Implemented | models/mixtral.rb |  |
| models/mixtral.py | 35 | MixtralAttention | Partial | models/mixtral.rb | Ruby file exists; classes differ: ModelArgs, Attention, SparseMoeBlock, MixtralDecoderLayer, MixtralModel, Model |
| models/mixtral.py | 97 | MixtralSparseMoeBlock | Partial | models/mixtral.rb | Ruby file exists; classes differ: ModelArgs, Attention, SparseMoeBlock, MixtralDecoderLayer, MixtralModel, Model |
| models/mixtral.py | 124 | MixtralDecoderLayer | Implemented | models/mixtral.rb |  |
| models/mixtral.py | 150 | MixtralModel | Implemented | models/mixtral.rb |  |
| models/mixtral.py | 184 | Model | Implemented | models/mixtral.rb |  |
| models/mla.py | 9 | MultiLinear | Implemented | models/mla.rb |  |
| models/mla.py | 45 | QuantizedMultiLinear | Implemented | models/mla.rb |  |
| models/nanochat.py | 15 | ModelArgs | Implemented | models/nanochat.rb |  |
| models/nanochat.py | 66 | Attention | Implemented | models/nanochat.rb |  |
| models/nanochat.py | 144 | MLP | Implemented | models/nanochat.rb |  |
| models/nanochat.py | 157 | TransformerBlock | Implemented | models/nanochat.rb |  |
| models/nanochat.py | 175 | NanoChatModel | Implemented | models/nanochat.rb |  |
| models/nanochat.py | 210 | Model | Implemented | models/nanochat.rb |  |
| models/nemotron-nas.py | 15 | AttentionConfig | Implemented | models/nemotron_nas.rb |  |
| models/nemotron-nas.py | 47 | FFNConfig | Implemented | models/nemotron_nas.rb |  |
| models/nemotron-nas.py | 66 | BlockConfig | Implemented | models/nemotron_nas.rb |  |
| models/nemotron-nas.py | 102 | ModelArgs | Implemented | models/nemotron_nas.rb |  |
| models/nemotron-nas.py | 150 | Attention | Implemented | models/nemotron_nas.rb |  |
| models/nemotron-nas.py | 215 | MLP | Implemented | models/nemotron_nas.rb |  |
| models/nemotron-nas.py | 238 | LinearSubblockReplacement | Implemented | models/nemotron_nas.rb |  |
| models/nemotron-nas.py | 250 | TransformerBlock | Implemented | models/nemotron_nas.rb |  |
| models/nemotron-nas.py | 319 | NemotronNASModel | Implemented | models/nemotron_nas.rb |  |
| models/nemotron-nas.py | 361 | Model | Implemented | models/nemotron_nas.rb |  |
| models/nemotron.py | 14 | ModelArgs | Implemented | models/nemotron.rb |  |
| models/nemotron.py | 54 | NemotronLayerNorm1P | Implemented | models/nemotron.rb |  |
| models/nemotron.py | 61 | Attention | Implemented | models/nemotron.rb |  |
| models/nemotron.py | 123 | MLP | Implemented | models/nemotron.rb |  |
| models/nemotron.py | 138 | TransformerBlock | Implemented | models/nemotron.rb |  |
| models/nemotron.py | 163 | NemotronModel | Implemented | models/nemotron.rb |  |
| models/nemotron.py | 193 | Model | Implemented | models/nemotron.rb |  |
| models/nemotron_h.py | 23 | ModelArgs | Implemented | models/nemotron_h.rb |  |
| models/nemotron_h.py | 67 | MambaRMSNormGated | Partial | models/nemotron_h.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/nemotron_h.py | 82 | NemotronHMamba2Mixer | Partial | models/nemotron_h.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/nemotron_h.py | 228 | NemotronHAttention | Partial | models/nemotron_h.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/nemotron_h.py | 288 | NemotronHMLP | Partial | models/nemotron_h.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/nemotron_h.py | 338 | MoEGate | Partial | models/nemotron_h.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/nemotron_h.py | 363 | NemotronHMoE | Partial | models/nemotron_h.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/nemotron_h.py | 392 | NemotronHBlock | Partial | models/nemotron_h.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/nemotron_h.py | 423 | NemotronHModel | Partial | models/nemotron_h.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/nemotron_h.py | 474 | Model | Implemented | models/nemotron_h.rb |  |
| models/olmo.py | 21 | ModelArgs | Implemented | models/olmo.rb |  |
| models/olmo.py | 42 | TransformerBlock | Implemented | models/olmo.rb |  |
| models/olmo.py | 113 | Transformer | Implemented | models/olmo.rb |  |
| models/olmo.py | 148 | OlmoModel | Implemented | models/olmo.rb |  |
| models/olmo.py | 161 | Model | Implemented | models/olmo.rb |  |
| models/olmo2.py | 15 | ModelArgs | Implemented | models/olmo2.rb |  |
| models/olmo2.py | 38 | Attention | Implemented | models/olmo2.rb |  |
| models/olmo2.py | 103 | MLP | Implemented | models/olmo2.rb |  |
| models/olmo2.py | 122 | TransformerBlock | Implemented | models/olmo2.rb |  |
| models/olmo2.py | 150 | LlamaModel | Partial | models/olmo2.rb | Ruby file exists; classes differ: ModelArgs, Attention, MLP, TransformerBlock, OLMo2Model, Model |
| models/olmo2.py | 180 | Model | Implemented | models/olmo2.rb |  |
| models/olmo3.py | 16 | ModelArgs | Implemented | models/olmo3.rb |  |
| models/olmo3.py | 44 | Olmo3Attention | Implemented | models/olmo3.rb |  |
| models/olmo3.py | 127 | Olmo3MLP | Implemented | models/olmo3.rb |  |
| models/olmo3.py | 138 | Olmo3DecoderLayer | Implemented | models/olmo3.rb |  |
| models/olmo3.py | 166 | Olmo3Model | Implemented | models/olmo3.rb |  |
| models/olmo3.py | 204 | Model | Implemented | models/olmo3.rb |  |
| models/olmoe.py | 15 | ModelArgs | Implemented | models/olmoe.rb |  |
| models/olmoe.py | 41 | Attention | Partial | models/olmoe.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/olmoe.py | 96 | OlmoeSparseMoeBlock | Partial | models/olmoe.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/olmoe.py | 128 | TransformerBlock | Partial | models/olmoe.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/olmoe.py | 149 | OlmoeModel | Partial | models/olmoe.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/olmoe.py | 176 | Model | Implemented | models/olmoe.rb |  |
| models/openelm.py | 14 | ModelArgs | Implemented | models/openelm.rb |  |
| models/openelm.py | 57 | Attention | Implemented | models/openelm.rb |  |
| models/openelm.py | 120 | MLP | Implemented | models/openelm.rb |  |
| models/openelm.py | 143 | TransformerBlock | Implemented | models/openelm.rb |  |
| models/openelm.py | 165 | OpenELMModel | Implemented | models/openelm.rb |  |
| models/openelm.py | 196 | Model | Implemented | models/openelm.rb |  |
| models/phi.py | 13 | ModelArgs | Implemented | models/phi.rb |  |
| models/phi.py | 31 | PhiAttention | Implemented | models/phi.rb |  |
| models/phi.py | 109 | PhiMLP | Implemented | models/phi.rb |  |
| models/phi.py | 119 | PhiDecoderLayer | Implemented | models/phi.rb |  |
| models/phi.py | 135 | PhiModel | Implemented | models/phi.rb |  |
| models/phi.py | 156 | Model | Implemented | models/phi.rb |  |
| models/phi3.py | 15 | ModelArgs | Implemented | models/phi3.rb |  |
| models/phi3.py | 48 | Attention | Implemented | models/phi3.rb |  |
| models/phi3.py | 121 | MLP | Implemented | models/phi3.rb |  |
| models/phi3.py | 133 | TransformerBlock | Implemented | models/phi3.rb |  |
| models/phi3.py | 159 | Phi3Model | Implemented | models/phi3.rb |  |
| models/phi3.py | 190 | Model | Implemented | models/phi3.rb |  |
| models/phi3small.py | 15 | ModelArgs | Implemented | models/phi3small.rb |  |
| models/phi3small.py | 58 | Attention | Implemented | models/phi3small.rb |  |
| models/phi3small.py | 198 | MLP | Implemented | models/phi3small.rb |  |
| models/phi3small.py | 212 | TransformerBlock | Implemented | models/phi3small.rb |  |
| models/phi3small.py | 241 | Phi3Model | Implemented | models/phi3small.rb |  |
| models/phi3small.py | 278 | Model | Implemented | models/phi3small.rb |  |
| models/phimoe.py | 15 | ModelArgs | Implemented | models/phimoe.rb |  |
| models/phimoe.py | 32 | Attention | Implemented | models/phimoe.rb |  |
| models/phimoe.py | 89 | PhiMoESparseMoeBlock | Implemented | models/phimoe.rb |  |
| models/phimoe.py | 114 | PhiMoEDecoderLayer | Partial | models/phimoe.rb | Ruby file exists; classes differ: ModelArgs, Attention, PhiMoESparseMoeBlock, DecoderLayer, PhiMoEModel, Model |
| models/phimoe.py | 145 | PhiMoEModel | Implemented | models/phimoe.rb |  |
| models/phimoe.py | 172 | Model | Implemented | models/phimoe.rb |  |
| models/phixtral.py | 16 | ModelArgs | Implemented | models/phixtral.rb |  |
| models/phixtral.py | 37 | RoPEAttention | Implemented | models/phixtral.rb |  |
| models/phixtral.py | 87 | MOE | Implemented | models/phixtral.rb |  |
| models/phixtral.py | 113 | ParallelBlock | Implemented | models/phixtral.rb |  |
| models/phixtral.py | 129 | TransformerDecoder | Implemented | models/phixtral.rb |  |
| models/phixtral.py | 145 | Embd | Implemented | models/phixtral.rb |  |
| models/phixtral.py | 154 | OutputHead | Implemented | models/phixtral.rb |  |
| models/phixtral.py | 164 | Model | Implemented | models/phixtral.rb |  |
| models/pipeline.py | 6 | PipelineMixin | Partial | models/pipeline.rb | Ruby file exists but defines no classes |
| models/pixtral.py | 15 | ModelArgs | Implemented | models/pixtral.rb |  |
| models/pixtral.py | 26 | Model | Implemented | models/pixtral.rb |  |
| models/plamo.py | 15 | ModelArgs | Implemented | models/plamo.rb |  |
| models/plamo.py | 28 | Attention | Implemented | models/plamo.rb |  |
| models/plamo.py | 108 | MLP | Implemented | models/plamo.rb |  |
| models/plamo.py | 122 | PlamoDecoderLayer | Partial | models/plamo.rb | Ruby file exists; classes differ: ModelArgs, Attention, MLP, DecoderLayer, PlamoModel, Model |
| models/plamo.py | 156 | PlamoDecoder | Partial | models/plamo.rb | Ruby file exists; classes differ: ModelArgs, Attention, MLP, DecoderLayer, PlamoModel, Model |
| models/plamo.py | 164 | PlamoModel | Implemented | models/plamo.rb |  |
| models/plamo.py | 192 | Model | Implemented | models/plamo.rb |  |
| models/plamo2.py | 18 | ModelArgs | Implemented | models/plamo2.rb |  |
| models/plamo2.py | 40 | RMSNorm | Partial | models/plamo2.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/plamo2.py | 58 | Mamba | Partial | models/plamo2.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/plamo2.py | 223 | Attention | Partial | models/plamo2.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/plamo2.py | 295 | MLP | Partial | models/plamo2.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/plamo2.py | 312 | PlamoDecoderLayer | Partial | models/plamo2.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/plamo2.py | 378 | PlamoDecoder | Partial | models/plamo2.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/plamo2.py | 411 | PlamoModel | Partial | models/plamo2.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/plamo2.py | 439 | Model | Implemented | models/plamo2.rb |  |
| models/qwen.py | 13 | ModelArgs | Implemented | models/qwen.rb |  |
| models/qwen.py | 31 | Attention | Implemented | models/qwen.rb |  |
| models/qwen.py | 76 | MLP | Implemented | models/qwen.rb |  |
| models/qwen.py | 96 | TransformerBlock | Implemented | models/qwen.rb |  |
| models/qwen.py | 117 | QwenModel | Implemented | models/qwen.rb |  |
| models/qwen.py | 137 | Model | Implemented | models/qwen.rb |  |
| models/qwen2.py | 16 | ModelArgs | Implemented | models/qwen2.rb |  |
| models/qwen2.py | 32 | Attention | Implemented | models/qwen2.rb |  |
| models/qwen2.py | 87 | MLP | Implemented | models/qwen2.rb |  |
| models/qwen2.py | 98 | TransformerBlock | Implemented | models/qwen2.rb |  |
| models/qwen2.py | 124 | Qwen2Model | Implemented | models/qwen2.rb |  |
| models/qwen2.py | 158 | Model | Implemented | models/qwen2.rb |  |
| models/qwen2_moe.py | 15 | ModelArgs | Implemented | models/qwen2_moe.rb |  |
| models/qwen2_moe.py | 46 | Attention | Partial | models/qwen2_moe.rb | Ruby file exists; classes differ: ModelArgs, SharedExpertMLP, SparseMoeBlock, DecoderLayer, Qwen2MoeModel, Model |
| models/qwen2_moe.py | 99 | MLP | Partial | models/qwen2_moe.rb | Ruby file exists; classes differ: ModelArgs, SharedExpertMLP, SparseMoeBlock, DecoderLayer, Qwen2MoeModel, Model |
| models/qwen2_moe.py | 110 | Qwen2MoeSparseMoeBlock | Partial | models/qwen2_moe.rb | Ruby file exists; classes differ: ModelArgs, SharedExpertMLP, SparseMoeBlock, DecoderLayer, Qwen2MoeModel, Model |
| models/qwen2_moe.py | 148 | Qwen2MoeDecoderLayer | Partial | models/qwen2_moe.rb | Ruby file exists; classes differ: ModelArgs, SharedExpertMLP, SparseMoeBlock, DecoderLayer, Qwen2MoeModel, Model |
| models/qwen2_moe.py | 174 | Qwen2MoeModel | Implemented | models/qwen2_moe.rb |  |
| models/qwen2_moe.py | 205 | Model | Implemented | models/qwen2_moe.rb |  |
| models/qwen2_vl.py | 15 | ModelArgs | Implemented | models/qwen2_vl.rb |  |
| models/qwen2_vl.py | 26 | Model | Implemented | models/qwen2_vl.rb |  |
| models/qwen3.py | 16 | ModelArgs | Implemented | models/qwen3.rb |  |
| models/qwen3.py | 32 | Attention | Implemented | models/qwen3.rb |  |
| models/qwen3.py | 92 | MLP | Implemented | models/qwen3.rb |  |
| models/qwen3.py | 103 | TransformerBlock | Implemented | models/qwen3.rb |  |
| models/qwen3.py | 129 | Qwen3Model | Implemented | models/qwen3.rb |  |
| models/qwen3.py | 163 | Model | Implemented | models/qwen3.rb |  |
| models/qwen3_5.py | 24 | TextModelArgs | Partial | models/qwen3_5.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/qwen3_5.py | 85 | GatedDeltaNet | Partial | models/qwen3_5.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/qwen3_5.py | 191 | DecoderLayer | Partial | models/qwen3_5.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/qwen3_5.py | 225 | Qwen3_5TextModel | Partial | models/qwen3_5.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/qwen3_5.py | 260 | TextModel | Partial | models/qwen3_5.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/qwen3_5.py | 338 | ModelArgs | Implemented | models/qwen3_5.rb |  |
| models/qwen3_5.py | 349 | Model | Implemented | models/qwen3_5.rb |  |
| models/qwen3_5_moe.py | 12 | ModelArgs | Implemented | models/qwen3_5_moe.rb |  |
| models/qwen3_5_moe.py | 23 | Model | Implemented | models/qwen3_5_moe.rb |  |
| models/qwen3_moe.py | 15 | ModelArgs | Implemented | models/qwen3_moe.rb |  |
| models/qwen3_moe.py | 37 | Attention | Partial | models/qwen3_moe.rb | Ruby file exists; classes differ: ModelArgs, SparseMoeBlock, DecoderLayer, Qwen3MoeModel, Model |
| models/qwen3_moe.py | 99 | MLP | Partial | models/qwen3_moe.rb | Ruby file exists; classes differ: ModelArgs, SparseMoeBlock, DecoderLayer, Qwen3MoeModel, Model |
| models/qwen3_moe.py | 110 | Qwen3MoeSparseMoeBlock | Partial | models/qwen3_moe.rb | Ruby file exists; classes differ: ModelArgs, SparseMoeBlock, DecoderLayer, Qwen3MoeModel, Model |
| models/qwen3_moe.py | 142 | Qwen3MoeDecoderLayer | Partial | models/qwen3_moe.rb | Ruby file exists; classes differ: ModelArgs, SparseMoeBlock, DecoderLayer, Qwen3MoeModel, Model |
| models/qwen3_moe.py | 174 | Qwen3MoeModel | Implemented | models/qwen3_moe.rb |  |
| models/qwen3_moe.py | 210 | Model | Implemented | models/qwen3_moe.rb |  |
| models/qwen3_next.py | 25 | ModelArgs | Implemented | models/qwen3_next.rb |  |
| models/qwen3_next.py | 56 | Qwen3NextRMSNormGated | Partial | models/qwen3_next.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/qwen3_next.py | 71 | Qwen3NextAttention | Partial | models/qwen3_next.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/qwen3_next.py | 151 | Qwen3NextMLP | Partial | models/qwen3_next.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/qwen3_next.py | 162 | Qwen3NextGatedDeltaNet | Partial | models/qwen3_next.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/qwen3_next.py | 298 | Qwen3NextSparseMoeBlock | Partial | models/qwen3_next.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/qwen3_next.py | 337 | Qwen3NextDecoderLayer | Partial | models/qwen3_next.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/qwen3_next.py | 372 | Qwen3NextModel | Partial | models/qwen3_next.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/qwen3_next.py | 404 | Model | Implemented | models/qwen3_next.rb |  |
| models/qwen3_vl.py | 15 | ModelArgs | Implemented | models/qwen3_vl.rb |  |
| models/qwen3_vl.py | 26 | Model | Implemented | models/qwen3_vl.rb |  |
| models/qwen3_vl_moe.py | 15 | ModelArgs | Implemented | models/qwen3_vl_moe.rb |  |
| models/qwen3_vl_moe.py | 20 | Model | Implemented | models/qwen3_vl_moe.rb |  |
| models/recurrent_gemma.py | 15 | ModelArgs | Implemented | models/recurrent_gemma.rb |  |
| models/recurrent_gemma.py | 39 | RMSNorm | Implemented | models/recurrent_gemma.rb |  |
| models/recurrent_gemma.py | 79 | Conv1d | Partial | models/recurrent_gemma.rb | Ruby file exists; classes differ: ModelArgs, RMSNorm, RGLRU, RecurrentBlock, LocalAttentionBlock, MLPBlock, ResidualBlock, Griffin, Model |
| models/recurrent_gemma.py | 104 | RGLRU | Implemented | models/recurrent_gemma.rb |  |
| models/recurrent_gemma.py | 170 | RecurrentBlock | Implemented | models/recurrent_gemma.rb |  |
| models/recurrent_gemma.py | 220 | LocalAttentionBlock | Implemented | models/recurrent_gemma.rb |  |
| models/recurrent_gemma.py | 273 | MLPBlock | Implemented | models/recurrent_gemma.rb |  |
| models/recurrent_gemma.py | 287 | ResidualBlock | Implemented | models/recurrent_gemma.rb |  |
| models/recurrent_gemma.py | 363 | Griffin | Implemented | models/recurrent_gemma.rb |  |
| models/recurrent_gemma.py | 413 | Model | Implemented | models/recurrent_gemma.rb |  |
| models/rope_utils.py | 10 | SuScaledRoPE | Implemented | models/rope_utils.rb |  |
| models/rope_utils.py | 73 | Llama3RoPE | Implemented | models/rope_utils.rb |  |
| models/rope_utils.py | 128 | YarnRoPE | Implemented | models/rope_utils.rb |  |
| models/rwkv7.py | 15 | ModelArgs | Implemented | models/rwkv7.rb |  |
| models/rwkv7.py | 151 | LayerNormPerHead | Partial | models/rwkv7.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/rwkv7.py | 162 | LoRA | Partial | models/rwkv7.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/rwkv7.py | 198 | TokenShift | Partial | models/rwkv7.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/rwkv7.py | 209 | Rwkv7ChannelMixing | Partial | models/rwkv7.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/rwkv7.py | 232 | Rwkv7TimeMixing | Partial | models/rwkv7.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/rwkv7.py | 371 | Rwkv7Layer | Partial | models/rwkv7.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/rwkv7.py | 392 | Rwkv7Model | Partial | models/rwkv7.rb | Ruby file exists; classes differ: ModelArgs, Model |
| models/rwkv7.py | 412 | Model | Implemented | models/rwkv7.rb |  |
| models/seed_oss.py | 15 | ModelArgs | Implemented | models/seed_oss.rb |  |
| models/seed_oss.py | 35 | Attention | Implemented | models/seed_oss.rb |  |
| models/seed_oss.py | 92 | MLP | Implemented | models/seed_oss.rb |  |
| models/seed_oss.py | 103 | TransformerBlock | Implemented | models/seed_oss.rb |  |
| models/seed_oss.py | 128 | SeedModel | Implemented | models/seed_oss.rb |  |
| models/seed_oss.py | 155 | Model | Implemented | models/seed_oss.rb |  |
| models/smollm3.py | 13 | ModelArgs | Implemented | models/smollm3.rb |  |
| models/smollm3.py | 29 | NoPE | Implemented | models/smollm3.rb |  |
| models/smollm3.py | 36 | Model | Implemented | models/smollm3.rb |  |
| models/solar_open.py | 11 | ModelArgs | Implemented | models/solar_open.rb |  |
| models/stablelm.py | 14 | ModelArgs | Implemented | models/stablelm.rb |  |
| models/stablelm.py | 30 | LayerNormPerHead | Partial | models/stablelm.rb | Ruby file exists; classes differ: ModelArgs, Attention, MLP, DecoderLayer, StableLMModel, Model |
| models/stablelm.py | 44 | Attention | Implemented | models/stablelm.rb |  |
| models/stablelm.py | 131 | MLP | Implemented | models/stablelm.rb |  |
| models/stablelm.py | 142 | DecoderLayer | Implemented | models/stablelm.rb |  |
| models/stablelm.py | 171 | StableLM | Partial | models/stablelm.rb | Ruby file exists; classes differ: ModelArgs, Attention, MLP, DecoderLayer, StableLMModel, Model |
| models/stablelm.py | 191 | Model | Implemented | models/stablelm.rb |  |
| models/starcoder2.py | 13 | ModelArgs | Implemented | models/starcoder2.rb |  |
| models/starcoder2.py | 26 | Attention | Implemented | models/starcoder2.rb |  |
| models/starcoder2.py | 75 | MLP | Implemented | models/starcoder2.rb |  |
| models/starcoder2.py | 85 | TransformerBlock | Implemented | models/starcoder2.rb |  |
| models/starcoder2.py | 112 | Starcoder2Model | Implemented | models/starcoder2.rb |  |
| models/starcoder2.py | 143 | Model | Implemented | models/starcoder2.rb |  |
| models/step3p5.py | 25 | ClampedSwiGLU | Partial | models/step3p5.rb | Ruby file exists; classes differ: ModelArgs, ZeroCenteredRMSNorm, Step3p5MLP, Step3p5MoEGate, Step3p5MoE, Step3p5Attention, Step3p5DecoderLayer, Step3p5Model, Model |
| models/step3p5.py | 35 | ModelArgs | Implemented | models/step3p5.rb |  |
| models/step3p5.py | 66 | ZeroCenteredRMSNorm | Implemented | models/step3p5.rb |  |
| models/step3p5.py | 76 | Step3p5MLP | Implemented | models/step3p5.rb |  |
| models/step3p5.py | 116 | Step3p5MoEGate | Implemented | models/step3p5.rb |  |
| models/step3p5.py | 137 | Step3p5MoE | Implemented | models/step3p5.rb |  |
| models/step3p5.py | 186 | Step3p5Attention | Implemented | models/step3p5.rb |  |
| models/step3p5.py | 281 | Step3p5DecoderLayer | Implemented | models/step3p5.rb |  |
| models/step3p5.py | 327 | Step3p5Model | Implemented | models/step3p5.rb |  |
| models/step3p5.py | 376 | Model | Implemented | models/step3p5.rb |  |
| models/switch_layers.py | 27 | QuantizedSwitchLinear | Implemented | models/switch_layers.rb |  |
| models/switch_layers.py | 93 | SwitchLinear | Implemented | models/switch_layers.rb |  |
| models/switch_layers.py | 152 | SwiGLU | Partial | models/switch_layers.rb | Ruby file exists; classes differ: SwitchLinear, QuantizedSwitchLinear, SwitchGLU, SwitchMLP |
| models/switch_layers.py | 160 | SwitchGLU | Implemented | models/switch_layers.rb |  |
| models/switch_layers.py | 202 | SwitchMLP | Implemented | models/switch_layers.rb |  |
| models/telechat3.py | 15 | ModelArgs | Implemented | models/telechat3.rb |  |
| models/telechat3.py | 33 | Telechat3Attention | Implemented | models/telechat3.rb |  |
| models/telechat3.py | 103 | Telechat3MLP | Implemented | models/telechat3.rb |  |
| models/telechat3.py | 120 | Telechat3DecoderLayer | Implemented | models/telechat3.rb |  |
| models/telechat3.py | 145 | Telechat3Model | Implemented | models/telechat3.rb |  |
| models/telechat3.py | 178 | Model | Implemented | models/telechat3.rb |  |
| models/youtu_llm.py | 15 | ModelArgs | Implemented | models/youtu_llm.rb |  |
| models/youtu_llm.py | 38 | YoutuLLMAttention | Implemented | models/youtu_llm.rb |  |
| models/youtu_llm.py | 141 | YoutuLLMMLP | Implemented | models/youtu_llm.rb |  |
| models/youtu_llm.py | 158 | YoutuLLMDecoderLayer | Implemented | models/youtu_llm.rb |  |
| models/youtu_llm.py | 182 | YoutuLLMModel | Implemented | models/youtu_llm.rb |  |
| models/youtu_llm.py | 211 | Model | Implemented | models/youtu_llm.rb |  |
| quant/awq.py | 25 | ScaleConfig | Missing | - |  |
| quant/awq.py | 34 | AWQConfig | Missing | - |  |
| quant/awq.py | 430 | Catcher | Missing | - |  |
| quant/gptq.py | 40 | Catcher | Missing | - |  |
| server.py | 64 | StopCondition | Partial | server.rb | Ruby file exists; classes differ: ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChunk, ModelsListResponse |
| server.py | 185 | LRUPromptCache | Partial | server.rb | Ruby file exists; classes differ: ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChunk, ModelsListResponse |
| server.py | 188 | CacheEntry | Partial | server.rb | Ruby file exists; classes differ: ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChunk, ModelsListResponse |
| server.py | 194 | SearchResult | Partial | server.rb | Ruby file exists; classes differ: ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChunk, ModelsListResponse |
| server.py | 352 | ModelDescription | Partial | server.rb | Ruby file exists; classes differ: ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChunk, ModelsListResponse |
| server.py | 359 | SamplingArguments | Partial | server.rb | Ruby file exists; classes differ: ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChunk, ModelsListResponse |
| server.py | 369 | LogitsProcessorArguments | Partial | server.rb | Ruby file exists; classes differ: ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChunk, ModelsListResponse |
| server.py | 376 | GenerationArguments | Partial | server.rb | Ruby file exists; classes differ: ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChunk, ModelsListResponse |
| server.py | 392 | CompletionRequest | Partial | server.rb | Ruby file exists; classes differ: ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChunk, ModelsListResponse |
| server.py | 403 | GenerationContext | Partial | server.rb | Ruby file exists; classes differ: ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChunk, ModelsListResponse |
| server.py | 424 | Response | Partial | server.rb | Ruby file exists; classes differ: ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChunk, ModelsListResponse |
| server.py | 432 | TimeBudget | Partial | server.rb | Ruby file exists; classes differ: ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChunk, ModelsListResponse |
| server.py | 472 | ModelProvider | Partial | server.rb | Ruby file exists; classes differ: ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChunk, ModelsListResponse |
| server.py | 618 | ResponseGenerator | Partial | server.rb | Ruby file exists; classes differ: ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChunk, ModelsListResponse |
| server.py | 1059 | APIHandler | Partial | server.rb | Ruby file exists; classes differ: ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChunk, ModelsListResponse |
| share.py | 27 | DirectoryEntry | Missing | - |  |
| tokenizer_utils.py | 11 | StreamingDetokenizer | Implemented | tokenizer_utils.rb |  |
| tokenizer_utils.py | 61 | NaiveStreamingDetokenizer | Partial | tokenizer_utils.rb | Ruby file exists; classes differ: TokenizerWrapper, StreamingDetokenizer |
| tokenizer_utils.py | 107 | SPMStreamingDetokenizer | Partial | tokenizer_utils.rb | Ruby file exists; classes differ: TokenizerWrapper, StreamingDetokenizer |
| tokenizer_utils.py | 155 | BPEStreamingDetokenizer | Partial | tokenizer_utils.rb | Ruby file exists; classes differ: TokenizerWrapper, StreamingDetokenizer |
| tokenizer_utils.py | 256 | TokenizerWrapper | Implemented | tokenizer_utils.rb |  |
| tokenizer_utils.py | 401 | NewlineTokenizer | Partial | tokenizer_utils.rb | Ruby file exists; classes differ: TokenizerWrapper, StreamingDetokenizer |
| tuner/callbacks.py | 16 | TrainingCallback | Missing | - |  |
| tuner/callbacks.py | 27 | WandBCallback | Missing | - |  |
| tuner/callbacks.py | 65 | SwanLabCallback | Missing | - |  |
| tuner/datasets.py | 11 | TextDataset | Missing | - |  |
| tuner/datasets.py | 39 | ChatDataset | Missing | - |  |
| tuner/datasets.py | 86 | CompletionsDataset | Missing | - |  |
| tuner/datasets.py | 136 | ConcatenatedDataset | Missing | - |  |
| tuner/datasets.py | 158 | CacheDataset | Missing | - |  |
| tuner/dora.py | 9 | DoRALinear | Missing | - |  |
| tuner/dora.py | 131 | DoRAEmbedding | Missing | - |  |
| tuner/lora.py | 11 | LoRALinear | Implemented | tuner/lora.rb |  |
| tuner/lora.py | 101 | LoRASwitchLinear | Partial | tuner/lora.rb | Ruby file exists; classes differ: LoRALinear, LoRAEmbedding |
| tuner/lora.py | 198 | LoRAEmbedding | Implemented | tuner/lora.rb |  |
| tuner/trainer.py | 37 | TrainingArgs | Missing | - |  |
