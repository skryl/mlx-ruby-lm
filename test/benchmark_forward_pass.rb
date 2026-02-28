#!/usr/bin/env ruby
# frozen_string_literal: true

# Python vs Ruby Forward Pass Benchmark
# Benchmarks every registered model architecture with a small config,
# running a single forward pass in both Python and Ruby, and comparing timing.

$LOAD_PATH.unshift File.expand_path("../lib", __dir__)
$LOAD_PATH.unshift File.expand_path("../mlx-ruby/lib", __dir__)

require "mlx"
require "mlx_lm"
require "json"
require "open3"
require "timeout"

PYTHON = File.expand_path("../.venv-test/bin/python", __dir__)
TIMEOUT_SECONDS = 120

# Minimal configs for each registered model architecture.
# Each config should be just enough to instantiate and run a forward pass
# with small random weights.
MODEL_CONFIGS = {
  # === Standard transformer models (Llama-like) ===
  "llama" => {
    "model_type" => "llama", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128, "rms_norm_eps" => 1e-5,
  },
  "gemma" => {
    "model_type" => "gemma", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128, "head_dim" => 32,
    "rms_norm_eps" => 1e-6,
  },
  "gemma2" => {
    "model_type" => "gemma2", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128, "head_dim" => 32,
    "rms_norm_eps" => 1e-6, "attn_logit_softcapping" => 50.0,
    "final_logit_softcapping" => 30.0, "query_pre_attn_scalar" => 144.0,
  },
  "gemma3_text" => {
    "model_type" => "gemma3_text", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128, "head_dim" => 32,
    "rms_norm_eps" => 1e-6, "query_pre_attn_scalar" => 144.0,
    "sliding_window" => 64,
  },
  "qwen2" => {
    "model_type" => "qwen2", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128, "rms_norm_eps" => 1e-6,
    "tie_word_embeddings" => true,
  },
  "qwen3" => {
    "model_type" => "qwen3", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128, "rms_norm_eps" => 1e-6,
    "head_dim" => 32,
  },
  "qwen3_5" => {
    "model_type" => "qwen3_5", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128, "rms_norm_eps" => 1e-6,
    "head_dim" => 32,
  },
  "phi3" => {
    "model_type" => "phi3", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128, "rms_norm_eps" => 1e-5,
    "tie_word_embeddings" => false,
  },
  "starcoder2" => {
    "model_type" => "starcoder2", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128, "norm_epsilon" => 1e-5,
    "tie_word_embeddings" => true,
  },
  "cohere" => {
    "model_type" => "cohere", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128, "layer_norm_eps" => 1e-5,
    "logit_scale" => 0.0625,
  },
  "cohere2" => {
    "model_type" => "cohere2", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128, "layer_norm_eps" => 1e-5,
    "logit_scale" => 0.0625, "sliding_window" => 64,
  },
  "stablelm" => {
    "model_type" => "stablelm", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128, "rope_theta" => 10000.0,
    "use_qkv_bias" => false, "partial_rotary_factor" => 0.5,
    "layer_norm_eps" => 1e-5, "use_parallel_residual" => false,
  },
  "olmo" => {
    "model_type" => "olmo", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "intermediate_size" => 128, "vocab_size" => 128,
    "clip_qkv" => nil,
  },
  "olmo2" => {
    "model_type" => "olmo2", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128, "rms_norm_eps" => 1e-6,
    "tie_word_embeddings" => true,
  },
  "olmo3" => {
    "model_type" => "olmo3", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128, "rms_norm_eps" => 1e-6,
    "tie_word_embeddings" => true,
  },
  "internlm2" => {
    "model_type" => "internlm2", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128, "rms_norm_eps" => 1e-6,
    "tie_word_embeddings" => false,
  },
  "internlm3" => {
    "model_type" => "internlm3", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128, "rms_norm_eps" => 1e-6,
  },
  "gpt_neox" => {
    "model_type" => "gpt_neox", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "vocab_size" => 128, "layer_norm_eps" => 1e-5,
    "rotary_pct" => 0.25, "rotary_emb_base" => 10000,
    "use_parallel_residual" => true,
  },
  "gpt2" => {
    "model_type" => "gpt2", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "intermediate_size" => 128, "vocab_size" => 128,
    "max_position_embeddings" => 128, "layer_norm_epsilon" => 1e-5,
  },
  "gpt_bigcode" => {
    "model_type" => "gpt_bigcode", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "intermediate_size" => 128, "vocab_size" => 128,
    "max_position_embeddings" => 128, "layer_norm_epsilon" => 1e-5,
    "multi_query" => true,
  },
  "phi" => {
    "model_type" => "phi", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128, "partial_rotary_factor" => 0.5,
    "layer_norm_eps" => 1e-5,
  },
  "exaone" => {
    "model_type" => "exaone", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
  },
  "exaone4" => {
    "model_type" => "exaone4", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128, "rms_norm_eps" => 1e-6,
  },
  "granite" => {
    "model_type" => "granite", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
    "attention_multiplier" => 1.0, "embedding_multiplier" => 1.0,
    "residual_multiplier" => 1.0, "logits_scaling" => 1.0,
  },
  "dbrx" => {
    "model_type" => "dbrx", "d_model" => 64, "n_layers" => 2,
    "n_heads" => 2, "vocab_size" => 128,
    "attn_config" => { "kv_n_heads" => 2, "clip_qkv" => nil, "rope_theta" => 10000.0 },
    "ffn_config" => { "ffn_hidden_size" => 128, "moe_num_experts" => 4,
                      "moe_top_k" => 2, "moe_jitter_eps" => nil },
  },
  "plamo" => {
    "model_type" => "plamo", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
  },
  "openelm" => {
    "model_type" => "openelm", "model_dim" => 64, "num_transformer_layers" => 2,
    "num_query_heads" => [2, 2], "num_kv_heads" => [2, 2],
    "head_dim" => 32, "ffn_dim_divisor" => 2, "ffn_multipliers" => [2.0, 2.0],
    "vocab_size" => 128, "qkv_multipliers" => [1.0, 1.0],
    "normalize_qk_projections" => false,
  },
  "helium" => {
    "model_type" => "helium", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
  },
  "nemotron" => {
    "model_type" => "nemotron", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
  },
  "solar_open" => {
    "model_type" => "solar_open", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
  },
  "hunyuan" => {
    "model_type" => "hunyuan", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128, "rms_norm_eps" => 1e-6,
    "use_cla" => false, "cla_share_factor" => 1,
  },
  "hunyuan_v1_dense" => {
    "model_type" => "hunyuan_v1_dense", "hidden_size" => 64,
    "num_hidden_layers" => 2, "num_attention_heads" => 2,
    "num_key_value_heads" => 2, "intermediate_size" => 128, "vocab_size" => 128,
    "rms_norm_eps" => 1e-6, "kv_lora_rank" => 16, "q_lora_rank" => 32,
    "qk_rope_head_dim" => 16, "v_head_dim" => 32, "qk_nope_head_dim" => 16,
  },
  "telechat3" => {
    "model_type" => "telechat3", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
  },
  "nanochat" => {
    "model_type" => "nanochat", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
  },
  "smollm3" => {
    "model_type" => "smollm3", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
  },
  "glm" => {
    "model_type" => "glm", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128, "head_dim" => 32,
  },
  "glm4" => {
    "model_type" => "glm4", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
  },
  "dots1" => {
    "model_type" => "dots1", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
  },
  "gpt_oss" => {
    "model_type" => "gpt_oss", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
  },
  "apertus" => {
    "model_type" => "apertus", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
  },
  "youtu_llm" => {
    "model_type" => "youtu_llm", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
  },
  "ernie4_5" => {
    "model_type" => "ernie4_5", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
  },
  "iquestloopcoder" => {
    "model_type" => "iquestloopcoder", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
  },
  "qwen3_next" => {
    "model_type" => "qwen3_next", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128, "head_dim" => 32,
  },
  "afm7" => {
    "model_type" => "afm7", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
  },

  # === MoE models ===
  "mixtral" => {
    "model_type" => "mixtral", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
    "num_local_experts" => 4, "num_experts_per_tok" => 2,
  },
  "qwen2_moe" => {
    "model_type" => "qwen2_moe", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
    "num_experts" => 4, "num_experts_per_tok" => 2,
    "shared_expert_intermediate_size" => 128,
    "moe_intermediate_size" => 64,
  },
  "qwen3_moe" => {
    "model_type" => "qwen3_moe", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128, "head_dim" => 32,
    "num_experts" => 4, "num_experts_per_tok" => 2,
    "shared_expert_intermediate_size" => 128,
    "moe_intermediate_size" => 64,
  },
  "qwen3_5_moe" => {
    "model_type" => "qwen3_5_moe", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128, "head_dim" => 32,
    "num_experts" => 4, "num_experts_per_tok" => 2,
    "shared_expert_intermediate_size" => 128,
    "moe_intermediate_size" => 64,
  },
  "deepseek" => {
    "model_type" => "deepseek", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128, "rms_norm_eps" => 1e-6,
  },
  "deepseek_v2" => {
    "model_type" => "deepseek_v2", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
    "kv_lora_rank" => 16, "q_lora_rank" => 32,
    "qk_rope_head_dim" => 16, "v_head_dim" => 32, "qk_nope_head_dim" => 16,
    "moe_intermediate_size" => 64,
  },
  "deepseek_v3" => {
    "model_type" => "deepseek_v3", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
    "kv_lora_rank" => 16, "q_lora_rank" => 32,
    "qk_rope_head_dim" => 16, "v_head_dim" => 32, "qk_nope_head_dim" => 16,
    "moe_intermediate_size" => 64,
  },
  "deepseek_v32" => {
    "model_type" => "deepseek_v32", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
    "kv_lora_rank" => 16, "q_lora_rank" => 32,
    "qk_rope_head_dim" => 16, "v_head_dim" => 32, "qk_nope_head_dim" => 16,
    "moe_intermediate_size" => 64,
  },
  "olmoe" => {
    "model_type" => "olmoe", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
    "num_experts" => 4, "num_experts_per_tok" => 2,
  },
  "phimoe" => {
    "model_type" => "phimoe", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
    "num_local_experts" => 4, "num_experts_per_tok" => 2,
  },
  "granitemoe" => {
    "model_type" => "granitemoe", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
    "num_local_experts" => 4, "num_experts_per_tok" => 2,
    "attention_multiplier" => 1.0, "embedding_multiplier" => 1.0,
    "residual_multiplier" => 1.0, "logits_scaling" => 1.0,
  },
  "afmoe" => {
    "model_type" => "afmoe", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
    "num_local_experts" => 4, "num_experts_per_tok" => 2,
  },
  "glm4_moe" => {
    "model_type" => "glm4_moe", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
    "num_experts" => 4, "num_experts_per_tok" => 2,
    "shared_expert_intermediate_size" => 128,
    "moe_intermediate_size" => 64,
  },
  "glm4_moe_lite" => {
    "model_type" => "glm4_moe_lite", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
    "num_experts" => 4, "num_experts_per_tok" => 2,
    "shared_expert_intermediate_size" => 128,
    "moe_intermediate_size" => 64,
  },
  "glm_moe_dsa" => {
    "model_type" => "glm_moe_dsa", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
    "kv_lora_rank" => 16, "q_lora_rank" => 32,
    "qk_rope_head_dim" => 16, "v_head_dim" => 32, "qk_nope_head_dim" => 16,
    "moe_intermediate_size" => 64,
  },
  "ernie4_5_moe" => {
    "model_type" => "ernie4_5_moe", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
    "num_experts" => 4, "num_experts_per_tok" => 2,
    "moe_intermediate_size" => 64,
  },
  "exaone_moe" => {
    "model_type" => "exaone_moe", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
    "num_local_experts" => 4, "num_experts_per_tok" => 2,
    "moe_intermediate_size" => 64,
  },
  "qwen3_vl_moe" => {
    "model_type" => "qwen3_vl_moe", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128, "head_dim" => 32,
    "num_experts" => 4, "num_experts_per_tok" => 2,
    "shared_expert_intermediate_size" => 128,
    "moe_intermediate_size" => 64,
  },
  "lfm2_moe" => {
    "model_type" => "lfm2_moe", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
    "num_local_experts" => 4, "num_experts_per_tok" => 2,
    "moe_intermediate_size" => 64,
  },

  # === SSM/Mamba models ===
  "mamba" => {
    "model_type" => "mamba", "hidden_size" => 64, "num_hidden_layers" => 2,
    "intermediate_size" => 128, "state_size" => 16, "conv_kernel" => 4,
    "vocab_size" => 128, "use_bias" => false, "use_conv_bias" => true,
  },
  "mamba2" => {
    "model_type" => "mamba2", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_heads" => 2, "head_dim" => 32, "state_size" => 16,
    "conv_kernel" => 4, "n_groups" => 1, "vocab_size" => 128,
  },

  # === Composite/Wrapper models ===
  "gemma3" => {
    "model_type" => "gemma3",
    "text_config" => {
      "model_type" => "gemma3_text", "hidden_size" => 64,
      "num_hidden_layers" => 2, "num_attention_heads" => 2,
      "num_key_value_heads" => 2, "intermediate_size" => 128,
      "vocab_size" => 128, "head_dim" => 32, "rms_norm_eps" => 1e-6,
      "query_pre_attn_scalar" => 144.0, "sliding_window" => 64,
    },
  },
  "gemma3n" => {
    "model_type" => "gemma3n",
    "text_config" => {
      "model_type" => "gemma3_text", "hidden_size" => 64,
      "num_hidden_layers" => 2, "num_attention_heads" => 2,
      "num_key_value_heads" => 2, "intermediate_size" => 128,
      "vocab_size" => 128, "head_dim" => 32, "rms_norm_eps" => 1e-6,
      "query_pre_attn_scalar" => 144.0, "sliding_window" => 64,
    },
  },
  "mistral3" => {
    "model_type" => "mistral3",
    "text_config" => {
      "model_type" => "llama", "hidden_size" => 64, "num_hidden_layers" => 2,
      "num_attention_heads" => 2, "num_key_value_heads" => 2,
      "intermediate_size" => 128, "vocab_size" => 128,
      "tie_word_embeddings" => false,
    },
  },
  "ministral3" => {
    "model_type" => "ministral3", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
  },
  "pixtral" => {
    "model_type" => "pixtral",
    "text_config" => {
      "model_type" => "llama", "hidden_size" => 64, "num_hidden_layers" => 2,
      "num_attention_heads" => 2, "num_key_value_heads" => 2,
      "intermediate_size" => 128, "vocab_size" => 128,
      "tie_word_embeddings" => false,
    },
  },
  "llama4" => {
    "model_type" => "llama4",
    "text_config" => {
      "model_type" => "llama4_text", "hidden_size" => 64,
      "num_hidden_layers" => 2, "num_attention_heads" => 2,
      "num_key_value_heads" => 2, "intermediate_size" => 128,
      "vocab_size" => 128, "no_rope_layers" => [],
    },
  },
  "llama4_text" => {
    "model_type" => "llama4_text", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128, "no_rope_layers" => [],
  },
  "kimi_vl" => {
    "model_type" => "kimi_vl",
    "text_config" => {
      "model_type" => "deepseek", "hidden_size" => 64, "num_hidden_layers" => 2,
      "num_attention_heads" => 2, "num_key_value_heads" => 2,
      "intermediate_size" => 128, "vocab_size" => 128,
    },
  },
  "kimi_k25" => {
    "model_type" => "kimi_k25",
    "text_config" => {
      "model_type" => "deepseek", "hidden_size" => 64, "num_hidden_layers" => 2,
      "num_attention_heads" => 2, "num_key_value_heads" => 2,
      "intermediate_size" => 128, "vocab_size" => 128,
    },
  },
  "kimi_linear" => {
    "model_type" => "kimi_linear",
    "text_config" => {
      "model_type" => "deepseek", "hidden_size" => 64, "num_hidden_layers" => 2,
      "num_attention_heads" => 2, "num_key_value_heads" => 2,
      "intermediate_size" => 128, "vocab_size" => 128,
    },
  },
  "qwen2_vl" => {
    "model_type" => "qwen2_vl",
    "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
  },
  "qwen3_vl" => {
    "model_type" => "qwen3_vl",
    "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128, "head_dim" => 32,
  },
  "lfm2" => {
    "model_type" => "lfm2", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
  },
  "lfm2-vl" => {
    "model_type" => "lfm2-vl",
    "text_config" => {
      "model_type" => "lfm2", "hidden_size" => 64, "num_hidden_layers" => 2,
      "num_attention_heads" => 2, "num_key_value_heads" => 2,
      "intermediate_size" => 128, "vocab_size" => 128,
    },
  },

  # === Hybrid models (Mamba+Attention) ===
  "falcon_h1" => {
    "model_type" => "falcon_h1", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128, "head_dim" => 32,
    "mamba_d_conv" => 4,
  },
  "jamba" => {
    "model_type" => "jamba", "hidden_size" => 64,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128, "head_dim" => 32,
    "mamba_d_conv" => 4, "mamba_d_state" => 16, "mamba_expand" => 2,
    "layers_block_type" => ["mamba", "attention"],
    "num_experts" => 1, "num_experts_per_tok" => 1,
  },
  "recurrent_gemma" => {
    "model_type" => "recurrent_gemma", "hidden_size" => 64,
    "num_hidden_layers" => 2, "num_attention_heads" => 2,
    "num_key_value_heads" => 2, "intermediate_size" => 128,
    "vocab_size" => 128, "head_dim" => 32,
    "block_types" => ["recurrent", "attention"],
    "attention_window_size" => 64,
  },
  "granitemoehybrid" => {
    "model_type" => "granitemoehybrid", "hidden_size" => 64,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
    "mamba_d_conv" => 4, "mamba_d_state" => 16, "mamba_expand" => 2,
    "num_hidden_layers" => 2,
    "layer_types" => ["mamba", "attention"],
    "ssm_cfg" => { "state_size" => 16, "conv_kernel" => 4, "expand" => 2 },
    "attention_multiplier" => 1.0, "embedding_multiplier" => 1.0,
    "residual_multiplier" => 1.0, "logits_scaling" => 1.0,
  },
  "nemotron_h" => {
    "model_type" => "nemotron_h", "hidden_size" => 64,
    "num_hidden_layers" => 2, "num_attention_heads" => 2,
    "num_key_value_heads" => 2, "intermediate_size" => 128,
    "vocab_size" => 128, "head_dim" => 32,
    "block_types" => ["recurrent", "attention"],
    "mamba_d_conv" => 4,
  },
  "minimax" => {
    "model_type" => "minimax", "hidden_size" => 64,
    "num_hidden_layers" => 2, "num_attention_heads" => 2,
    "num_key_value_heads" => 2, "intermediate_size" => 128,
    "vocab_size" => 128,
    "block_types" => ["attention", "attention"],
  },

  # === Models with special architectures ===
  "phi3small" => {
    "model_type" => "phi3small", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
    "blocksparse_block_size" => 64, "dense_attention_every_n_layers" => 1,
    "gegelu_limit" => 20.0,
  },
  "phixtral" => {
    "model_type" => "phixtral", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
    "num_local_experts" => 4, "num_experts_per_tok" => 2,
    "partial_rotary_factor" => 0.5,
  },
  "bitnet" => {
    "model_type" => "bitnet", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
  },
  "minicpm" => {
    "model_type" => "minicpm", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
    "scale_depth" => 1.0, "scale_emb" => 1,
    "dim_model_base" => 64,
  },
  "minicpm3" => {
    "model_type" => "minicpm3", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
    "qk_nope_head_dim" => 16, "qk_rope_head_dim" => 16,
    "scale_depth" => 1.0, "scale_emb" => 1,
    "dim_model_base" => 64, "q_lora_rank" => 32,
  },
  "plamo2" => {
    "model_type" => "plamo2", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
    "mamba_d_conv" => 4, "mamba_d_state" => 16, "mamba_expand" => 2,
  },
  "rwkv7" => {
    "model_type" => "rwkv7", "hidden_size" => 64, "num_hidden_layers" => 2,
    "intermediate_size" => 128, "vocab_size" => 128, "head_dim" => 32,
  },
  "Klear" => {
    "model_type" => "Klear", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
    "kv_lora_rank" => 16, "q_lora_rank" => 32,
    "qk_rope_head_dim" => 16, "v_head_dim" => 32, "qk_nope_head_dim" => 16,
  },
  "seed_oss" => {
    "model_type" => "seed_oss", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
    "kv_lora_rank" => 16, "q_lora_rank" => 32,
    "qk_rope_head_dim" => 16, "v_head_dim" => 32, "qk_nope_head_dim" => 16,
    "moe_intermediate_size" => 64,
  },
  "longcat_flash" => {
    "model_type" => "longcat_flash", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
  },
  "longcat_flash_ngram" => {
    "model_type" => "longcat_flash_ngram", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
  },
  "mimo" => {
    "model_type" => "mimo", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
  },
  "mimo_v2_flash" => {
    "model_type" => "mimo_v2_flash", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
  },
  "qwen" => {
    "model_type" => "qwen", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "intermediate_size" => 128, "vocab_size" => 128,
    "seq_length" => 128, "kv_channels" => 32,
  },
  "nemotron-nas" => {
    "model_type" => "nemotron-nas", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
    "block_configs" => [
      { "block_type" => "attention", "num_attention_heads" => 2,
        "num_key_value_heads" => 2, "intermediate_size" => 128 },
      { "block_type" => "attention", "num_attention_heads" => 2,
        "num_key_value_heads" => 2, "intermediate_size" => 128 },
    ],
  },
  "step3p5" => {
    "model_type" => "step3p5", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_attention_groups" => 2,
    "head_dim" => 32, "intermediate_size" => 128, "vocab_size" => 128,
    "moe_num_experts" => 4, "moe_top_k" => 2, "moe_intermediate_size" => 64,
    "share_expert_dim" => 64,
  },
  "lille-130m" => {
    "model_type" => "lille-130m", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
    "quantize_method" => nil,
  },
  "baichuan_m1" => {
    "model_type" => "baichuan_m1", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
    "mamba_d_conv" => 4, "mamba_d_state" => 16, "mamba_expand" => 2,
    "block_types" => ["attention", "attention"],
  },
  "bailing_moe" => {
    "model_type" => "bailing_moe", "hidden_size" => 64, "num_hidden_layers" => 2,
    "num_attention_heads" => 2, "num_key_value_heads" => 2,
    "intermediate_size" => 128, "vocab_size" => 128,
    "num_experts" => 4, "num_experts_per_tok" => 2,
    "moe_intermediate_size" => 64,
  },
  "bailing_moe_linear" => {
    "model_type" => "bailing_moe_linear", "hidden_size" => 64,
    "num_hidden_layers" => 2, "num_attention_heads" => 2,
    "num_key_value_heads" => 2, "intermediate_size" => 128,
    "vocab_size" => 128, "num_experts" => 4, "num_experts_per_tok" => 2,
    "moe_intermediate_size" => 64,
  },
}

def ruby_forward_pass(model_type, config)
  mx = MLX::Core

  model_class, args_class = MlxLm::Models.get_classes(config)
  args = args_class.from_dict(config)
  model = model_class.new(args)

  # Evaluate random parameters
  params = MLX::Utils.tree_flatten(model.parameters)
  params.each { |_, v| mx.eval(v) }

  tokens = mx.array([[1, 2, 3]], dtype: mx.int32)

  # Warm up
  output = model.call(tokens)
  mx.eval(output)

  # Timed forward pass
  start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  output = model.call(tokens)
  mx.eval(output)
  elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - start

  { time: elapsed, shape: output.shape }
end

def python_forward_pass(model_type, config)
  config_json = JSON.dump(config)
  python_code = <<~PYTHON
    import json, sys, time
    import mlx.core as mx
    import mlx.nn as nn

    config = json.loads(#{config_json.inspect})
    model_type_key = config.get("model_type", "")

    # Dynamically import the right model
    try:
        mod = __import__(f"mlx_lm.models.{model_type_key}", fromlist=["Model", "ModelArgs"])
        ModelClass = mod.Model
        ModelArgsClass = mod.ModelArgs
    except (ImportError, AttributeError) as e:
        print(json.dumps({"error": f"Import failed: {e}"}))
        sys.exit(0)

    try:
        args = ModelArgsClass(**{k: v for k, v in config.items()})
    except TypeError as e:
        # Try from_dict if available
        if hasattr(ModelArgsClass, 'from_dict'):
            args = ModelArgsClass.from_dict(config)
        else:
            print(json.dumps({"error": f"Args init failed: {e}"}))
            sys.exit(0)

    try:
        model = ModelClass(args)
    except Exception as e:
        print(json.dumps({"error": f"Model init failed: {e}"}))
        sys.exit(0)

    # Evaluate parameters
    mx.eval(model.parameters())

    tokens = mx.array([[1, 2, 3]], dtype=mx.int32)

    try:
        # Warm up
        output = model(tokens)
        mx.eval(output)

        # Timed forward pass
        start = time.monotonic()
        output = model(tokens)
        mx.eval(output)
        elapsed = time.monotonic() - start

        print(json.dumps({
            "time": elapsed,
            "shape": list(output.shape),
        }))
    except Exception as e:
        print(json.dumps({"error": f"Forward pass failed: {e}"}))
  PYTHON

  stdout, stderr, status = nil, nil, nil
  begin
    Timeout.timeout(TIMEOUT_SECONDS) do
      stdout, stderr, status = Open3.capture3(
        { "MLX_DISABLE_COMPILE" => "1" },
        PYTHON, "-c", python_code
      )
    end
  rescue Timeout::Error
    return { "error" => "TIMEOUT (#{TIMEOUT_SECONDS}s)" }
  end

  return { "error" => "Python process failed: #{stderr}" } unless status.success? || !stdout.strip.empty?

  begin
    JSON.parse(stdout.strip)
  rescue JSON::ParserError
    { "error" => "JSON parse error. stdout=#{stdout[0..200]}, stderr=#{stderr[0..200]}" }
  end
end

# ===== Main benchmark runner =====

results = []
all_models = MODEL_CONFIGS.keys.sort
total = all_models.size
passed = 0
failed = 0
skipped = 0

puts "=" * 80
puts "Python vs Ruby Forward Pass Benchmark"
puts "=" * 80
puts "Models to benchmark: #{total}"
puts "Sequence: [1, 2, 3] (batch=1, seq_len=3)"
puts "-" * 80
printf "%-25s %12s %12s %8s %s\n", "Model", "Python (ms)", "Ruby (ms)", "Ratio", "Status"
puts "-" * 80

all_models.each_with_index do |model_type, idx|
  config = MODEL_CONFIGS[model_type]
  $stdout.write "[#{idx + 1}/#{total}] #{model_type}... "
  $stdout.flush

  ruby_result = nil
  python_result = nil
  status = nil

  # Run Ruby forward pass
  begin
    Timeout.timeout(TIMEOUT_SECONDS) do
      ruby_result = ruby_forward_pass(model_type, config)
    end
  rescue Timeout::Error
    ruby_result = { error: "TIMEOUT (#{TIMEOUT_SECONDS}s)" }
  rescue => e
    ruby_result = { error: "#{e.class}: #{e.message}" }
  end

  # Run Python forward pass
  python_result = python_forward_pass(model_type, config)

  ruby_time_ms = ruby_result.is_a?(Hash) && ruby_result[:time] ? (ruby_result[:time] * 1000).round(2) : nil
  python_time_ms = python_result.is_a?(Hash) && python_result["time"] ? (python_result["time"] * 1000).round(2) : nil

  ruby_err = ruby_result.is_a?(Hash) ? ruby_result[:error] : nil
  python_err = python_result.is_a?(Hash) ? python_result["error"] : nil

  if ruby_err && python_err
    status = "BOTH_FAIL"
    skipped += 1
  elsif ruby_err
    status = "RUBY_FAIL"
    failed += 1
  elsif python_err
    status = "PY_FAIL"
    failed += 1
  else
    status = "PASS"
    passed += 1
  end

  ratio = (ruby_time_ms && python_time_ms && python_time_ms > 0) ? (ruby_time_ms / python_time_ms).round(2) : nil

  ruby_str = ruby_time_ms ? format("%.2f", ruby_time_ms) : (ruby_err || "N/A")[0..11]
  python_str = python_time_ms ? format("%.2f", python_time_ms) : (python_err || "N/A")[0..11]
  ratio_str = ratio ? format("%.2fx", ratio) : "N/A"

  result = {
    model: model_type,
    python_time_ms: python_time_ms,
    ruby_time_ms: ruby_time_ms,
    ratio: ratio,
    python_shape: python_result.is_a?(Hash) ? python_result["shape"] : nil,
    ruby_shape: ruby_result.is_a?(Hash) && ruby_result[:shape] ? ruby_result[:shape] : nil,
    python_error: python_err,
    ruby_error: ruby_err,
    status: status,
  }
  results << result

  # Print a clean result line
  puts ""
  printf "  %-25s %12s %12s %8s %s\n",
    model_type, python_str, ruby_str, ratio_str, status
  if ruby_err
    puts "    Ruby error: #{ruby_err[0..100]}"
  end
  if python_err
    puts "    Python error: #{python_err[0..100]}"
  end
end

# ===== Summary =====
puts ""
puts "=" * 80
puts "SUMMARY"
puts "=" * 80
puts "Total models: #{total}"
puts "Passed: #{passed}"
puts "Failed: #{failed}"
puts "Both failed: #{skipped}"
puts ""

passing = results.select { |r| r[:status] == "PASS" }
if passing.any?
  avg_ratio = passing.map { |r| r[:ratio] }.compact.sum / passing.size
  puts "Average Ruby/Python ratio: #{format('%.2f', avg_ratio)}x (for #{passing.size} passing models)"
  puts ""
  puts "Fastest Ruby models (lowest Ruby time):"
  passing.sort_by { |r| r[:ruby_time_ms] || Float::INFINITY }.first(5).each do |r|
    printf "  %-25s Ruby: %8.2f ms  Python: %8.2f ms  Ratio: %.2fx\n",
      r[:model], r[:ruby_time_ms], r[:python_time_ms], r[:ratio]
  end
  puts ""
  puts "Slowest Ruby models (highest Ruby time):"
  passing.sort_by { |r| -(r[:ruby_time_ms] || 0) }.first(5).each do |r|
    printf "  %-25s Ruby: %8.2f ms  Python: %8.2f ms  Ratio: %.2fx\n",
      r[:model], r[:ruby_time_ms], r[:python_time_ms], r[:ratio]
  end
end

# Failed models detail
ruby_failures = results.select { |r| r[:ruby_error] }
if ruby_failures.any?
  puts ""
  puts "Ruby failures:"
  ruby_failures.each do |r|
    puts "  #{r[:model]}: #{r[:ruby_error]}"
  end
end

python_failures = results.select { |r| r[:python_error] }
if python_failures.any?
  puts ""
  puts "Python failures:"
  python_failures.each do |r|
    puts "  #{r[:model]}: #{r[:python_error]}"
  end
end

# Write JSON report
report_path = File.expand_path("reports/forward_pass_benchmark.json", __dir__)
FileUtils.mkdir_p(File.dirname(report_path))
File.write(report_path, JSON.pretty_generate({
  timestamp: Time.now.iso8601,
  total_models: total,
  passed: passed,
  failed: failed,
  both_failed: skipped,
  average_ratio: passing.any? ? (passing.map { |r| r[:ratio] }.compact.sum / passing.size).round(3) : nil,
  results: results,
}))
puts ""
puts "Report saved to: #{report_path}"
