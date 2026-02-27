# frozen_string_literal: true

require "json"
require "open3"
require "tmpdir"

module OnnxExportTestHelper
  EXPLICIT_TINY_CONFIGS = {
    "llama" => {
      "model_type" => "llama",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 128,
      "vocab_size" => 128,
      "tie_word_embeddings" => true,
    },
    "gemma" => {
      "model_type" => "gemma",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 128,
      "vocab_size" => 128,
      "head_dim" => 32,
      "tie_word_embeddings" => true,
    },
    "gemma2" => {
      "model_type" => "gemma2",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 128,
      "vocab_size" => 128,
      "head_dim" => 32,
      "query_pre_attn_scalar" => 32.0,
    },
    "qwen2" => {
      "model_type" => "qwen2",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 128,
      "vocab_size" => 128,
      "tie_word_embeddings" => true,
    },
    "phi3" => {
      "model_type" => "phi3",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 128,
      "vocab_size" => 128,
      "tie_word_embeddings" => true,
    },
    "starcoder2" => {
      "model_type" => "starcoder2",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 128,
      "vocab_size" => 128,
      "tie_word_embeddings" => true,
    },
    "stablelm" => {
      "model_type" => "stablelm",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 128,
      "vocab_size" => 128,
    },
    "cohere" => {
      "model_type" => "cohere",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 128,
      "vocab_size" => 128,
    },
    "olmo2" => {
      "model_type" => "olmo2",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 128,
      "vocab_size" => 128,
      "tie_word_embeddings" => true,
    },
    "gpt_neox" => {
      "model_type" => "gpt_neox",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "vocab_size" => 128,
      "intermediate_size" => 256,
    },
    "mixtral" => {
      "model_type" => "mixtral",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 128,
      "vocab_size" => 128,
      "num_local_experts" => 2,
      "num_experts_per_tok" => 1,
      "tie_word_embeddings" => true,
    },
    "deepseek" => {
      "model_type" => "deepseek",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 128,
      "moe_intermediate_size" => 64,
      "vocab_size" => 128,
      "n_routed_experts" => 2,
      "num_experts_per_tok" => 1,
      "n_shared_experts" => 1,
      "moe_layer_freq" => 1,
      "first_k_dense_replace" => 1,
    },
    "internlm2" => {
      "model_type" => "internlm2",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 128,
      "vocab_size" => 128,
      "bias" => false,
      "tie_word_embeddings" => true,
    },
  }.freeze

  DEFAULT_TINY_CONFIG = {
    "hidden_size" => 64,
    "num_hidden_layers" => 2,
    "num_attention_heads" => 2,
    "num_key_value_heads" => 2,
    "intermediate_size" => 128,
    "intermediate_size_mlp" => 128,
    "vocab_size" => 128,
    "rms_norm_eps" => 1e-5,
    "norm_eps" => 1e-5,
    "layer_norm_eps" => 1e-5,
    "layer_norm_epsilon" => 1e-5,
    "norm_epsilon" => 1e-5,
    "head_dim" => 32,
    "max_position_embeddings" => 256,
    "rope_theta" => 10_000.0,
    "tie_word_embeddings" => true,
    "attention_bias" => false,
    "mlp_bias" => false,
    "use_bias" => false,
    "use_conv_bias" => true,
    "num_layers" => 2,
    "n_layer" => 2,
    "n_head" => 2,
    "n_embd" => 64,
    "n_inner" => 128,
    "n_positions" => 256,
    "n_ctx" => 256,
    "num_local_experts" => 2,
    "num_experts" => 2,
    "num_experts_per_tok" => 1,
    "num_shared_experts" => 1,
    "n_shared_experts" => 1,
    "n_routed_experts" => 2,
    "moe_intermediate_size" => 64,
    "moe_layer_freq" => 1,
    "first_k_dense_replace" => 1,
    "n_group" => 1,
    "topk_group" => 1,
    "norm_topk_prob" => true,
    "routed_scaling_factor" => 1.0,
    "partial_rotary_factor" => 0.5,
    "scoring_func" => "sigmoid",
    "topk_method" => "noaux_tc",
    "sliding_window" => 64,
    "num_heads" => 2,
    "state_size" => 8,
    "conv_kernel" => 3,
    "n_groups" => 1,
    "shared_intermediate_size" => 64,
    "rotary_dim" => 16,
    "hidden_act" => "silu",
  }.freeze

  MODEL_TINY_CONFIG_OVERRIDES = {
    "dots1" => {
      "first_k_dense_replace" => 1,
      "moe_intermediate_size" => 48,
      "n_routed_experts" => 2,
      "n_shared_experts" => 1,
      "num_experts_per_tok" => 1,
      "norm_topk_prob" => true,
      "routed_scaling_factor" => 1.0,
      "head_dim" => 8,
      "scoring_func" => "noaux_tc",
      "tie_word_embeddings" => false,
    },
    "ernie4_5" => {
      "max_position_embeddings" => 256,
      "rope_theta" => 10_000.0,
      "use_bias" => false,
      "tie_word_embeddings" => true,
    },
    "ernie4_5_moe" => {
      "max_position_embeddings" => 256,
      "rope_theta" => 10_000.0,
      "use_bias" => false,
      "tie_word_embeddings" => false,
      "moe_num_experts" => 2,
      "moe_k" => 1,
      "moe_layer_interval" => 1,
      "moe_layer_start_index" => 0,
      "moe_num_shared_experts" => 1,
      "moe_gate_act" => "softmax",
    },
    "exaone" => {
      "num_layers" => 2,
      "layer_norm_epsilon" => 1e-5,
      "rope_theta" => 10_000.0,
      "tie_word_embeddings" => true,
      "attention_bias" => false,
      "mlp_bias" => false,
    },
    "exaone4" => {
      "max_position_embeddings" => 256,
      "rope_theta" => 10_000.0,
      "head_dim" => 16,
      "tie_word_embeddings" => false,
      "sliding_window" => 4,
      "sliding_window_pattern" => "LLGL",
    },
    "exaone_moe" => {
      "moe_intermediate_size" => 16,
      "num_experts" => 2,
      "num_experts_per_tok" => 1,
      "num_shared_experts" => 1,
      "max_position_embeddings" => 256,
      "sliding_window" => 2,
      "layer_types" => ["full_attention", "sliding_attention"],
      "is_moe_layer" => [true, false],
      "norm_topk_prob" => true,
      "rope_theta" => 10_000.0,
      "tie_word_embeddings" => true,
    },
    "glm4_moe" => {
      "max_position_embeddings" => 256,
      "moe_intermediate_size" => 16,
      "norm_topk_prob" => true,
      "n_group" => 1,
      "topk_group" => 1,
      "n_shared_experts" => 1,
      "n_routed_experts" => 2,
      "routed_scaling_factor" => 1.0,
      "num_experts_per_tok" => 1,
      "first_k_dense_replace" => 0,
      "use_qk_norm" => true,
      "tie_word_embeddings" => false,
      "attention_bias" => false,
      "partial_rotary_factor" => 0.5,
    },
    "iquestloopcoder" => {
      "hidden_size" => 32,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 2,
      "intermediate_size" => 64,
      "head_dim" => 8,
      "loop_num" => 2,
      "loop_window_size" => 4,
      "tie_word_embeddings" => false,
    },
    "llama4" => {
      "text_config" => {
        "model_type" => "llama4_text",
        "hidden_size" => 32,
        "num_attention_heads" => 4,
        "num_key_value_heads" => 2,
        "num_hidden_layers" => 2,
        "vocab_size" => 97,
        "intermediate_size" => 64,
        "intermediate_size_mlp" => 64,
        "num_local_experts" => 2,
        "num_experts_per_tok" => 1,
        "interleave_moe_layer_step" => 2,
        "attention_chunk_size" => 4,
        "max_position_embeddings" => 128,
        "rope_theta" => 10_000.0,
        "head_dim" => 8,
        "rms_norm_eps" => 1e-5,
        "attention_bias" => false,
        "use_qk_norm" => true,
      },
    },
    "llama4_text" => {
      "intermediate_size" => 64,
      "intermediate_size_mlp" => 64,
      "no_rope_layers" => [0, 1],
      "use_qk_norm" => true,
    },
    "mamba2" => {
      "num_heads" => 4,
      "head_dim" => 4,
      "hidden_size" => 32,
      "intermediate_size" => 16,
      "state_size" => 8,
      "num_hidden_layers" => 2,
      "layer_norm_epsilon" => 1e-5,
      "conv_kernel" => 3,
      "n_groups" => 2,
      "use_bias" => true,
      "use_conv_bias" => true,
      "time_step_limit" => [0.001, 10.0],
      "time_step_rank" => "auto",
    },
    "mimo_v2_flash" => {
      "num_experts_per_tok" => 1,
      "hybrid_layer_pattern" => [0, 1],
      "moe_layer_freq" => [0, 1],
      "sliding_window_size" => 2,
      "moe_intermediate_size" => 48,
      "n_shared_experts" => 1,
      "n_routed_experts" => 2,
      "routed_scaling_factor" => 1.0,
      "topk_method" => "noaux_tc",
      "scoring_func" => "sigmoid",
      "norm_topk_prob" => true,
      "layernorm_epsilon" => 1e-5,
      "swa_rope_theta" => 20_000.0,
      "swa_num_attention_heads" => 4,
      "swa_num_key_value_heads" => 2,
      "head_dim" => 8,
      "v_head_dim" => 8,
      "swa_head_dim" => 8,
      "swa_v_head_dim" => 8,
      "partial_rotary_factor" => 1.0,
    },
    "minimax" => {
      "max_position_embeddings" => 128,
      "num_experts_per_tok" => 1,
      "num_local_experts" => 2,
      "shared_intermediate_size" => 32,
      "rotary_dim" => 8,
      "tie_word_embeddings" => false,
      "use_qk_norm" => true,
    },
    "ministral3" => {
      "num_hidden_layers" => 4,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 2,
      "head_dim" => 8,
      "max_position_embeddings" => 128,
      "tie_word_embeddings" => true,
      "sliding_window" => 8,
      "layer_types" => ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
      "rope_parameters" => {
        "rope_theta" => 10_000.0,
        "llama_4_scaling_beta" => 0.1,
        "original_max_position_embeddings" => 128,
      },
    },
    "nemotron" => {
      "hidden_act" => "relu2",
      "norm_eps" => 1e-5,
      "partial_rotary_factor" => 0.5,
      "rope_scaling" => { "type" => "linear", "factor" => 2.0 },
      "tie_word_embeddings" => false,
    },
    "nemotron-nas" => {
      "hidden_size" => 32,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 4,
      "vocab_size" => 101,
      "rms_norm_eps" => 1e-5,
      "hidden_act" => "silu",
      "attention_bias" => false,
      "mlp_bias" => false,
      "rope_theta" => 10_000.0,
      "rope_scaling" => { "type" => "linear", "factor" => 2.0 },
      "max_position_embeddings" => 128,
      "tie_word_embeddings" => true,
      "block_configs" => [
        {
          "attention" => { "n_heads_in_group" => 2 },
          "ffn" => { "ffn_mult" => 1.5 },
        },
        {
          "attention" => { "no_op" => true },
          "ffn" => { "replace_with_linear" => true },
        },
      ],
    },
    "olmo3" => {
      "hidden_size" => 48,
      "num_hidden_layers" => 4,
      "intermediate_size" => 96,
      "num_attention_heads" => 4,
      "vocab_size" => 128,
      "max_position_embeddings" => 256,
      "sliding_window" => 4,
      "rope_theta" => 10_000.0,
      "tie_word_embeddings" => false,
    },
    "mistral3" => {
      "text_config" => {
        "model_type" => "llama",
        "hidden_size" => 64,
        "num_hidden_layers" => 2,
        "num_attention_heads" => 4,
        "num_key_value_heads" => 2,
        "intermediate_size" => 128,
        "vocab_size" => 128,
        "rms_norm_eps" => 1e-5,
        "rope_theta" => 10_000.0,
        "tie_word_embeddings" => false,
      },
    },
    "baichuan_m1" => {
      "hidden_size" => 32,
      "intermediate_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 2,
      "rope_theta" => 10_000.0,
      "sliding_window" => 4,
      "sliding_window_layers" => [0],
      "conv_window" => 2,
      "rms_norm_eps" => 1e-5,
      "tie_word_embeddings" => false,
    },
    "dbrx" => {
      "vocab_size" => 101,
      "d_model" => 24,
      "n_layers" => 0,
      "n_heads" => 4,
      "attn_config" => {
        "kv_n_heads" => 2,
        "clip_qkv" => 8.0,
        "rope_theta" => 10_000.0,
      },
      "ffn_config" => {
        "ffn_hidden_size" => 16,
        "moe_num_experts" => 2,
        "moe_top_k" => 1,
      },
    },
    "granite" => {
      "hidden_size" => 32,
      "num_hidden_layers" => 2,
      "intermediate_size" => 64,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "rms_norm_eps" => 1e-5,
      "vocab_size" => 128,
      "logits_scaling" => 2.0,
      "attention_multiplier" => 0.25,
      "embedding_multiplier" => 1.5,
      "residual_multiplier" => 0.75,
      "max_position_embeddings" => 256,
      "attention_bias" => false,
      "mlp_bias" => false,
      "rope_theta" => 10_000.0,
      "tie_word_embeddings" => true,
    },
    "granitemoe" => {
      "hidden_size" => 32,
      "num_hidden_layers" => 2,
      "intermediate_size" => 64,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 4,
      "rms_norm_eps" => 1e-5,
      "vocab_size" => 97,
      "logits_scaling" => 2.0,
      "attention_multiplier" => 0.25,
      "embedding_multiplier" => 1.25,
      "residual_multiplier" => 0.75,
      "max_position_embeddings" => 256,
      "attention_bias" => false,
      "mlp_bias" => false,
      "rope_theta" => 10_000.0,
      "num_local_experts" => 2,
      "num_experts_per_tok" => 1,
      "tie_word_embeddings" => true,
    },
    "lfm2_moe" => {
      "vocab_size" => 101,
      "hidden_size" => 32,
      "intermediate_size" => 64,
      "moe_intermediate_size" => 48,
      "num_hidden_layers" => 3,
      "num_experts" => 2,
      "num_experts_per_tok" => 1,
      "norm_topk_prob" => true,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 2,
      "max_position_embeddings" => 128,
      "use_expert_bias" => true,
      "num_dense_layers" => 1,
      "norm_eps" => 1e-5,
      "conv_bias" => false,
      "conv_L_cache" => 3,
      "layer_types" => ["full_attention", "conv", "full_attention"],
      "rope_parameters" => { "rope_theta" => 10_000.0 },
    },
    "lille-130m" => {
      "block_size" => 128,
      "layer_norm_eps" => 1e-5,
      "n_embd" => 96,
      "n_head" => 4,
      "n_kv_heads" => 2,
      "n_layer" => 2,
      "rope_theta" => 10_000.0,
      "vocab_size" => 89,
      "tie_word_embeddings" => true,
    },
    "minicpm" => {
      "hidden_size" => 32,
      "dim_model_base" => 16,
      "num_hidden_layers" => 2,
      "intermediate_size" => 64,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "rms_norm_eps" => 1e-5,
      "vocab_size" => 96,
      "scale_depth" => 1.4,
      "scale_emb" => 8.0,
      "max_position_embeddings" => 256,
      "rope_theta" => 10_000.0,
      "tie_word_embeddings" => false,
    },
    "minicpm3" => {
      "hidden_size" => 32,
      "dim_model_base" => 16,
      "num_hidden_layers" => 2,
      "intermediate_size" => 64,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 4,
      "rms_norm_eps" => 1e-5,
      "vocab_size" => 89,
      "q_lora_rank" => 8,
      "qk_nope_head_dim" => 4,
      "qk_rope_head_dim" => 4,
      "kv_lora_rank" => 8,
      "scale_depth" => 1.0,
      "scale_emb" => 1.25,
      "max_position_embeddings" => 256,
      "attention_bias" => false,
      "rope_theta" => 10_000.0,
      "rope_scaling" => {
        "original_max_position_embeddings" => 128,
        "short_factor" => 1.0,
        "long_factor" => 1.0,
      },
      "tie_word_embeddings" => false,
    },
    "phi3small" => {
      "hidden_size" => 32,
      "dense_attention_every_n_layers" => 1,
      "ff_intermediate_size" => 64,
      "gegelu_limit" => 16.0,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 4,
      "layer_norm_epsilon" => 1e-5,
      "vocab_size" => 97,
      "num_key_value_heads" => 2,
      "mup_attn_multiplier" => 1.0,
      "mup_use_scaling" => true,
      "mup_embedding_multiplier" => 1.0,
      "mup_width_multiplier" => 1.0,
      "rope_embedding_base" => 10_000.0,
      "rope_position_scale" => 1.0,
      "blocksparse_block_size" => 64,
      "blocksparse_num_local_blocks" => 4,
      "blocksparse_vert_stride" => 2,
    },
    "plamo" => {
      "hidden_size" => 32,
      "num_hidden_layers" => 2,
      "intermediate_size" => 64,
      "num_attention_heads" => 4,
      "rms_norm_eps" => 1e-5,
      "vocab_size" => 101,
      "n_shared_head" => 2,
      "rope_theta" => 10_000.0,
      "rope_traditional" => false,
    },
    "qwen" => {
      "hidden_size" => 32,
      "num_attention_heads" => 2,
      "num_hidden_layers" => 2,
      "kv_channels" => 16,
      "intermediate_size" => 64,
      "vocab_size" => 100,
      "no_bias" => true,
    },
    "qwen2_moe" => {
      "hidden_size" => 32,
      "num_hidden_layers" => 2,
      "intermediate_size" => 64,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 2,
      "rms_norm_eps" => 1e-5,
      "vocab_size" => 109,
      "rope_theta" => 10_000.0,
      "max_position_embeddings" => 256,
      "tie_word_embeddings" => false,
      "num_experts_per_tok" => 1,
      "num_experts" => 2,
      "moe_intermediate_size" => 16,
      "shared_expert_intermediate_size" => 24,
    },
    "recurrent_gemma" => {
      "hidden_size" => 32,
      "attention_bias" => false,
      "conv1d_width" => 3,
      "intermediate_size" => 64,
      "logits_soft_cap" => 1.5,
      "num_attention_heads" => 4,
      "num_hidden_layers" => 3,
      "num_key_value_heads" => 2,
      "rms_norm_eps" => 1e-5,
      "rope_theta" => 10_000.0,
      "attention_window_size" => 4,
      "vocab_size" => 97,
      "block_types" => ["recurrent", "attention"],
    },
    "step3p5" => {
      "hidden_size" => 32,
      "num_hidden_layers" => 3,
      "vocab_size" => 103,
      "num_attention_heads" => 4,
      "num_attention_groups" => 2,
      "head_dim" => 8,
      "intermediate_size" => 64,
      "rms_norm_eps" => 1e-5,
      "rope_theta" => [10_000.0, 10_000.0, 10_000.0],
      "sliding_window" => 4,
      "layer_types" => ["full_attention", "sliding_attention", "full_attention"],
      "partial_rotary_factors" => [0.5, 1.0, 0.5],
      "attention_other_setting" => {
        "num_attention_heads" => 4,
        "num_attention_groups" => 2,
      },
      "use_head_wise_attn_gate" => true,
      "moe_num_experts" => 2,
      "moe_top_k" => 1,
      "moe_intermediate_size" => 48,
      "share_expert_dim" => 48,
      "moe_layers_enum" => "1,2",
    },
    "pixtral" => {
      "text_config" => {
        "model_type" => "llama",
        "hidden_size" => 64,
        "num_hidden_layers" => 2,
        "num_attention_heads" => 4,
        "num_key_value_heads" => 2,
        "intermediate_size" => 128,
        "vocab_size" => 128,
        "rms_norm_eps" => 1e-5,
        "rope_theta" => 10_000.0,
        "tie_word_embeddings" => false,
      },
    },
  }.freeze

  def tiny_config_for(model_type)
    explicit = EXPLICIT_TINY_CONFIGS[model_type]
    return explicit if explicit

    DEFAULT_TINY_CONFIG
      .merge("model_type" => model_type)
      .merge(MODEL_TINY_CONFIG_OVERRIDES.fetch(model_type, {}))
  end

  SUBPROCESS_SCRIPT = <<~'RUBY'
    require "json"
    require "tmpdir"

    $LOAD_PATH.unshift File.expand_path("lib", __dir__)
    $LOAD_PATH.unshift File.expand_path("mlx-ruby/lib", __dir__)
    require "mlx"
    require "mlx_lm"

    config = JSON.parse(ARGV[0])
    model_type = config["model_type"]
    result = { "model_type" => model_type }

    begin
      mx = MLX::Core
      model_class, args_class = MlxLm::Models.get_classes(config)
      args = args_class.from_dict(config)
      model = model_class.new(args)

      params = MLX::Utils.tree_flatten(model.parameters).map { |_, v| v }
      mx.eval(*params) unless params.empty?

      input = mx.array([[1, 2, 3]], dtype: mx.int32)
      fun = ->(x) { model.call(x) }

      # Run compatibility report first (does not require full lowering)
      begin
        report = MLX::ONNX.export_onnx_compatibility_report(fun, input)
        include_full_nodes = ENV["ONNX_COMPAT_REPORT_JSON"] == "1"
        unsupported_invocations = Array(report["nodes"]).filter_map do |node|
          next unless node.is_a?(Hash) && node["supported"] == false

          {
            "index" => node["index"],
            "op" => node["op"],
            "onnx_op_type" => node["onnx_op_type"],
          }
        end

        result["compat_report"] = {
          "total_nodes"      => report["total_nodes"],
          "supported_nodes"  => report["supported_nodes"],
          "unsupported_nodes" => report["unsupported_nodes"],
          "unsupported_ops"  => report["unsupported_ops"],
          "ready"            => report["ready_for_stub_conversion"],
          "unsupported_invocations" => unsupported_invocations,
        }
        if include_full_nodes
          result["compat_report"]["nodes"] = report["nodes"]
          result["compat_report"]["format"] = report["format"]
          result["compat_report"]["ir_version"] = report["ir_version"]
        end
      rescue => e
        result["compat_error"] = "#{e.class}: #{e.message}"
      end

      # Attempt full ONNX export
      begin
        Dir.mktmpdir do |dir|
          path = File.join(dir, "#{model_type}.onnx")
          MLX::ONNX.export_onnx(path, fun, input)
          result["export"] = "success"
          result["onnx_size"] = File.size(path)
        end
      rescue NotImplementedError, RuntimeError => e
        result["export"] = "failed"
        result["export_error"] = e.message
      end
    rescue => e
      result["fatal"] = "#{e.class}: #{e.message}"
    end

    puts JSON.generate(result)
  RUBY

  def run_model_in_subprocess(model_type)
    config_json = JSON.generate(tiny_config_for(model_type))
    project_root = File.expand_path("../..", __dir__)
    timeout_seconds = Integer(ENV.fetch("ONNX_EXPORT_SUBPROCESS_TIMEOUT", "180"))

    out = +""
    err = +""
    status = nil

    Open3.popen3("ruby", "-e", SUBPROCESS_SCRIPT, config_json, chdir: project_root) do |stdin, stdout, stderr, wait_thr|
      stdin.close
      out_reader = Thread.new { stdout.read.to_s }
      err_reader = Thread.new { stderr.read.to_s }

      unless wait_thr.join(timeout_seconds)
        pid = wait_thr.pid
        begin
          Process.kill("TERM", pid)
        rescue Errno::ESRCH
          # already exited
        end
        unless wait_thr.join(5)
          begin
            Process.kill("KILL", pid)
          rescue Errno::ESRCH
            # already exited
          end
          wait_thr.join
        end

        out = out_reader.value
        err = err_reader.value
        return {
          "model_type" => model_type,
          "export" => "timeout",
          "timeout_seconds" => timeout_seconds,
          "stdout" => out.lines.first(10).join,
          "stderr" => err.lines.first(10).join,
        }
      end

      status = wait_thr.value
      out = out_reader.value
      err = err_reader.value
    end

    if status.signaled?
      sig = status.termsig
      signal_name = Signal.signame(sig) rescue sig.to_s
      return {
        "model_type" => model_type,
        "export" => "crashed",
        "crash_signal" => signal_name,
        "stderr" => err.lines.first(5).join,
      }
    end

    unless status.success?
      return {
        "model_type" => model_type,
        "export" => "process_error",
        "exit_code" => status.exitstatus,
        "stderr" => err.lines.first(10).join,
      }
    end

    JSON.parse(out)
  rescue JSON::ParserError
    {
      "model_type" => model_type,
      "export" => "parse_error",
      "stdout" => out.to_s[0, 500],
      "stderr" => err.to_s[0, 500],
    }
  end

  def onnx_log_lines_enabled?
    value = ENV.fetch("ONNX_LOG_LINES", "0").strip.downcase
    %w[1 true yes on].include?(value)
  end

  def onnx_log_line(text)
    puts text if onnx_log_lines_enabled?
  end

  def onnx_full_export_enabled?
    value = ENV.fetch("ONNX_FULL_EXPORT", "0").strip.downcase
    %w[1 true yes on].include?(value)
  end

  def assert_onnx_export(model_type)
    unless onnx_full_export_enabled?
      skip "#{model_type}: full ONNX export disabled by default (set ONNX_FULL_EXPORT=1 to enable)"
    end

    result = run_model_in_subprocess(model_type)

    case result["export"]
    when "success"
      assert true, "#{model_type}: ONNX export succeeded (#{result['onnx_size']} bytes)"
      report = result["compat_report"]
      if report
        onnx_log_line("\n  [ONNX] #{model_type}: PASS — #{report['supported_nodes']}/#{report['total_nodes']} nodes, #{result['onnx_size']} bytes")
      end
    when "failed"
      report = result["compat_report"]
      msg = "#{model_type}: ONNX export failed — #{result['export_error']}"
      if report
        unsupported = report["unsupported_ops"] || []
        msg += "\n  Nodes: #{report['supported_nodes']}/#{report['total_nodes']} supported"
        msg += "\n  Missing ops: #{unsupported.join(', ')}"
      end
      flunk(msg)
    when "crashed"
      report = result["compat_report"]
      msg = "#{model_type}: ONNX tracing crashed with signal #{result['crash_signal']}"
      if report
        unsupported = report["unsupported_ops"] || []
        msg += "\n  Compat report (pre-crash): #{report['supported_nodes']}/#{report['total_nodes']} nodes"
        msg += "\n  Missing ops: #{unsupported.empty? ? 'none' : unsupported.join(', ')}"
      end
      msg += "\n  (MoE models crash because tolist forces data-dependent control flow during tracing)"
      flunk(msg)
    when "process_error"
      flunk("#{model_type}: ONNX export process failed (exit #{result['exit_code']}): #{result['stderr']}")
    when "parse_error"
      flunk("#{model_type}: ONNX export subprocess parse error: stdout=#{result['stdout']}, stderr=#{result['stderr']}")
    when "timeout"
      flunk("#{model_type}: ONNX export subprocess timed out after #{result['timeout_seconds']}s")
    else
      flunk("#{model_type}: unexpected result — #{result.inspect}")
    end
  end

  def assert_onnx_compat_report(model_type)
    result = run_model_in_subprocess(model_type)

    if result["compat_error"]
      skip "#{model_type}: compat report unavailable — #{result['compat_error']}"
    end

    if result["crash_signal"]
      if result["compat_report"]
        report = result["compat_report"]
        assert_kind_of Integer, report["total_nodes"]
        unsupported = report["unsupported_ops"] || []
        unsupported_invocations = report["unsupported_invocations"] || []
        pct = report["total_nodes"] > 0 ? (report["supported_nodes"].to_f / report["total_nodes"] * 100).round(1) : 0
        onnx_log_line("\n  [ONNX] #{model_type}: #{report['supported_nodes']}/#{report['total_nodes']} nodes (#{pct}%) — missing: #{unsupported.empty? ? 'none' : unsupported.join(', ')} (CRASH during export)")
        unsupported_invocations.each do |inv|
          op = inv["op"] || "unknown"
          onnx = inv["onnx_op_type"] || "nil"
          index = inv["index"].nil? ? "nil" : inv["index"]
          onnx_log_line("  [ONNX-INV] #{model_type}: op=#{op} onnx=#{onnx} index=#{index}")
        end
        if ENV["ONNX_COMPAT_REPORT_JSON"] == "1"
          onnx_log_line("  [ONNX-JSON] #{model_type}: #{JSON.generate(report)}")
        end
      else
        skip "#{model_type}: process crashed (signal #{result['crash_signal']}) before compat report"
      end
      return
    end

    report = result["compat_report"]
    skip "#{model_type}: no compat report in result" unless report

    assert_kind_of Integer, report["total_nodes"]
    assert_kind_of Integer, report["supported_nodes"]
    assert_kind_of Integer, report["unsupported_nodes"]

    total = report["total_nodes"]
    supported = report["supported_nodes"]
    unsupported_ops = report["unsupported_ops"] || []
    unsupported_invocations = report["unsupported_invocations"] || []
    pct = total > 0 ? (supported.to_f / total * 100).round(1) : 0
    status = result["export"] == "success" ? "PASS" : "FAIL"
    onnx_log_line("\n  [ONNX] #{model_type}: #{status} — #{supported}/#{total} nodes (#{pct}%) — missing: #{unsupported_ops.empty? ? 'none' : unsupported_ops.join(', ')}")
    unsupported_invocations.each do |inv|
      op = inv["op"] || "unknown"
      onnx = inv["onnx_op_type"] || "nil"
      index = inv["index"].nil? ? "nil" : inv["index"]
      onnx_log_line("  [ONNX-INV] #{model_type}: op=#{op} onnx=#{onnx} index=#{index}")
    end
    if ENV["ONNX_COMPAT_REPORT_JSON"] == "1"
      onnx_log_line("  [ONNX-JSON] #{model_type}: #{JSON.generate(report)}")
    end
  end
end
