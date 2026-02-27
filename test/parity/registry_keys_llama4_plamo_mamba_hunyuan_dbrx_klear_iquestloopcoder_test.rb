# frozen_string_literal: true

require_relative "../test_helper"

class RegistryKeysLlama4PlamoMambaHunyuanDbrxKlearIquestloopcoderTest < Minitest::Test
  MODEL_TYPES = %w[
    llama4_text
    plamo
    mamba
    mamba2
    hunyuan_v1_dense
    dbrx
    Klear
    iquestloopcoder
  ].freeze

  def test_phase23_model_keys_resolve_with_tiny_configs
    MODEL_TYPES.each do |model_type|
      assert MlxLm::Models::REGISTRY.key?(model_type), "#{model_type} should be registered"

      model_class, args_class = MlxLm::Models.get_classes(tiny_config(model_type))

      assert_kind_of Class, model_class, "#{model_type} should resolve to a model class"
      assert_kind_of Class, args_class, "#{model_type} should resolve to a model args class"
      assert_instance_of args_class, args_class.from_dict(tiny_config(model_type))
    end
  end

  private

  def tiny_config(model_type)
    case model_type
    when "llama4_text"
      {
        "model_type" => "llama4_text",
        "hidden_size" => 16,
        "num_attention_heads" => 2,
        "num_key_value_heads" => 1,
        "num_hidden_layers" => 2,
        "vocab_size" => 64,
        "intermediate_size" => 32,
        "intermediate_size_mlp" => 32,
        "head_dim" => 8,
        "no_rope_layers" => [0, 1],
      }
    when "plamo"
      {
        "model_type" => "plamo",
        "hidden_size" => 16,
        "num_hidden_layers" => 1,
        "intermediate_size" => 32,
        "num_attention_heads" => 2,
        "rms_norm_eps" => 1e-5,
        "vocab_size" => 64,
        "n_shared_head" => 1,
      }
    when "mamba"
      {
        "model_type" => "mamba",
        "vocab_size" => 64,
        "hidden_size" => 16,
        "intermediate_size" => 16,
        "state_size" => 8,
        "num_hidden_layers" => 1,
        "conv_kernel" => 3,
        "use_bias" => true,
        "use_conv_bias" => true,
        "time_step_rank" => "auto",
      }
    when "mamba2"
      {
        "model_type" => "mamba2",
        "num_heads" => 2,
        "head_dim" => 8,
        "vocab_size" => 64,
        "hidden_size" => 16,
        "state_size" => 8,
        "num_hidden_layers" => 1,
        "conv_kernel" => 3,
        "n_groups" => 1,
        "time_step_rank" => "auto",
        "time_step_limit" => [0.001, 10.0],
      }
    when "hunyuan_v1_dense"
      {
        "model_type" => "hunyuan_v1_dense",
        "vocab_size" => 64,
        "hidden_size" => 16,
        "num_hidden_layers" => 1,
        "intermediate_size" => 32,
        "num_attention_heads" => 2,
        "num_key_value_heads" => 1,
      }
    when "dbrx"
      {
        "model_type" => "dbrx",
        "vocab_size" => 64,
        "d_model" => 16,
        "n_layers" => 1,
        "n_heads" => 2,
        "attn_config" => {
          "kv_n_heads" => 1,
          "clip_qkv" => 8.0,
          "rope_theta" => 10_000.0,
        },
        "ffn_config" => {
          "ffn_hidden_size" => 32,
          "moe_num_experts" => 2,
          "moe_top_k" => 1,
        },
      }
    when "Klear"
      {
        "model_type" => "Klear",
        "hidden_size" => 16,
        "num_hidden_layers" => 1,
        "intermediate_size" => 32,
        "num_attention_heads" => 2,
        "num_key_value_heads" => 2,
        "attention_bias" => false,
        "mlp_only_layers" => [],
        "num_experts" => 2,
        "num_experts_per_tok" => 1,
        "decoder_sparse_step" => 1,
        "n_shared_experts" => 1,
        "moe_intermediate_size" => 24,
        "rms_norm_eps" => 1e-5,
        "vocab_size" => 64,
        "rope_theta" => 10_000.0,
        "max_position_embeddings" => 128,
        "norm_topk_prob" => false,
      }
    else
      {
        "model_type" => "iquestloopcoder",
        "hidden_size" => 16,
        "num_hidden_layers" => 1,
        "intermediate_size" => 32,
        "num_attention_heads" => 2,
        "num_key_value_heads" => 1,
        "rms_norm_eps" => 1e-5,
        "vocab_size" => 64,
        "head_dim" => 8,
        "loop_num" => 2,
        "loop_window_size" => 4,
      }
    end
  end
end
