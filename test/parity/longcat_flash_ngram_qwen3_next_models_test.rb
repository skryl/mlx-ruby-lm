require_relative "../test_helper"
require_relative "../../lib/mlx_lm/models/longcat_flash_ngram"
require_relative "../../lib/mlx_lm/models/qwen3_next"

class Phase46DenseLaneASLongcatFlashNgramTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_longcat_flash_ngram_wrapper_from_dict_forward_shape_and_sanitize_mapping
    args = MlxLm::Models::LongcatFlashNgram::ModelArgs.from_dict({
      "model_type" => "longcat_flash_ngram",
      "vocab_size" => 67,
      "hidden_size" => 32,
      "ffn_hidden_size" => 64,
      "expert_ffn_hidden_size" => 16,
      "num_layers" => 2,
      "num_attention_heads" => 4,
      "n_routed_experts" => 2,
      "zero_expert_num" => 1,
      "moe_topk" => 1,
      "kv_lora_rank" => 8,
      "q_lora_rank" => 8,
      "qk_rope_head_dim" => 8,
      "qk_nope_head_dim" => 8,
      "v_head_dim" => 8,
      "routed_scaling_factor" => 1.0,
      "max_position_embeddings" => 128,
      "rms_norm_eps" => 1e-5,
      "rope_theta" => 10_000.0,
      "attention_bias" => false,
      "norm_topk_prob" => true,
      "n_group" => 1,
      "topk_group" => 1,
      "first_k_dense_replace" => 0,
      "tie_word_embeddings" => false,
    })

    assert_equal 2, args.num_hidden_layers
    assert_equal 1, args.num_experts_per_tok
    assert_equal 16, args.moe_intermediate_size

    model = MlxLm::Models::LongcatFlashNgram::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 3, 67], output.shape
    assert_equal 2, model.layers.length
    assert_nil model.make_cache

    embed_q_weight = @mx.array([[1.0, 2.0], [3.0, 4.0]], dtype: @mx.float32)
    weights = {
      "model.ngram_embeddings.word_embeddings.weight" => @mx.zeros([67, 32]).astype(@mx.float32),
      "model.layers.0.attention.embed_q.weight" => embed_q_weight,
      "model.layers.0.mlp.experts.0.up_proj.weight" => @mx.array([[1.0, 2.0], [3.0, 4.0]], dtype: @mx.float32),
      "model.layers.0.mlp.experts.1.up_proj.weight" => @mx.array([[5.0, 6.0], [7.0, 8.0]], dtype: @mx.float32),
      "model.layers.0.mlp.experts.2.up_proj.weight" => @mx.array([[9.0, 10.0], [11.0, 12.0]], dtype: @mx.float32),
      "model.layers.2.mlp.up_proj.weight" => @mx.zeros([2, 2]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)
    mapped_q_proj = sanitized["model.layers.0.self_attn.q_proj.weight"]
    stacked = sanitized["model.layers.0.mlp.switch_mlp.up_proj.weight"]
    mapped_ngram_embed = sanitized["model.ngram_embeddings.word_embeddings.weight"]
    @mx.eval(mapped_q_proj, stacked, mapped_ngram_embed)

    refute sanitized.key?("model.layers.0.attention.embed_q.weight")
    refute sanitized.key?("model.layers.0.mlp.experts.0.up_proj.weight")
    refute sanitized.key?("model.layers.0.mlp.experts.1.up_proj.weight")
    refute sanitized.key?("model.layers.0.mlp.experts.2.up_proj.weight")
    refute sanitized.key?("model.layers.2.mlp.up_proj.weight")
    refute sanitized.key?("model.embed_tokens.weight")
    assert sanitized.key?("model.layers.0.self_attn.q_proj.weight")
    assert sanitized.key?("model.layers.0.mlp.switch_mlp.up_proj.weight")
    assert sanitized.key?("model.ngram_embeddings.word_embeddings.weight")
    assert_equal embed_q_weight.to_a, mapped_q_proj.to_a
    assert_equal [3, 2, 2], stacked.shape
  end
end

class Phase46DenseLaneASQwen3NextTest < Minitest::Test
  def setup
    @mx = MLX::Core
  end

  def test_qwen3_next_wrapper_from_dict_forward_shape_and_sanitize_mapping
    args = MlxLm::Models::Qwen3Next::ModelArgs.from_dict({
      "model_type" => "qwen3_next",
      "vocab_size" => 71,
      "hidden_size" => 32,
      "intermediate_size" => 64,
      "moe_intermediate_size" => 24,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 2,
      "num_experts" => 2,
      "num_experts_per_tok" => 1,
      "shared_expert_intermediate_size" => 20,
      "decoder_sparse_step" => 1,
      "mlp_only_layers" => [0],
      "linear_num_value_heads" => 2,
      "linear_num_key_heads" => 2,
      "linear_key_head_dim" => 8,
      "linear_value_head_dim" => 8,
      "linear_conv_kernel_dim" => 3,
      "max_position_embeddings" => 128,
      "rms_norm_eps" => 1e-5,
      "rope_theta" => 10_000.0,
      "partial_rotary_factor" => 0.5,
      "attention_bias" => false,
      "tie_word_embeddings" => false,
    })

    assert_equal 1, args.first_k_dense_replace
    assert_equal 1, args.num_shared_experts
    assert_equal 20, args.moe_shared_expert_intermediate_size

    model = MlxLm::Models::Qwen3Next::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, value| value })

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)
    assert_equal [1, 3, 71], output.shape
    assert_equal 2, model.layers.length
    assert_nil model.make_cache

    gate_weight = @mx.array([[1.0, 2.0], [3.0, 4.0]], dtype: @mx.float32)
    weights = {
      "model.layers.1.mlp.experts.0.gate_proj.weight" => @mx.array([[1.0, 2.0], [3.0, 4.0]], dtype: @mx.float32),
      "model.layers.1.mlp.experts.1.gate_proj.weight" => @mx.array([[5.0, 6.0], [7.0, 8.0]], dtype: @mx.float32),
      "model.layers.1.mlp.router.weight" => gate_weight,
      "model.layers.1.mlp.router.bias" => @mx.array([0.1, -0.2], dtype: @mx.float32),
      "model.layers.1.mlp.shared_expert.gate_proj.weight" => @mx.zeros([20, 32]).astype(@mx.float32),
      "model.embed_tokens.weight" => @mx.zeros([71, 32]).astype(@mx.float32),
      "model.mtp.layers.0.weight" => @mx.zeros([2, 2]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)
    stacked = sanitized["model.layers.1.mlp.switch_mlp.gate_proj.weight"]
    remapped_gate = sanitized["model.layers.1.mlp.gate.gate_proj.weight"]
    remapped_bias = sanitized["model.layers.1.mlp.gate.gate_proj.bias"]
    remapped_shared = sanitized["model.layers.1.mlp.shared_experts.gate_proj.weight"]
    remapped_embed = sanitized["model.word_embeddings.weight"]
    @mx.eval(stacked, remapped_gate, remapped_bias, remapped_shared, remapped_embed)

    refute sanitized.key?("model.layers.1.mlp.experts.0.gate_proj.weight")
    refute sanitized.key?("model.layers.1.mlp.experts.1.gate_proj.weight")
    refute sanitized.key?("model.layers.1.mlp.router.weight")
    refute sanitized.key?("model.layers.1.mlp.router.bias")
    refute sanitized.key?("model.layers.1.mlp.shared_expert.gate_proj.weight")
    refute sanitized.key?("model.embed_tokens.weight")
    refute sanitized.key?("model.mtp.layers.0.weight")
    assert sanitized.key?("model.layers.1.mlp.switch_mlp.gate_proj.weight")
    assert sanitized.key?("model.layers.1.mlp.gate.gate_proj.weight")
    assert sanitized.key?("model.layers.1.mlp.gate.gate_proj.bias")
    assert sanitized.key?("model.layers.1.mlp.shared_experts.gate_proj.weight")
    assert sanitized.key?("model.word_embeddings.weight")
    assert_equal [2, 2, 2], stacked.shape
    assert_equal gate_weight.to_a, remapped_gate.to_a
  end
end

class Phase46DenseLaneASRegistryTest < Minitest::Test
  def test_models_registered_and_resolve
    assert MlxLm::Models::REGISTRY.key?("longcat_flash_ngram"), "longcat_flash_ngram should be registered"
    assert MlxLm::Models::REGISTRY.key?("qwen3_next"), "qwen3_next should be registered"

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "longcat_flash_ngram" })
    assert_equal MlxLm::Models::LongcatFlashNgram::Model, model_class
    assert_equal MlxLm::Models::LongcatFlashNgram::ModelArgs, args_class

    model_class, args_class = MlxLm::Models.get_classes({ "model_type" => "qwen3_next" })
    assert_equal MlxLm::Models::Qwen3Next::Model, model_class
    assert_equal MlxLm::Models::Qwen3Next::ModelArgs, args_class
  end
end
