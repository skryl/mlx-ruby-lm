require_relative "../test_helper"

# Phase 11: Model Architectures (Batch 3)
# Tests OLMo2, GPTNeoX, Mixtral, DeepSeek, and InternLM2 model architectures.

class Phase11OLMo2Test < Minitest::Test
  include ParityTestHelpers

  def setup
    @mx = MLX::Core
  end

  # Test 1: OLMo2 model instantiation and forward pass
  def test_olmo2_forward
    args = MlxLm::Models::OLMo2::ModelArgs.from_dict({
      "model_type" => "olmo2",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 128,
      "vocab_size" => 128,
      "rms_norm_eps" => 1e-6,
      "tie_word_embeddings" => true,
    })
    model = MlxLm::Models::OLMo2::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    input = @mx.array([[1, 2, 3]]).astype(@mx.int32)
    output = model.call(input)
    @mx.eval(output)
    assert_equal [1, 3, 128], output.shape
  end

  # Test 2: OLMo2 Q/K normalization layers exist
  def test_olmo2_qk_norm
    args = MlxLm::Models::OLMo2::ModelArgs.from_dict({
      "model_type" => "olmo2",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 128,
      "vocab_size" => 128,
    })
    model = MlxLm::Models::OLMo2::Model.new(args)

    layer = model.layers[0]
    attn = layer.self_attn
    assert_respond_to attn, :q_norm
    assert_respond_to attn, :k_norm
  end

  # Test 3: OLMo2 registered in registry
  def test_olmo2_registered
    assert MlxLm::Models::REGISTRY.key?("olmo2"), "OLMo2 should be registered"
  end
end

class Phase11GPTNeoXTest < Minitest::Test
  include ParityTestHelpers

  def setup
    @mx = MLX::Core
  end

  # Test 4: GPTNeoX forward pass
  def test_gpt_neox_forward
    args = MlxLm::Models::GPTNeoX::ModelArgs.from_dict({
      "model_type" => "gpt_neox",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "vocab_size" => 128,
      "layer_norm_eps" => 1e-5,
      "rotary_pct" => 0.25,
      "rotary_emb_base" => 10000,
      "use_parallel_residual" => true,
    })
    model = MlxLm::Models::GPTNeoX::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    input = @mx.array([[1, 2, 3]]).astype(@mx.int32)
    output = model.call(input)
    @mx.eval(output)
    assert_equal [1, 3, 128], output.shape
  end

  # Test 5: GPTNeoX parallel residual path
  def test_gpt_neox_parallel_residual
    args = MlxLm::Models::GPTNeoX::ModelArgs.from_dict({
      "model_type" => "gpt_neox",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "vocab_size" => 128,
      "layer_norm_eps" => 1e-5,
      "rotary_pct" => 0.25,
      "use_parallel_residual" => false,
    })
    model = MlxLm::Models::GPTNeoX::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    input = @mx.array([[1, 2, 3]]).astype(@mx.int32)
    output = model.call(input)
    @mx.eval(output)
    assert_equal [1, 3, 128], output.shape
  end

  # Test 6: GPTNeoX registered
  def test_gpt_neox_registered
    assert MlxLm::Models::REGISTRY.key?("gpt_neox"), "GPTNeoX should be registered"
  end
end

class Phase11MixtralTest < Minitest::Test
  include ParityTestHelpers

  def setup
    @mx = MLX::Core
  end

  # Test 7: Mixtral forward pass with MoE
  def test_mixtral_forward
    args = MlxLm::Models::Mixtral::ModelArgs.from_dict({
      "model_type" => "mixtral",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 128,
      "vocab_size" => 128,
      "num_local_experts" => 4,
      "num_experts_per_tok" => 2,
      "rms_norm_eps" => 1e-5,
    })
    model = MlxLm::Models::Mixtral::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    input = @mx.array([[1, 2, 3]]).astype(@mx.int32)
    output = model.call(input)
    @mx.eval(output)
    assert_equal [1, 3, 128], output.shape
  end

  # Test 8: Mixtral MoE layer has gate
  def test_mixtral_moe_gate
    args = MlxLm::Models::Mixtral::ModelArgs.from_dict({
      "model_type" => "mixtral",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 128,
      "vocab_size" => 128,
      "num_local_experts" => 4,
      "num_experts_per_tok" => 2,
    })
    model = MlxLm::Models::Mixtral::Model.new(args)
    layer = model.layers[0]
    assert_respond_to layer, :block_sparse_moe
    moe = layer.block_sparse_moe
    assert_respond_to moe, :gate
  end

  # Test 9: Mixtral registered
  def test_mixtral_registered
    assert MlxLm::Models::REGISTRY.key?("mixtral"), "Mixtral should be registered"
  end
end

class Phase11DeepSeekTest < Minitest::Test
  include ParityTestHelpers

  def setup
    @mx = MLX::Core
  end

  # Test 10: DeepSeek forward pass (dense-only)
  def test_deepseek_forward_dense
    args = MlxLm::Models::DeepSeek::ModelArgs.from_dict({
      "model_type" => "deepseek",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 128,
      "vocab_size" => 128,
      "rms_norm_eps" => 1e-6,
    })
    model = MlxLm::Models::DeepSeek::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    input = @mx.array([[1, 2, 3]]).astype(@mx.int32)
    output = model.call(input)
    @mx.eval(output)
    assert_equal [1, 3, 128], output.shape
  end

  # Test 11: DeepSeek forward pass with MoE layers
  def test_deepseek_forward_moe
    args = MlxLm::Models::DeepSeek::ModelArgs.from_dict({
      "model_type" => "deepseek",
      "hidden_size" => 64,
      "num_hidden_layers" => 4,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 128,
      "moe_intermediate_size" => 64,
      "vocab_size" => 128,
      "n_routed_experts" => 4,
      "num_experts_per_tok" => 2,
      "n_shared_experts" => 1,
      "moe_layer_freq" => 2,
      "first_k_dense_replace" => 1,
    })
    model = MlxLm::Models::DeepSeek::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    input = @mx.array([[1, 2, 3]]).astype(@mx.int32)
    output = model.call(input)
    @mx.eval(output)
    assert_equal [1, 3, 128], output.shape
  end

  # Test 12: DeepSeek registered
  def test_deepseek_registered
    assert MlxLm::Models::REGISTRY.key?("deepseek"), "DeepSeek should be registered"
  end
end

class Phase11InternLM2Test < Minitest::Test
  include ParityTestHelpers

  def setup
    @mx = MLX::Core
  end

  # Test 13: InternLM2 forward pass
  def test_internlm2_forward
    args = MlxLm::Models::InternLM2::ModelArgs.from_dict({
      "model_type" => "internlm2",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 128,
      "vocab_size" => 128,
      "rms_norm_eps" => 1e-6,
      "tie_word_embeddings" => false,
    })
    model = MlxLm::Models::InternLM2::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    input = @mx.array([[1, 2, 3]]).astype(@mx.int32)
    output = model.call(input)
    @mx.eval(output)
    assert_equal [1, 3, 128], output.shape
  end

  # Test 14: InternLM2 combined QKV projection
  def test_internlm2_combined_qkv
    args = MlxLm::Models::InternLM2::ModelArgs.from_dict({
      "model_type" => "internlm2",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 128,
      "vocab_size" => 128,
    })
    model = MlxLm::Models::InternLM2::Model.new(args)
    layer = model.layers[0]
    attn = layer.attention
    assert_respond_to attn, :wqkv
  end

  # Test 15: InternLM2 registered
  def test_internlm2_registered
    assert MlxLm::Models::REGISTRY.key?("internlm2"), "InternLM2 should be registered"
  end
end
