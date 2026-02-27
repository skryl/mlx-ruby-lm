require_relative "../test_helper"

# Tests that Gemma, Qwen2, Phi3, and Starcoder2 can be instantiated,
# produce correct output shapes, and are registered in the model registry.
class RegistryGemmaQwen2Phi3Starcoder2Test < Minitest::Test
  include ParityTestHelpers

  # Test 1: Core models are registered
  def test_all_models_registered
    %w[gemma qwen2 phi3 starcoder2].each do |name|
      assert MlxLm::Models::REGISTRY.key?(name), "#{name} should be registered"
    end
  end

  # Test 2: Remapping for mistral -> llama still works
  def test_mistral_remap
    config = { "model_type" => "mistral" }
    model_class, _ = MlxLm::Models.get_classes(config)
    assert_equal MlxLm::Models::Llama::Model, model_class
  end

  # Test 3: get_classes resolves each new architecture
  def test_get_classes_resolves_all
    {
      "gemma" => MlxLm::Models::Gemma::Model,
      "qwen2" => MlxLm::Models::Qwen2::Model,
      "phi3" => MlxLm::Models::Phi3::Model,
      "starcoder2" => MlxLm::Models::Starcoder2::Model,
    }.each do |model_type, expected_class|
      config = { "model_type" => model_type }
      model_class, _ = MlxLm::Models.get_classes(config)
      assert_equal expected_class, model_class, "#{model_type} should resolve to #{expected_class}"
    end
  end
end

class RegistryGemmaQwen2Phi3Starcoder2GemmaInstantiatesTest < Minitest::Test
  include ParityTestHelpers

  def setup
    @mx = MLX::Core
    @args = MlxLm::Models::Gemma::ModelArgs.from_dict({
      "model_type" => "gemma",
      "hidden_size" => 32,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 64,
      "vocab_size" => 100,
      "rms_norm_eps" => 1e-6,
      "head_dim" => 16,
    })
  end

  # Test 4: Gemma model instantiates
  def test_gemma_instantiates
    model = MlxLm::Models::Gemma::Model.new(@args)
    assert_instance_of MlxLm::Models::Gemma::Model, model
    assert_equal 2, model.layers.length
  end

  # Test 5: Gemma forward pass produces correct output shape
  def test_gemma_forward_shape
    model = MlxLm::Models::Gemma::Model.new(@args)
    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    assert_equal [1, 3, 100], output.shape, "Output should be [batch, seq_len, vocab_size]"
  end

  # Test 6: Gemma forward with KV cache
  def test_gemma_forward_with_cache
    model = MlxLm::Models::Gemma::Model.new(@args)
    cache = Array.new(2) { MlxLm::KVCache.new }
    token = @mx.array([[5]], dtype: @mx.int32)
    output = model.call(token, cache: cache)
    assert_equal [1, 1, 100], output.shape
  end
end

class RegistryGemmaQwen2Phi3Starcoder2Qwen2Test < Minitest::Test
  include ParityTestHelpers

  def setup
    @mx = MLX::Core
    @args = MlxLm::Models::Qwen2::ModelArgs.from_dict({
      "model_type" => "qwen2",
      "hidden_size" => 32,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 64,
      "vocab_size" => 100,
      "rms_norm_eps" => 1e-6,
      "tie_word_embeddings" => true,
    })
  end

  # Test 7: Qwen2 model instantiates
  def test_qwen2_instantiates
    model = MlxLm::Models::Qwen2::Model.new(@args)
    assert_instance_of MlxLm::Models::Qwen2::Model, model
  end

  # Test 8: Qwen2 forward pass produces correct output shape
  def test_qwen2_forward_shape
    model = MlxLm::Models::Qwen2::Model.new(@args)
    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    assert_equal [1, 3, 100], output.shape
  end

  # Test 9: Qwen2 sanitize removes rotary_emb and tied lm_head
  def test_qwen2_sanitize
    model = MlxLm::Models::Qwen2::Model.new(@args)
    weights = {
      "model.embed_tokens.weight" => @mx.zeros([100, 32]).astype(@mx.float32),
      "model.layers.0.self_attn.rotary_emb.inv_freq" => @mx.zeros([16]).astype(@mx.float32),
      "lm_head.weight" => @mx.zeros([100, 32]).astype(@mx.float32),
    }
    sanitized = model.sanitize(weights)
    refute sanitized.key?("model.layers.0.self_attn.rotary_emb.inv_freq")
    refute sanitized.key?("lm_head.weight")
    assert sanitized.key?("model.embed_tokens.weight")
  end
end

class RegistryGemmaQwen2Phi3Starcoder2Phi3Test < Minitest::Test
  include ParityTestHelpers

  def setup
    @mx = MLX::Core
    @args = MlxLm::Models::Phi3::ModelArgs.from_dict({
      "model_type" => "phi3",
      "hidden_size" => 32,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 64,
      "vocab_size" => 100,
      "rms_norm_eps" => 1e-5,
      "tie_word_embeddings" => false,
    })
  end

  # Test 10: Phi3 model instantiates with combined QKV projection
  def test_phi3_instantiates
    model = MlxLm::Models::Phi3::Model.new(@args)
    assert_instance_of MlxLm::Models::Phi3::Model, model
  end

  # Test 11: Phi3 forward pass
  def test_phi3_forward_shape
    model = MlxLm::Models::Phi3::Model.new(@args)
    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    assert_equal [1, 3, 100], output.shape
  end
end

class RegistryGemmaQwen2Phi3Starcoder2Starcoder2Test < Minitest::Test
  include ParityTestHelpers

  def setup
    @mx = MLX::Core
    @args = MlxLm::Models::Starcoder2::ModelArgs.from_dict({
      "model_type" => "starcoder2",
      "hidden_size" => 32,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 64,
      "vocab_size" => 100,
      "norm_epsilon" => 1e-5,
      "tie_word_embeddings" => true,
    })
  end

  # Test 12: Starcoder2 model instantiates (uses LayerNorm)
  def test_starcoder2_instantiates
    model = MlxLm::Models::Starcoder2::Model.new(@args)
    assert_instance_of MlxLm::Models::Starcoder2::Model, model
  end

  # Test 13: Starcoder2 forward pass
  def test_starcoder2_forward_shape
    model = MlxLm::Models::Starcoder2::Model.new(@args)
    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    assert_equal [1, 3, 100], output.shape
  end

  # Test 14: Starcoder2 uses LayerNorm (not RMSNorm)
  def test_starcoder2_uses_layernorm
    model = MlxLm::Models::Starcoder2::Model.new(@args)
    block = model.layers[0]
    # Access the normalization layers
    norm = block.input_layernorm
    assert_instance_of MLX::NN::LayerNorm, norm, "Starcoder2 should use LayerNorm"
  end
end
