require_relative "../test_helper"

# Phase 7: Quantization Engine
# Tests quantization, dequantization, and quantization-aware loading.
class Phase7QuantizeTest < Minitest::Test
  include ParityTestHelpers

  def setup
    @mx = MLX::Core
  end

  def make_tiny_llama(hidden: 64, layers: 2, vocab: 128)
    args = MlxLm::Models::Llama::ModelArgs.from_dict({
      "model_type" => "llama",
      "hidden_size" => hidden,
      "num_hidden_layers" => layers,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 128,
      "vocab_size" => vocab,
      "rms_norm_eps" => 1e-6,
      "tie_word_embeddings" => true,
    })
    MlxLm::Models::Llama::Model.new(args)
  end

  # Test 1: quantize_model converts Linear layers to QuantizedLinear
  def test_quantize_model_converts_layers
    model = make_tiny_llama
    MlxLm::Quantize.quantize_model(model, group_size: 64, bits: 4)

    # Check that at least some layers are now quantized
    has_quantized = false
    model.named_modules.each do |name, mod|
      if mod.is_a?(MLX::NN::QuantizedLinear)
        has_quantized = true
        break
      end
    end
    assert has_quantized, "Model should have QuantizedLinear layers after quantization"
  end

  # Test 2: Quantized model still produces correct output shape
  def test_quantized_model_forward
    model = make_tiny_llama
    @mx.eval(*model.parameters.values)
    MlxLm::Quantize.quantize_model(model, group_size: 32, bits: 4)

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    assert_equal [1, 3, 128], output.shape, "Quantized model should produce same shape output"
  end

  # Test 3: quantize_model returns config dict
  def test_quantize_model_returns_config
    model = make_tiny_llama
    config = MlxLm::Quantize.quantize_model(model, group_size: 64, bits: 4)
    assert_equal 64, config["group_size"]
    assert_equal 4, config["bits"]
  end

  # Test 4: quantize_model with weights parameter only quantizes layers with scales
  def test_quantize_model_with_weights_predicate
    model = make_tiny_llama

    # Create fake weights dict with only one layer having scales
    weights = {
      "model.layers.0.self_attn.q_proj.weight" => @mx.zeros([64, 64]).astype(@mx.float32),
      "model.layers.0.self_attn.q_proj.scales" => @mx.zeros([64, 1]).astype(@mx.float32),
      "model.layers.0.self_attn.k_proj.weight" => @mx.zeros([64, 64]).astype(@mx.float32),
      # No scales for k_proj - it shouldn't be quantized
    }

    MlxLm::Quantize.quantize_model(model, group_size: 64, bits: 4, weights: weights)

    # The q_proj should be quantized (has scales), k_proj should not
    layer0 = model.layers[0]
    attn = layer0.self_attn
    q_proj = attn.q_proj
    k_proj = attn.k_proj

    assert_instance_of MLX::NN::QuantizedLinear, q_proj, "q_proj should be quantized (has scales)"
    assert_instance_of MLX::NN::Linear, k_proj, "k_proj should NOT be quantized (no scales)"
  end

  # Test 5: MLX Ruby QuantizedLinear exists and works
  def test_quantized_linear_from_linear
    linear = MLX::NN::Linear.new(32, 64, bias: false)
    @mx.eval(*linear.parameters.values)
    qlinear = linear.to_quantized(group_size: 32, bits: 4)
    assert_instance_of MLX::NN::QuantizedLinear, qlinear

    # Forward pass should work
    x = @mx.ones([1, 32]).astype(@mx.float32)
    output = qlinear.call(x)
    assert_equal [1, 64], output.shape
  end

  # Test 6: QuantizedEmbedding exists and works
  def test_quantized_embedding
    embed = MLX::NN::Embedding.new(100, 32)
    @mx.eval(*embed.parameters.values)
    qembed = embed.to_quantized(group_size: 32, bits: 4)
    assert_instance_of MLX::NN::QuantizedEmbedding, qembed

    ids = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = qembed.call(ids)
    assert_equal [1, 3, 32], output.shape
  end

  # Test 7: nn.quantize function works
  def test_nn_quantize
    model = make_tiny_llama
    @mx.eval(*model.parameters.values)

    MLX::NN.quantize(model, group_size: 32, bits: 4)

    # After quantization, Linear layers should be QuantizedLinear
    has_quantized = false
    model.named_modules.each do |name, mod|
      if mod.is_a?(MLX::NN::QuantizedLinear)
        has_quantized = true
        break
      end
    end
    assert has_quantized, "nn.quantize should convert Linear to QuantizedLinear"
  end

  # Test 8: Quantized model with KV cache works
  def test_quantized_model_with_cache
    model = make_tiny_llama
    @mx.eval(*model.parameters.values)
    MlxLm::Quantize.quantize_model(model, group_size: 32, bits: 4)

    cache = Array.new(2) { MlxLm::KVCache.new }

    # First token
    token1 = @mx.array([[1]], dtype: @mx.int32)
    out1 = model.call(token1, cache: cache)
    assert_equal [1, 1, 128], out1.shape

    # Second token (using cache)
    token2 = @mx.array([[2]], dtype: @mx.int32)
    out2 = model.call(token2, cache: cache)
    assert_equal [1, 1, 128], out2.shape
  end
end
