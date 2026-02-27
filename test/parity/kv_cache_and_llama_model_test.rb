require_relative "../test_helper"

class KvCacheAndLlamaModelTest < Minitest::Test
  include ParityTestHelpers

  def mx
    MLX::Core
  end

  # Test 3.4: KVCache state after N forward passes has identical shape/values
  def test_kv_cache_update_and_fetch
    cache = MlxLm::KVCache.new

    # Simulate first token batch (prefill with 4 tokens)
    k1 = mx.ones([1, 2, 4, 8])  # B=1, n_kv_heads=2, S=4, head_dim=8
    v1 = mx.ones([1, 2, 4, 8]) * 2.0

    keys, values = cache.update_and_fetch(k1, v1)
    mx.eval(keys, values)

    assert_equal [1, 2, 4, 8], keys.shape
    assert_equal 4, cache.offset

    # Simulate second token (generation step)
    k2 = mx.ones([1, 2, 1, 8]) * 3.0
    v2 = mx.ones([1, 2, 1, 8]) * 4.0

    keys, values = cache.update_and_fetch(k2, v2)
    mx.eval(keys, values)

    assert_equal [1, 2, 5, 8], keys.shape
    assert_equal 5, cache.offset
  end

  # Test 3.5: RotatingKVCache wraps correctly at boundary
  def test_rotating_kv_cache_basic
    cache = MlxLm::RotatingKVCache.new(max_size: 8, keep: 2)

    # Prefill with 4 tokens
    k1 = mx.ones([1, 2, 4, 8])
    v1 = mx.ones([1, 2, 4, 8])

    keys, values = cache.update_and_fetch(k1, v1)
    mx.eval(keys, values)

    assert_equal [1, 2, 4, 8], keys.shape
    assert_equal 4, cache.offset
  end

  def test_rotating_kv_cache_wraps
    cache = MlxLm::RotatingKVCache.new(max_size: 6, keep: 0)

    # Prefill with 4 tokens
    k1 = mx.ones([1, 1, 4, 4])
    v1 = mx.ones([1, 1, 4, 4])
    cache.update_and_fetch(k1, v1)

    # Add tokens one at a time until we exceed max_size
    6.times do
      k = mx.ones([1, 1, 1, 4])
      v = mx.ones([1, 1, 1, 4])
      keys, values = cache.update_and_fetch(k, v)
      mx.eval(keys, values)
    end

    # After 4 + 6 = 10 tokens, cache should be capped at max_size=6
    assert_equal 10, cache.offset
    assert_equal 6, cache.size
  end

  def test_make_prompt_cache
    # Create a minimal mock model with layers
    model = MlxLm::Models::Llama::Model.new(
      MlxLm::Models::Llama::ModelArgs.from_dict(
        "hidden_size" => 32,
        "num_hidden_layers" => 2,
        "num_attention_heads" => 2,
        "num_key_value_heads" => 2,
        "intermediate_size" => 64,
        "vocab_size" => 100,
      )
    )

    cache = MlxLm::Cache.make_prompt_cache(model)
    assert_equal 2, cache.length
    assert_instance_of MlxLm::KVCache, cache[0]
  end
end

class KvCacheAndLlamaModelForwardPassShapesTest < Minitest::Test
  include ParityTestHelpers

  def mx
    MLX::Core
  end

  def make_small_model
    args = MlxLm::Models::Llama::ModelArgs.from_dict(
      "hidden_size" => 32,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 2,
      "intermediate_size" => 64,
      "vocab_size" => 100,
      "rms_norm_eps" => 1e-5,
      "rope_theta" => 10000.0,
    )
    MlxLm::Models::Llama::Model.new(args)
  end

  # Test 3.1: Llama forward pass (random weights, fixed input) → logits shape correct
  def test_forward_pass_shapes
    mx_mod = MLX::Core
    mx_mod.random_seed(42)

    model = make_small_model
    mx_mod.eval(model.parameters)

    input_ids = mx_mod.array([[1, 2, 3, 4]], dtype: mx_mod.int32)
    logits = model.call(input_ids)
    mx_mod.eval(logits)

    # Output shape should be [batch=1, seq_len=4, vocab_size=100]
    assert_equal [1, 4, 100], logits.shape
  end

  # Test 3.2: RoPE embeddings produce correct shapes
  def test_rope_output_shape
    rope = MLX::NN::RoPE.new(8)
    x = mx.ones([1, 4, 2, 8])  # B, L, n_heads, head_dim
    out = rope.call(x)
    mx.eval(out)
    assert_equal [1, 4, 2, 8], out.shape
  end

  # Test 3.3: GQA attention output shape
  def test_attention_output_shape
    args = MlxLm::Models::Llama::ModelArgs.from_dict(
      "hidden_size" => 32,
      "num_hidden_layers" => 1,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 2,
      "intermediate_size" => 64,
      "vocab_size" => 100,
    )
    attn = MlxLm::Models::Llama::Attention.new(args)
    mx.eval(attn.parameters)

    x = mx.ones([1, 4, 32])
    out = attn.call(x)
    mx.eval(out)
    assert_equal [1, 4, 32], out.shape
  end

  # Test 3.7: Forward pass with cache
  def test_forward_with_cache
    model = make_small_model
    mx.eval(model.parameters)

    cache = MlxLm::Cache.make_prompt_cache(model)

    # Prefill
    input_ids = mx.array([[1, 2, 3]], dtype: mx.int32)
    logits1 = model.call(input_ids, cache: cache)
    mx.eval(logits1)

    assert_equal [1, 3, 100], logits1.shape

    # Generation step
    next_token = mx.array([[4]], dtype: mx.int32)
    logits2 = model.call(next_token, cache: cache)
    mx.eval(logits2)

    assert_equal [1, 1, 100], logits2.shape
  end

  # Test: Model with tied embeddings
  def test_tied_embeddings
    args = MlxLm::Models::Llama::ModelArgs.from_dict(
      "hidden_size" => 32,
      "num_hidden_layers" => 1,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 64,
      "vocab_size" => 100,
      "tie_word_embeddings" => true,
    )
    model = MlxLm::Models::Llama::Model.new(args)
    mx.eval(model.parameters)

    input_ids = mx.array([[1, 2]], dtype: mx.int32)
    logits = model.call(input_ids)
    mx.eval(logits)
    assert_equal [1, 2, 100], logits.shape
  end

  def test_untied_embeddings
    args = MlxLm::Models::Llama::ModelArgs.from_dict(
      "hidden_size" => 32,
      "num_hidden_layers" => 1,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 64,
      "vocab_size" => 100,
      "tie_word_embeddings" => false,
    )
    model = MlxLm::Models::Llama::Model.new(args)
    mx.eval(model.parameters)

    input_ids = mx.array([[1, 2]], dtype: mx.int32)
    logits = model.call(input_ids)
    mx.eval(logits)
    assert_equal [1, 2, 100], logits.shape
  end

  # Test: sanitize removes rotary_emb keys
  def test_sanitize_weights
    model = make_small_model
    weights = {
      "model.layers.0.self_attn.q_proj.weight" => mx.zeros([32, 32]),
      "model.layers.0.self_attn.rotary_emb.inv_freq" => mx.zeros([16]),
      "lm_head.weight" => mx.zeros([100, 32]),
    }
    sanitized = model.sanitize(weights)
    refute sanitized.key?("model.layers.0.self_attn.rotary_emb.inv_freq")
    # tie_word_embeddings=true by default, so lm_head.weight should be removed
    refute sanitized.key?("lm_head.weight")
  end
end
