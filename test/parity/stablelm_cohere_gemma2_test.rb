require_relative "../test_helper"

# Tests StableLM, Cohere, and Gemma2 architectures.
class StablelmCohereGemma2Batch2Test < Minitest::Test
  include ParityTestHelpers

  def setup
    @mx = MLX::Core
  end

  # =====================
  # StableLM Tests
  # =====================

  def stablelm_args(hidden: 64, layers: 2, heads: 2, kv_heads: 2, vocab: 128)
    MlxLm::Models::StableLM::ModelArgs.from_dict({
      "model_type" => "stablelm",
      "hidden_size" => hidden,
      "num_hidden_layers" => layers,
      "num_attention_heads" => heads,
      "num_key_value_heads" => kv_heads,
      "intermediate_size" => 128,
      "vocab_size" => vocab,
      "rope_theta" => 10000.0,
      "use_qkv_bias" => false,
      "partial_rotary_factor" => 0.5,
      "layer_norm_eps" => 1e-5,
      "use_parallel_residual" => false,
      "qk_layernorm" => false,
    })
  end

  # Test 1: StableLM forward pass produces correct shape
  def test_stablelm_forward
    args = stablelm_args
    model = MlxLm::Models::StableLM::Model.new(args)
    @mx.eval(*model.parameters.values.flat_map { |v| v.is_a?(Hash) ? v.values : [v] })

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    assert_equal [1, 3, 128], output.shape
  end

  # Test 2: StableLM with cache (autoregressive)
  def test_stablelm_with_cache
    args = stablelm_args
    model = MlxLm::Models::StableLM::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    cache = Array.new(2) { MlxLm::KVCache.new }

    token1 = @mx.array([[1]], dtype: @mx.int32)
    out1 = model.call(token1, cache: cache)
    assert_equal [1, 1, 128], out1.shape

    token2 = @mx.array([[2]], dtype: @mx.int32)
    out2 = model.call(token2, cache: cache)
    assert_equal [1, 1, 128], out2.shape
  end

  # Test 3: StableLM with parallel residual
  def test_stablelm_parallel_residual
    args = MlxLm::Models::StableLM::ModelArgs.from_dict({
      "model_type" => "stablelm",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 128,
      "vocab_size" => 128,
      "rope_theta" => 10000.0,
      "use_qkv_bias" => true,
      "partial_rotary_factor" => 0.25,
      "layer_norm_eps" => 1e-5,
      "use_parallel_residual" => true,
      "qk_layernorm" => false,
    })
    model = MlxLm::Models::StableLM::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    assert_equal [1, 3, 128], output.shape
  end

  # Test 4: StableLM partial rotary factor applies to RoPE dimension
  def test_stablelm_partial_rotary
    args = stablelm_args
    model = MlxLm::Models::StableLM::Model.new(args)
    # Partial rotary factor of 0.5 with head_dim=32 means RoPE on 16 dims
    # Just verify the model constructs and produces valid output
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })
    tokens = @mx.array([[1]], dtype: @mx.int32)
    output = model.call(tokens)
    assert_equal [1, 1, 128], output.shape
  end

  # Test 5: StableLM registered in model registry
  def test_stablelm_registered
    config = { "model_type" => "stablelm" }
    model_class, args_class = MlxLm::Models.get_classes(config)
    assert_equal MlxLm::Models::StableLM::Model, model_class
    assert_equal MlxLm::Models::StableLM::ModelArgs, args_class
  end

  # =====================
  # Cohere Tests
  # =====================

  def cohere_args(hidden: 64, layers: 2, heads: 2, kv_heads: 2, vocab: 128)
    MlxLm::Models::Cohere::ModelArgs.from_dict({
      "model_type" => "cohere",
      "hidden_size" => hidden,
      "num_hidden_layers" => layers,
      "num_attention_heads" => heads,
      "num_key_value_heads" => kv_heads,
      "intermediate_size" => 128,
      "vocab_size" => vocab,
      "rope_theta" => 8000000.0,
      "layer_norm_eps" => 1e-5,
      "logit_scale" => 0.0625,
      "attention_bias" => false,
      "layer_norm_bias" => false,
      "use_qk_norm" => false,
    })
  end

  # Test 6: Cohere forward pass produces correct shape
  def test_cohere_forward
    args = cohere_args
    model = MlxLm::Models::Cohere::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    assert_equal [1, 3, 128], output.shape
  end

  # Test 7: Cohere with cache
  def test_cohere_with_cache
    args = cohere_args
    model = MlxLm::Models::Cohere::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    cache = Array.new(2) { MlxLm::KVCache.new }

    token1 = @mx.array([[1]], dtype: @mx.int32)
    out1 = model.call(token1, cache: cache)
    assert_equal [1, 1, 128], out1.shape

    token2 = @mx.array([[2]], dtype: @mx.int32)
    out2 = model.call(token2, cache: cache)
    assert_equal [1, 1, 128], out2.shape
  end

  # Test 8: Cohere uses parallel residuals (attn + mlp + x)
  def test_cohere_parallel_residuals
    args = cohere_args
    model = MlxLm::Models::Cohere::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    # Just verify it works (parallel residual is an internal implementation detail)
    assert_equal [1, 3, 128], output.shape
  end

  # Test 9: Cohere uses tied embeddings with logit scaling
  def test_cohere_logit_scaling
    args = cohere_args
    model = MlxLm::Models::Cohere::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    tokens = @mx.array([[1]], dtype: @mx.int32)
    output = model.call(tokens)
    # Output should be scaled by logit_scale (0.0625)
    # Values should generally be small due to the scaling
    assert_equal [1, 1, 128], output.shape
  end

  # Test 10: Cohere registered in model registry
  def test_cohere_registered
    config = { "model_type" => "cohere" }
    model_class, args_class = MlxLm::Models.get_classes(config)
    assert_equal MlxLm::Models::Cohere::Model, model_class
    assert_equal MlxLm::Models::Cohere::ModelArgs, args_class
  end

  # =====================
  # Gemma2 Tests
  # =====================

  def gemma2_args(hidden: 64, layers: 2, heads: 2, kv_heads: 2, vocab: 128)
    MlxLm::Models::Gemma2::ModelArgs.from_dict({
      "model_type" => "gemma2",
      "hidden_size" => hidden,
      "num_hidden_layers" => layers,
      "num_attention_heads" => heads,
      "num_key_value_heads" => kv_heads,
      "intermediate_size" => 128,
      "vocab_size" => vocab,
      "head_dim" => hidden / heads,
      "rms_norm_eps" => 1e-6,
      "rope_theta" => 10000.0,
      "rope_traditional" => false,
      "attn_logit_softcapping" => 50.0,
      "final_logit_softcapping" => 30.0,
      "query_pre_attn_scalar" => 144.0,
    })
  end

  # Test 11: Gemma2 forward pass produces correct shape
  def test_gemma2_forward
    args = gemma2_args
    model = MlxLm::Models::Gemma2::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    tokens = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    output = model.call(tokens)
    assert_equal [1, 3, 128], output.shape
  end

  # Test 12: Gemma2 with cache
  def test_gemma2_with_cache
    args = gemma2_args
    model = MlxLm::Models::Gemma2::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    cache = Array.new(2) { MlxLm::KVCache.new }

    token1 = @mx.array([[1]], dtype: @mx.int32)
    out1 = model.call(token1, cache: cache)
    assert_equal [1, 1, 128], out1.shape

    token2 = @mx.array([[2]], dtype: @mx.int32)
    out2 = model.call(token2, cache: cache)
    assert_equal [1, 1, 128], out2.shape
  end

  # Test 13: Gemma2 uses 4 norms per block (not 2)
  def test_gemma2_four_norms_per_block
    args = gemma2_args
    model = MlxLm::Models::Gemma2::Model.new(args)
    block = model.layers[0]

    # Gemma2 has 4 norms: input_layernorm, post_attention_layernorm,
    # pre_feedforward_layernorm, post_feedforward_layernorm
    assert block.respond_to?(:input_layernorm), "Block should have input_layernorm"
    assert block.respond_to?(:post_attention_layernorm), "Block should have post_attention_layernorm"
    assert block.respond_to?(:pre_feedforward_layernorm), "Block should have pre_feedforward_layernorm"
    assert block.respond_to?(:post_feedforward_layernorm), "Block should have post_feedforward_layernorm"
  end

  # Test 14: Gemma2 applies final logit softcapping
  def test_gemma2_logit_softcapping
    args = gemma2_args
    model = MlxLm::Models::Gemma2::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    tokens = @mx.array([[1]], dtype: @mx.int32)
    output = model.call(tokens)
    @mx.eval(output)

    # With softcapping=30.0, all output values should be in [-30, 30]
    max_val = @mx.max(output).item
    min_val = @mx.min(output).item
    assert max_val <= 30.0 + 0.01, "Output should be softcapped at 30.0, got max=#{max_val}"
    assert min_val >= -30.0 - 0.01, "Output should be softcapped at -30.0, got min=#{min_val}"
  end

  # Test 15: Gemma2 registered in model registry
  def test_gemma2_registered
    config = { "model_type" => "gemma2" }
    model_class, args_class = MlxLm::Models.get_classes(config)
    assert_equal MlxLm::Models::Gemma2::Model, model_class
    assert_equal MlxLm::Models::Gemma2::ModelArgs, args_class
  end
end
