require_relative "../test_helper"

# Phase 12: Advanced Features & Polish
# Tests prompt caching, perplexity, benchmarking, and model conversion utilities.

class Phase12PromptCacheTest < Minitest::Test
  include ParityTestHelpers

  def setup
    @mx = MLX::Core
  end

  # Test 1: Prompt cache can be created and used
  def test_prompt_cache_creation
    args = MlxLm::Models::Llama::ModelArgs.from_dict({
      "model_type" => "llama",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 128,
      "vocab_size" => 128,
      "tie_word_embeddings" => true,
    })
    model = MlxLm::Models::Llama::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    cache = MlxLm::Cache.make_prompt_cache(model)
    assert_equal 2, cache.length
    cache.each { |c| assert_instance_of MlxLm::KVCache, c }
  end

  # Test 2: Prompt cache save and load round-trip
  def test_prompt_cache_save_load
    dir = File.join(Dir.tmpdir, "mlx_lm_cache_test_#{$$}")
    FileUtils.mkdir_p(dir)
    begin
      args = MlxLm::Models::Llama::ModelArgs.from_dict({
        "model_type" => "llama",
        "hidden_size" => 64,
        "num_hidden_layers" => 2,
        "num_attention_heads" => 2,
        "num_key_value_heads" => 2,
        "intermediate_size" => 128,
        "vocab_size" => 128,
        "tie_word_embeddings" => true,
      })
      model = MlxLm::Models::Llama::Model.new(args)
      @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

      # Create cache and populate it
      cache = MlxLm::Cache.make_prompt_cache(model)
      input = @mx.array([[1, 2, 3]]).astype(@mx.int32)
      model.call(input, cache: cache)
      @mx.eval(*cache.map(&:state).flatten.compact)

      # Verify cache has data
      assert cache[0].offset > 0

      # Save and load
      cache_path = File.join(dir, "prompt_cache.safetensors")
      MlxLm::Cache.save_prompt_cache(cache_path, cache)
      assert File.exist?(cache_path)

      loaded = MlxLm::Cache.load_prompt_cache(cache_path, model)
      assert_equal cache.length, loaded.length
      assert_equal cache[0].offset, loaded[0].offset
    ensure
      FileUtils.rm_rf(dir)
    end
  end
end

class Phase12PerplexityTest < Minitest::Test
  include ParityTestHelpers

  def setup
    @mx = MLX::Core
  end

  # Test 3: Perplexity computation returns a positive number
  def test_perplexity_computation
    args = MlxLm::Models::Llama::ModelArgs.from_dict({
      "model_type" => "llama",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 128,
      "vocab_size" => 128,
      "tie_word_embeddings" => true,
    })
    model = MlxLm::Models::Llama::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    # Compute perplexity on a small sequence
    tokens = @mx.array([1, 2, 3, 4, 5, 6, 7, 8]).astype(@mx.int32)
    ppl = MlxLm::Perplexity.compute(model, tokens)
    assert ppl > 0, "Perplexity should be positive, got #{ppl}"
    assert ppl.is_a?(Numeric), "Perplexity should be numeric"
  end

  # Test 4: Log-likelihood scoring returns negative values
  def test_log_likelihood
    args = MlxLm::Models::Llama::ModelArgs.from_dict({
      "model_type" => "llama",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 128,
      "vocab_size" => 128,
      "tie_word_embeddings" => true,
    })
    model = MlxLm::Models::Llama::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    tokens = @mx.array([1, 2, 3, 4, 5]).astype(@mx.int32)
    ll = MlxLm::Perplexity.log_likelihood(model, tokens)
    assert ll < 0, "Log-likelihood should be negative, got #{ll}"
  end
end

class Phase12BenchmarkTest < Minitest::Test
  include ParityTestHelpers

  def setup
    @mx = MLX::Core
  end

  # Test 5: Benchmark utility measures tokens/sec
  def test_benchmark_measures_tps
    args = MlxLm::Models::Llama::ModelArgs.from_dict({
      "model_type" => "llama",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 128,
      "vocab_size" => 128,
      "tie_word_embeddings" => true,
    })
    model = MlxLm::Models::Llama::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    result = MlxLm::Benchmark.measure_generation(model, prompt_tokens: 4, gen_tokens: 4, vocab_size: 128)
    assert result.key?(:prompt_tps), "Should measure prompt tokens/sec"
    assert result.key?(:generation_tps), "Should measure generation tokens/sec"
    assert result[:prompt_tps] > 0
    assert result[:generation_tps] > 0
  end

  # Test 6: Benchmark reports model stats
  def test_benchmark_model_stats
    args = MlxLm::Models::Llama::ModelArgs.from_dict({
      "model_type" => "llama",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 128,
      "vocab_size" => 128,
      "tie_word_embeddings" => true,
    })
    model = MlxLm::Models::Llama::Model.new(args)

    stats = MlxLm::Benchmark.model_stats(model)
    assert stats.key?(:total_params)
    assert stats[:total_params] > 0
  end
end

class Phase12ModelRegistryTest < Minitest::Test
  # Test 7: All implemented models are registered
  def test_all_models_registered
    expected = %w[llama gemma qwen2 phi3 starcoder2 stablelm cohere gemma2
                  olmo2 gpt_neox mixtral deepseek internlm2]
    expected.each do |model_type|
      assert MlxLm::Models::REGISTRY.key?(model_type),
        "#{model_type} should be registered"
    end
  end

  # Test 8: Mistral remaps to Llama
  def test_model_remapping
    config = { "model_type" => "mistral" }
    model_class, _args_class = MlxLm::Models.get_classes(config)
    assert_equal MlxLm::Models::Llama::Model, model_class
  end

  # Test 9: Registry count covers all architectures
  def test_registry_count
    assert MlxLm::Models::REGISTRY.size >= 13,
      "Should have at least 13 registered architectures, got #{MlxLm::Models::REGISTRY.size}"
  end
end

class Phase12ConvertUtilsTest < Minitest::Test
  include ParityTestHelpers

  def setup
    @mx = MLX::Core
  end

  # Test 10: Weight dtype conversion utility
  def test_weight_dtype_conversion
    weight = @mx.ones([4, 4]).astype(@mx.float32)
    converted = MlxLm::ConvertUtils.convert_dtype(weight, "float16")
    @mx.eval(converted)
    assert_equal @mx.float16, converted.dtype
  end

  # Test 11: Model parameter counting
  def test_parameter_counting
    args = MlxLm::Models::Llama::ModelArgs.from_dict({
      "model_type" => "llama",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 128,
      "vocab_size" => 128,
      "tie_word_embeddings" => true,
    })
    model = MlxLm::Models::Llama::Model.new(args)

    count = MlxLm::ConvertUtils.count_parameters(model)
    assert count > 0, "Should count model parameters"
    assert count.is_a?(Integer), "Parameter count should be integer"
  end

  # Test 12: Model size estimation in bytes
  def test_model_size_estimation
    args = MlxLm::Models::Llama::ModelArgs.from_dict({
      "model_type" => "llama",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 128,
      "vocab_size" => 128,
      "tie_word_embeddings" => true,
    })
    model = MlxLm::Models::Llama::Model.new(args)

    size = MlxLm::ConvertUtils.model_size_bytes(model)
    assert size > 0, "Model size should be positive"
  end
end
