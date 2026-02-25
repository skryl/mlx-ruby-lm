require_relative "../test_helper"

# Phase 4: Sampling & Generation Engine
# Tests sampling functions (top-p, top-k, min-p, temperature, repetition penalty)
# and the generation pipeline (generate_step, stream_generate, generate).
class Phase4SamplingTest < Minitest::Test
  include ParityTestHelpers

  def setup
    @mx = MLX::Core
  end

  # Test 1: Greedy sampling (temp=0) returns argmax
  def test_greedy_sampling
    sampler = MlxLm::SampleUtils.make_sampler(temp: 0.0)
    logprobs = @mx.array([[-1.0, -0.5, -2.0, -0.1]], dtype: @mx.float32)
    token = sampler.call(logprobs)
    assert_equal 3, token.item, "Greedy should pick index of max logprob"
  end

  # Test 2: Temperature scaling changes distribution
  def test_temperature_sampling
    # With very low temp, should behave close to greedy
    # We use a fixed seed for reproducibility
    @mx.random_seed(42)
    logprobs = @mx.array([[-1.0, -0.5, -2.0, -0.1]], dtype: @mx.float32)

    # At very low temperature, categorical should pick the argmax (almost always)
    sampler_low = MlxLm::SampleUtils.make_sampler(temp: 0.01)
    counts = Hash.new(0)
    100.times do
      token = sampler_low.call(logprobs)
      counts[token.item] += 1
    end
    # Index 3 (highest logprob -0.1) should be picked most often
    assert counts[3] > 50, "Low temperature should mostly pick argmax, got: #{counts}"
  end

  # Test 3: Top-k filtering masks low-probability tokens
  def test_top_k_filtering
    logprobs = @mx.array([[-1.0, -0.5, -2.0, -0.1, -3.0]], dtype: @mx.float32)
    result = MlxLm::SampleUtils.apply_top_k(logprobs, 2)
    result_list = result.tolist.flatten

    # Only the top 2 should remain; others should be -inf
    non_inf_count = result_list.count { |v| v > -1e30 }
    assert_equal 2, non_inf_count, "Top-k=2 should keep exactly 2 tokens"

    # The two surviving tokens should be indices 1 (-0.5) and 3 (-0.1)
    assert result_list[3] > -1e30, "Index 3 (highest) should survive top-k"
    assert result_list[1] > -1e30, "Index 1 (second highest) should survive top-k"
  end

  # Test 4: Top-p (nucleus) filtering
  def test_top_p_filtering
    # Create logprobs where one token dominates
    logprobs = @mx.array([[0.0, -10.0, -10.0, -10.0]], dtype: @mx.float32)
    result = MlxLm::SampleUtils.apply_top_p(logprobs, 0.5)
    result_list = result.tolist.flatten

    # Token 0 has almost all probability mass, should survive
    assert result_list[0] > -1e30, "Dominant token should survive top-p"
  end

  # Test 5: Min-p filtering
  def test_min_p_filtering
    # One dominant token, rest very low
    logprobs = @mx.array([[-0.1, -5.0, -5.0, -5.0]], dtype: @mx.float32)
    result = MlxLm::SampleUtils.apply_min_p(logprobs, 0.5, 1)
    result_list = result.tolist.flatten

    # Token 0 is dominant - should always survive
    assert result_list[0] > -1e30, "Dominant token should survive min-p"
    # Low-probability tokens should be filtered
    assert result_list[1] < -1e30 || result_list[1] == result_list[0],
      "Low-prob tokens should be filtered by min-p"
  end

  # Test 6: Repetition penalty reduces probability of repeated tokens
  def test_repetition_penalty
    processor = MlxLm::SampleUtils.make_repetition_penalty(2.0, 20)
    logits = @mx.array([[1.0, 2.0, 3.0, 4.0]], dtype: @mx.float32)
    tokens = @mx.array([0, 2], dtype: @mx.int32) # Tokens 0 and 2 were already generated

    result = processor.call(tokens, logits)
    result_list = result.tolist.flatten

    # Token 0 (logit=1.0, positive) should be reduced by /2.0 -> 0.5
    assert_in_delta 0.5, result_list[0], 0.01, "Positive logit should be divided by penalty"
    # Token 1 (not repeated, logit=2.0) should be unchanged
    assert_in_delta 2.0, result_list[1], 0.01, "Non-repeated token should be unchanged"
    # Token 2 (logit=3.0, positive) should be reduced by /2.0 -> 1.5
    assert_in_delta 1.5, result_list[2], 0.01, "Positive logit should be divided by penalty"
    # Token 3 (not repeated) should be unchanged
    assert_in_delta 4.0, result_list[3], 0.01, "Non-repeated token should be unchanged"
  end

  # Test 7: make_sampler chaining - top_k + temperature
  def test_sampler_chaining
    @mx.random_seed(42)
    sampler = MlxLm::SampleUtils.make_sampler(temp: 1.0, top_k: 2)
    logprobs = @mx.array([[-1.0, -0.5, -2.0, -0.1, -3.0]], dtype: @mx.float32)

    counts = Hash.new(0)
    100.times do
      token = sampler.call(logprobs)
      counts[token.item] += 1
    end

    # Only tokens from top-2 (indices 1 and 3) should appear
    assert counts.keys.all? { |k| [1, 3].include?(k) },
      "With top_k=2, only top-2 tokens should be sampled, got: #{counts}"
  end

  # Test 8: make_logits_processors returns working processors
  def test_logits_processors
    processors = MlxLm::SampleUtils.make_logits_processors(
      repetition_penalty: 1.5,
      repetition_context_size: 10
    )
    assert_equal 1, processors.length, "Should return one processor for repetition penalty"
    assert processors[0].is_a?(Proc), "Processor should be a callable"
  end

  # Test 9: Categorical sampling respects distribution
  def test_categorical_sampling_distribution
    @mx.random_seed(123)
    # Make one token vastly more probable
    logprobs = @mx.array([[-10.0, -10.0, 0.0, -10.0]], dtype: @mx.float32)

    counts = Hash.new(0)
    100.times do
      token = MlxLm::SampleUtils.categorical_sampling(logprobs, 1.0)
      counts[token.item] += 1
    end

    # Token 2 should be picked the vast majority of the time
    assert counts[2] > 80, "Dominant token should be picked >80% of the time, got: #{counts}"
  end
end

class Phase4GenerateTest < Minitest::Test
  include ParityTestHelpers

  def setup
    @mx = MLX::Core
  end

  # A minimal model for testing the generation pipeline
  class DummyModel
    attr_reader :layers

    def initialize(vocab_size: 10, hidden_dim: 8, num_layers: 2)
      @vocab_size = vocab_size
      @hidden_dim = hidden_dim
      @layers = Array.new(num_layers) { nil } # Placeholder for layer count
      # Fixed weight matrix for deterministic output
      @mx = MLX::Core
      @weight = @mx.ones([@hidden_dim, @vocab_size]).astype(@mx.float32) * 0.1
    end

    def call(tokens, cache: nil)
      mx = @mx
      batch_size = tokens.shape[0]
      seq_len = tokens.shape[1]

      # Simple embedding: one-hot-ish -> linear
      # Just produce fixed logits based on position
      logits_data = []
      batch_size.times do
        seq_logits = []
        seq_len.times do |i|
          # Make token 3 have the highest logit for deterministic testing
          row = Array.new(@vocab_size, -1.0)
          row[3] = 1.0
          seq_logits << row
        end
        logits_data << seq_logits
      end

      # Update cache if present
      if cache
        cache.each do |c|
          next unless c.is_a?(MlxLm::KVCache)
          dummy_k = mx.zeros([batch_size, 1, seq_len, @hidden_dim]).astype(mx.float32)
          dummy_v = mx.zeros([batch_size, 1, seq_len, @hidden_dim]).astype(mx.float32)
          c.update_and_fetch(dummy_k, dummy_v)
        end
      end

      mx.array(logits_data, dtype: mx.float32)
    end
  end

  # Test 10: generate_step produces correct tokens from dummy model
  def test_generate_step_greedy
    model = DummyModel.new(vocab_size: 10, num_layers: 2)
    prompt = @mx.array([1, 2], dtype: @mx.uint32)

    tokens = []
    MlxLm::Generate.generate_step(prompt, model, max_tokens: 5).each do |token, logprobs|
      tokens << token
    end

    assert_equal 5, tokens.length, "Should generate exactly max_tokens tokens"
    # Our dummy model always makes token 3 have the highest logit
    assert tokens.all? { |t| t == 3 }, "Greedy should always pick token 3 from dummy model"
  end

  # Test 11: generate_step respects max_tokens
  def test_generate_step_max_tokens
    model = DummyModel.new(vocab_size: 10, num_layers: 2)
    prompt = @mx.array([1], dtype: @mx.uint32)

    tokens = []
    MlxLm::Generate.generate_step(prompt, model, max_tokens: 3).each do |token, _|
      tokens << token
    end

    assert_equal 3, tokens.length, "Should stop after max_tokens"
  end

  # Test 12: generate_step with custom sampler
  def test_generate_step_custom_sampler
    model = DummyModel.new(vocab_size: 10, num_layers: 2)
    prompt = @mx.array([1, 2], dtype: @mx.uint32)

    # Custom sampler that always picks token 7
    custom_sampler = ->(_logprobs) { @mx.array([7], dtype: @mx.int32) }

    tokens = []
    MlxLm::Generate.generate_step(prompt, model, max_tokens: 3, sampler: custom_sampler).each do |token, _|
      tokens << token
    end

    assert tokens.all? { |t| t == 7 }, "Custom sampler should always pick token 7"
  end

  # Test 13: GenerationResponse struct has all required fields
  def test_generation_response_struct
    resp = MlxLm::GenerationResponse.new(
      text: "hello",
      token: 42,
      logprobs: @mx.array([0.0]),
      prompt_tokens: 5,
      prompt_tps: 100.0,
      generation_tokens: 1,
      generation_tps: 50.0,
      peak_memory: 1.5,
      finish_reason: "stop"
    )

    assert_equal "hello", resp.text
    assert_equal 42, resp.token
    assert_equal 5, resp.prompt_tokens
    assert_equal "stop", resp.finish_reason
  end
end
