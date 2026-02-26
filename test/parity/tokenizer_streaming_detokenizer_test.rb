require_relative "../test_helper"

class Phase2TokenizerTest < Minitest::Test
  include ParityTestHelpers

  def setup
    @tokenizer_path = File.join(fixtures_dir, "gpt2_tokenizer")
    @reference = JSON.parse(File.read(File.join(fixtures_dir, "gpt2_tokenizer_reference.json")))
    @tokenizer = MlxLm::TokenizerWrapper.new(@tokenizer_path)
  end

  # Test 2.1: encode produces identical token ID arrays
  def test_encode_matches_python
    test_strings = [
      "Hello, world!",
      "The quick brown fox jumps over the lazy dog.",
      "Machine learning is transforming the world.",
      "  Multiple   spaces  and\ttabs",
      "1234567890",
      "Special chars: @#$%^&*()",
      "A",
      "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
      "Unicode: café résumé naïve",
    ]

    test_strings.each do |s|
      ref = @reference[s]
      next unless ref

      ruby_ids = @tokenizer.encode(s)
      assert_equal ref["ids"], ruby_ids, "encode mismatch for: #{s.inspect}"
    end
  end

  # Test 2.2: decode produces identical text
  def test_decode_matches_python
    test_strings = [
      "Hello, world!",
      "The quick brown fox jumps over the lazy dog.",
      "Machine learning is transforming the world.",
      "1234567890",
    ]

    test_strings.each do |s|
      ref = @reference[s]
      next unless ref

      decoded = @tokenizer.decode(ref["ids"])
      assert_equal ref["decoded"], decoded, "decode mismatch for ids from: #{s.inspect}"
    end
  end

  # Test 2.5: Special token handling (EOS, BOS)
  def test_special_tokens
    meta = @reference["__meta__"]
    assert_equal meta["eos_token_id"], @tokenizer.eos_token_id
    assert_equal meta["vocab_size"], @tokenizer.vocab_size
  end

  # Test 2.6: encode + decode round-trip
  def test_encode_decode_roundtrip
    strings = [
      "Hello, world!",
      "The quick brown fox jumps over the lazy dog.",
      "1234567890",
      "def foo(): pass",
      "Special chars: @#$%^&*()",
    ]

    strings.each do |s|
      ids = @tokenizer.encode(s)
      decoded = @tokenizer.decode(ids)
      assert_equal s, decoded, "Round-trip failed for: #{s.inspect}"
    end
  end
end

class Phase2StreamingDetokenizerTest < Minitest::Test
  include ParityTestHelpers

  def setup
    @tokenizer_path = File.join(fixtures_dir, "gpt2_tokenizer")
    @tokenizer = MlxLm::TokenizerWrapper.new(@tokenizer_path)
  end

  # Test 2.4: StreamingDetokenizer accumulates to the same final string
  def test_streaming_produces_same_final_string
    text = "The quick brown fox jumps over the lazy dog."
    ids = @tokenizer.encode(text)

    detok = MlxLm::StreamingDetokenizer.new(@tokenizer)
    segments = []
    ids.each do |id|
      segment = detok.add_token(id)
      segments << segment if segment && !segment.empty?
    end
    segments << detok.finalize

    full_output = segments.join
    assert_equal text, full_output
  end

  def test_streaming_incremental_output
    text = "Hello, world! This is a test."
    ids = @tokenizer.encode(text)

    detok = MlxLm::StreamingDetokenizer.new(@tokenizer)
    segments = []
    ids.each do |id|
      segment = detok.add_token(id)
      segments << segment if segment && !segment.empty?
    end
    segments << detok.finalize

    # Should produce multiple segments (not all in one chunk)
    assert segments.length > 1, "Expected multiple incremental segments"
    assert_equal text, segments.join
  end
end
