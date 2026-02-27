require_relative "../test_helper"
require "set"

class RegistryModelLoadingTokenizerTest < Minitest::Test
  include ParityTestHelpers

  def setup
    @mx = MLX::Core
  end

  # Test 1: Llama is registered in the model registry
  def test_llama_registered
    assert MlxLm::Models::REGISTRY.key?("llama"), "Llama should be registered"
    model_class, args_class = MlxLm::Models::REGISTRY["llama"]
    assert_equal MlxLm::Models::Llama::Model, model_class
    assert_equal MlxLm::Models::Llama::ModelArgs, args_class
  end

  # Test 2: get_classes resolves 'llama' model_type
  def test_get_classes_llama
    config = { "model_type" => "llama" }
    model_class, args_class = MlxLm::Models.get_classes(config)
    assert_equal MlxLm::Models::Llama::Model, model_class
    assert_equal MlxLm::Models::Llama::ModelArgs, args_class
  end

  # Test 3: get_classes resolves 'mistral' via remapping to llama
  def test_get_classes_mistral_remap
    config = { "model_type" => "mistral" }
    model_class, args_class = MlxLm::Models.get_classes(config)
    assert_equal MlxLm::Models::Llama::Model, model_class
  end

  # Test 4: get_classes raises for unknown architecture
  def test_get_classes_unknown
    config = { "model_type" => "unknown_arch_xyz" }
    assert_raises(ArgumentError) do
      MlxLm::Models.get_classes(config)
    end
  end

  # Test 5: get_classes raises for missing model_type
  def test_get_classes_missing_model_type
    config = {}
    assert_raises(ArgumentError) do
      MlxLm::Models.get_classes(config)
    end
  end

  # Test 6: ModelArgs.from_dict creates args with correct defaults
  def test_model_args_from_dict
    config = {
      "model_type" => "llama",
      "hidden_size" => 256,
      "num_hidden_layers" => 4,
      "num_attention_heads" => 4,
      "intermediate_size" => 512,
      "vocab_size" => 100,
    }
    args = MlxLm::Models::Llama::ModelArgs.from_dict(config)
    assert_equal 256, args.hidden_size
    assert_equal 4, args.num_hidden_layers
    assert_equal 4, args.num_attention_heads
    assert_equal 100, args.vocab_size
    # Default should be set
    assert_equal 1e-6, args.rms_norm_eps
  end

  # Test 7: ModelArgs.from_dict filters unknown keys
  def test_model_args_filters_unknown
    config = {
      "model_type" => "llama",
      "hidden_size" => 128,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "intermediate_size" => 256,
      "vocab_size" => 50,
      "unknown_key_that_should_be_ignored" => 999,
      "another_unknown" => "hello",
    }
    # Should not raise
    args = MlxLm::Models::Llama::ModelArgs.from_dict(config)
    assert_equal 128, args.hidden_size
  end
end

class RegistryModelLoadingTokenizerLoadModelTest < Minitest::Test
  include ParityTestHelpers

  def setup
    @mx = MLX::Core
    @model_dir = setup_tiny_model
  end

  def teardown
    FileUtils.rm_rf(@model_dir) if @model_dir && File.exist?(@model_dir)
  end

  # Create a tiny Llama model directory with config + weights for testing
  def setup_tiny_model
    dir = File.join(Dir.tmpdir, "mlx_lm_test_model_#{$$}")
    FileUtils.mkdir_p(dir)

    # Create a minimal config.json
    config = {
      "model_type" => "llama",
      "hidden_size" => 32,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 64,
      "vocab_size" => 100,
      "rms_norm_eps" => 1e-6,
      "rope_theta" => 10000.0,
      "tie_word_embeddings" => true,
      "max_position_embeddings" => 128,
    }
    File.write(File.join(dir, "config.json"), JSON.generate(config))

    # Create model weights using Python/MLX to produce valid safetensors
    # Instead, we'll create the model in Ruby and save its random weights
    args = MlxLm::Models::Llama::ModelArgs.from_dict(config)
    model = MlxLm::Models::Llama::Model.new(args)

    # Collect parameters and save as safetensors
    params = model.parameters
    save_as_safetensors(params, File.join(dir, "model.safetensors"))

    dir
  end

  def save_as_safetensors(params, path)
    # Build safetensors format manually using Ruby
    require "json"

    # Flatten nested parameter hash to flat "key.subkey" => array
    flat_params = MLX::Utils.tree_flatten(params, destination: {})

    tensors = {}
    flat_params.each do |name, arr|
      @mx.eval(arr)
      shape = arr.shape
      dtype = arr.dtype

      # Convert to float32 for simplicity
      arr = arr.astype(@mx.float32) unless dtype == @mx.float32
      @mx.eval(arr)

      data = arr.tolist
      data = data.flatten if data.is_a?(::Array) && data.first.is_a?(::Array)
      data = [data].flatten
      binary = data.pack("e*")

      tensors[name] = {
        "dtype" => "float32",
        "shape" => shape,
        "data" => binary,
      }
    end

    File.binwrite(path, Safetensors.serialize(tensors))
  end

  # Test 8: load_model loads a tiny Llama model from directory
  def test_load_model
    model, config = MlxLm::LoadUtils.load_model(@model_dir)
    assert_instance_of MlxLm::Models::Llama::Model, model
    assert_equal "llama", config["model_type"]
    assert_equal 2, model.layers.length
  end

  # Test 9: Model sanitize removes rotary embedding weights
  def test_model_sanitize
    config = JSON.parse(File.read(File.join(@model_dir, "config.json")))
    args = MlxLm::Models::Llama::ModelArgs.from_dict(config)
    model = MlxLm::Models::Llama::Model.new(args)

    weights = {
      "model.embed_tokens.weight" => @mx.zeros([100, 32]).astype(@mx.float32),
      "model.layers.0.self_attn.rotary_emb.inv_freq" => @mx.zeros([16]).astype(@mx.float32),
      "model.layers.0.self_attn.q_proj.weight" => @mx.zeros([32, 32]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)

    # rotary_emb.inv_freq should be removed
    refute sanitized.key?("model.layers.0.self_attn.rotary_emb.inv_freq"),
      "sanitize should remove rotary_emb.inv_freq"
    # Other weights should remain
    assert sanitized.key?("model.embed_tokens.weight")
    assert sanitized.key?("model.layers.0.self_attn.q_proj.weight")
  end

  # Test 10: Model sanitize removes lm_head.weight for tied embeddings
  def test_model_sanitize_tied_embeddings
    config = JSON.parse(File.read(File.join(@model_dir, "config.json")))
    args = MlxLm::Models::Llama::ModelArgs.from_dict(config)
    model = MlxLm::Models::Llama::Model.new(args)

    weights = {
      "model.embed_tokens.weight" => @mx.zeros([100, 32]).astype(@mx.float32),
      "lm_head.weight" => @mx.zeros([100, 32]).astype(@mx.float32),
    }

    sanitized = model.sanitize(weights)
    refute sanitized.key?("lm_head.weight"),
      "sanitize should remove lm_head.weight when tie_word_embeddings=true"
  end
end

class RegistryModelLoadingTokenizerTokenizerWrapperFromPathTest < Minitest::Test
  include ParityTestHelpers

  # Test 11: TokenizerWrapper from path has eos_token_ids
  def test_tokenizer_wrapper_from_path
    path = File.join(fixtures_dir, "gpt2_tokenizer")
    tokenizer = MlxLm::TokenizerWrapper.new(path)

    assert_respond_to tokenizer, :eos_token_ids
    assert_respond_to tokenizer, :detokenizer
    assert_respond_to tokenizer, :encode
    assert_respond_to tokenizer, :decode
  end

  # Test 12: TokenizerWrapper from Tokenizers object with overrides
  def test_tokenizer_wrapper_from_object
    path = File.join(fixtures_dir, "gpt2_tokenizer", "tokenizer.json")
    raw = Tokenizers::Tokenizer.from_file(path)

    tokenizer = MlxLm::TokenizerWrapper.new(raw, eos_token_id: [50256])
    assert tokenizer.eos_token_ids.include?(50256),
      "Should include overridden eos_token_id"
  end

  # Test 13: StreamingDetokenizer has last_segment
  def test_streaming_detokenizer_last_segment
    path = File.join(fixtures_dir, "gpt2_tokenizer")
    tokenizer = MlxLm::TokenizerWrapper.new(path)
    detok = tokenizer.detokenizer

    assert_respond_to detok, :last_segment
    assert_respond_to detok, :add_token
    assert_respond_to detok, :finalize

    # After adding a token, last_segment should be non-nil
    detok.add_token(15496)  # "Hello" in GPT-2
    refute_nil detok.last_segment
  end
end
