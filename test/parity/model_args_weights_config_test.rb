require_relative "../test_helper"

class Phase1BaseModelArgsTest < Minitest::Test
  include ParityTestHelpers

  # Test 1.1: BaseModelArgs round-trips from a config dict identically
  def test_from_dict_filters_unknown_keys
    # Python BaseModelArgs.from_dict silently drops keys not in the dataclass
    config = {
      "model_type" => "llama",
      "hidden_size" => 128,
      "num_hidden_layers" => 4,
      "num_attention_heads" => 4,
      "num_key_value_heads" => 2,
      "intermediate_size" => 256,
      "vocab_size" => 1000,
      "rms_norm_eps" => 1e-5,
      "rope_theta" => 10000.0,
      "extra_unknown_field" => true,
      "another_unknown" => [1, 2, 3],
    }

    args = MlxLm::Models::Llama::ModelArgs.from_dict(config)

    assert_equal 128, args.hidden_size
    assert_equal 4, args.num_hidden_layers
    assert_equal 4, args.num_attention_heads
    assert_equal 2, args.num_key_value_heads
    assert_equal 256, args.intermediate_size
    assert_equal 1000, args.vocab_size
    assert_in_delta 1e-5, args.rms_norm_eps
    assert_in_delta 10000.0, args.rope_theta

    # Unknown keys should NOT raise and should NOT be stored
    refute args.respond_to?(:extra_unknown_field)
    refute args.respond_to?(:another_unknown)
  end

  def test_from_dict_uses_defaults_for_missing_keys
    config = {
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "vocab_size" => 100,
    }

    args = MlxLm::Models::Llama::ModelArgs.from_dict(config)

    assert_equal 64, args.hidden_size
    assert_equal 2, args.num_hidden_layers
    # Default values should be used for missing keys
    assert_equal "llama", args.model_type
  end
end

class Phase1WeightLoadingTest < Minitest::Test
  include ParityTestHelpers

  # Test 1.2: load_safetensors returns tensors with identical shapes/dtypes
  def test_load_safetensors_shapes_and_dtypes
    path = File.join(fixtures_dir, "test_model.safetensors")
    weights = MlxLm::WeightUtils.load_safetensors(path)

    assert_instance_of Hash, weights
    assert_equal 4, weights.size

    # Check shapes match what we created in the fixture
    q_proj = weights["model.layers.0.self_attn.q_proj.weight"]
    assert_equal [2, 2], q_proj.shape

    k_proj = weights["model.layers.0.self_attn.k_proj.weight"]
    assert_equal [2, 2], k_proj.shape

    mlp = weights["model.layers.0.mlp.gate_proj.weight"]
    assert_equal [2, 2], mlp.shape

    embed = weights["model.embed_tokens.weight"]
    assert_equal [3, 2], embed.shape
  end

  # Test 1.3: load_safetensors tensor values match Python within atol=1e-6
  def test_load_safetensors_values
    path = File.join(fixtures_dir, "test_model.safetensors")
    weights = MlxLm::WeightUtils.load_safetensors(path)

    q_proj = weights["model.layers.0.self_attn.q_proj.weight"]
    MLX::Core.eval(q_proj)
    expected = [[1.0, 2.0], [3.0, 4.0]]
    assert_arrays_close expected.flatten, q_proj.to_a.flatten, atol: 1e-6,
      msg: "q_proj values"

    embed = weights["model.embed_tokens.weight"]
    MLX::Core.eval(embed)
    expected_embed = [[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]
    assert_arrays_close expected_embed.flatten, embed.to_a.flatten, atol: 1e-6,
      msg: "embed values"
  end

  # Test 1.2b: load_sharded loads multiple files
  def test_load_sharded_safetensors
    Dir.mktmpdir do |dir|
      # Copy fixture as a shard
      FileUtils.cp(
        File.join(fixtures_dir, "test_model.safetensors"),
        File.join(dir, "model-00001-of-00001.safetensors")
      )
      weights = MlxLm::WeightUtils.load_sharded_safetensors(dir)
      assert_equal 4, weights.size
      assert weights.key?("model.layers.0.self_attn.q_proj.weight")
    end
  end
end

class Phase1TreeUnflattenTest < Minitest::Test
  include ParityTestHelpers

  # Test 1.4: tree_unflatten produces identical nested key structure
  def test_tree_unflatten_basic
    flat = {
      "model.layers.0.self_attn.q_proj.weight" => :w1,
      "model.layers.0.self_attn.k_proj.weight" => :w2,
      "model.layers.0.mlp.gate_proj.weight" => :w3,
      "model.embed_tokens.weight" => :w4,
    }

    nested = MlxLm::WeightUtils.tree_unflatten(flat)

    assert_equal :w1, nested["model"]["layers"][0]["self_attn"]["q_proj"]["weight"]
    assert_equal :w2, nested["model"]["layers"][0]["self_attn"]["k_proj"]["weight"]
    assert_equal :w3, nested["model"]["layers"][0]["mlp"]["gate_proj"]["weight"]
    assert_equal :w4, nested["model"]["embed_tokens"]["weight"]
  end

  def test_tree_unflatten_multiple_layers
    flat = {
      "model.layers.0.weight" => :a,
      "model.layers.1.weight" => :b,
      "model.layers.2.weight" => :c,
    }

    nested = MlxLm::WeightUtils.tree_unflatten(flat)

    assert_instance_of Array, nested["model"]["layers"]
    assert_equal 3, nested["model"]["layers"].size
    assert_equal :a, nested["model"]["layers"][0]["weight"]
    assert_equal :b, nested["model"]["layers"][1]["weight"]
    assert_equal :c, nested["model"]["layers"][2]["weight"]
  end
end

class Phase1ConfigTest < Minitest::Test
  include ParityTestHelpers

  # Test 1.5: Config parsing extracts identical model hyperparameters
  def test_load_config_basic
    config = MlxLm::Config.load(fixtures_dir)

    assert_equal "llama", config["model_type"]
    assert_equal 2, config["hidden_size"]
    assert_equal 1, config["num_hidden_layers"]
    assert_equal 1, config["num_attention_heads"]
    assert_equal 1, config["num_key_value_heads"]
    assert_equal 4, config["intermediate_size"]
    assert_equal 3, config["vocab_size"]
    assert_in_delta 1e-5, config["rms_norm_eps"]
    assert_in_delta 10000.0, config["rope_theta"]
  end

  def test_load_config_merges_generation_config
    config = MlxLm::Config.load(fixtures_dir)

    # generation_config.json eos_token_id should be merged
    assert_equal [2, 128001], config["eos_token_id"]
  end

  def test_load_config_preserves_extra_fields
    config = MlxLm::Config.load(fixtures_dir)

    # Extra fields in config.json should be preserved
    assert_equal true, config["extra_field_should_be_ignored"]
  end

  # Compare against Python's load_config output
  def test_config_matches_python
    py_code = <<~PY
      import json, sys
      config = json.load(open("#{fixtures_dir}/config.json"))
      gen = {}
      try:
          gen = json.load(open("#{fixtures_dir}/generation_config.json"))
      except: pass
      if eos := gen.get("eos_token_id", False):
          config["eos_token_id"] = eos
      print(json.dumps(config))
    PY

    py_config = python_eval(py_code)
    rb_config = MlxLm::Config.load(fixtures_dir)

    %w[model_type hidden_size num_hidden_layers num_attention_heads
       vocab_size intermediate_size].each do |key|
      assert_equal py_config[key], rb_config[key], "Config key '#{key}' mismatch"
    end

    assert_equal py_config["eos_token_id"], rb_config["eos_token_id"],
      "eos_token_id mismatch"
  end
end
