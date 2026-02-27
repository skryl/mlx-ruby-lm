require_relative "../test_helper"
require "pathname"
require "tmpdir"

class MissingUtilityModulesTest < Minitest::Test
  include ParityTestHelpers

  class IdentityModule < MLX::NN::Module
    def call(x, *_args, **_kwargs)
      x
    end
  end

  class SwitchLinear < MLX::NN::Module
    def call(x, *_args, **_kwargs)
      x
    end
  end

  def setup
    @mx = MLX::Core
  end

  def test_gguf_enum_values
    assert_equal 1, MlxLm::TokenType::NORMAL
    assert_equal 6, MlxLm::TokenType::BYTE
    assert_equal 1, MlxLm::GGMLFileType::GGML_TYPE_F16
  end

  def test_hf_vocab_load_and_iterators
    tokenizer_file = File.join(fixtures_dir, "gpt2_tokenizer", "tokenizer.json")
    vocab = MlxLm::HfVocab.load(tokenizer_file)

    assert_operator vocab.vocab_size_base, :>, 0
    assert_operator vocab.vocab_size, :>=, vocab.vocab_size_base

    sample = vocab.hf_tokens.first(5)
    assert_equal 5, sample.length
    sample.each do |token_text, score, token_type|
      assert token_text.is_a?(String)
      assert_in_delta(-1000.0, score)
      assert_includes [
        MlxLm::TokenType::NORMAL,
        MlxLm::TokenType::CONTROL,
        MlxLm::TokenType::BYTE,
      ], token_type
    end

    combined = vocab.all_tokens.first(10)
    assert_equal 10, combined.length
  end

  def test_directory_entry_comparison_and_factory
    Dir.mktmpdir("mlx_lm_share_test") do |dir|
      root = Pathname.new(dir)
      (root / "a_dir").mkpath
      (root / "z_file.txt").write("hello")
      File.symlink("z_file.txt", (root / "m_link").to_s)

      entries = [
        MlxLm::DirectoryEntry.from_path(root, root / "z_file.txt"),
        MlxLm::DirectoryEntry.from_path(root, root / "m_link"),
        MlxLm::DirectoryEntry.from_path(root, root / "a_dir"),
      ].sort

      assert_equal ["directory", "symlink", "file"], entries.map(&:entry_type)

      lhs = MlxLm::DirectoryEntry.new("file", "z_file.txt")
      rhs = MlxLm::DirectoryEntry.from_path(root, root / "z_file.txt")
      assert_equal lhs, rhs
    end
  end

  def test_awq_config_and_scale_config
    scale_cfg = MlxLm::Quant::Awq::ScaleConfig.new(
      prev: "input_layernorm",
      layers: ["q_proj", "k_proj", "v_proj"],
      block: "self_attn",
      kwargs: ["mask"],
      use_config: ->(block) { !block.nil? }
    )
    awq_cfg = MlxLm::Quant::Awq::AWQConfig.new(
      embed: "embed_tokens",
      lm_head: "lm_head",
      no_clip: ["q_proj"],
      scale_configs: [scale_cfg]
    )

    assert_equal "embed_tokens", awq_cfg.embed
    assert_equal "self_attn", scale_cfg.block
    assert_equal true, scale_cfg.use_for?(Object.new)
  end

  def test_awq_catcher_captures_input_features
    mod = SwitchLinear.new
    catcher = MlxLm::Quant::Awq::Catcher.new(mod)

    x1 = @mx.ones([2, 4], @mx.float32)
    x2 = @mx.ones([1, 4], @mx.float32)
    i1 = @mx.array([0, 1], dtype: @mx.int32)
    i2 = @mx.array([2], dtype: @mx.int32)

    out1 = catcher.call(x1, i1)
    out2 = catcher.call(x2, i2)

    input_feat = mod.instance_variable_get(:@input_feat)
    indices = mod.instance_variable_get(:@indices)
    @mx.eval(out1, out2, input_feat, indices)

    assert_equal [2, 4], out1.shape
    assert_equal [1, 4], out2.shape
    assert_equal [3, 4], input_feat.shape
    assert_equal [3], indices.shape
  end

  def test_gptq_catcher_accumulates_hessian
    mod = IdentityModule.new
    catcher = MlxLm::Quant::Gptq::Catcher.new(mod)

    x = @mx.array([[[1.0, 2.0], [3.0, 4.0]]], dtype: @mx.float32)
    out = catcher.call(x)
    h = catcher.hessian
    @mx.eval(out, h)

    assert_equal x.shape, out.shape
    assert_equal [2, 2], h.shape

    h_list = h.tolist
    assert_in_delta h_list[0][1], h_list[1][0], 1e-6
    assert_operator h_list[0][0], :>, 0.0
  end

  def test_mlxlm_class_surface_exists
    assert defined?(MlxLm::MLXLM)
    assert_includes MlxLm::MLXLM.instance_methods(false), :generate
    assert_includes MlxLm::MLXLM.instance_methods(false), :stream_generate
    assert_includes MlxLm::MLXLM.instance_methods(false), :tokenizer_name
  end
end
