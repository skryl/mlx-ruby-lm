# frozen_string_literal: true

require_relative "../test_helper"
require "tmpdir"
require "json"
require "open3"

# ONNX Export Tests
#
# For every registered model architecture, instantiate a tiny model, trace a
# forward pass through MLX::ONNX.export_onnx, and report whether export
# succeeds.  When export fails, run the compatibility report to identify the
# unsupported ops.
#
# Each model runs in an isolated subprocess because MoE models (mixtral,
# deepseek) segfault during ONNX tracing due to data-dependent control flow
# (tolist + per-token expert routing).

class OnnxExportTest < Minitest::Test
  include ParityTestHelpers

  # ── Tiny model configs ────────────────────────────────────────────────
  # Each config uses the smallest possible dimensions so instantiation and
  # tracing finish quickly without consuming real memory.

  TINY_CONFIGS = {
    "llama" => {
      "model_type" => "llama",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 128,
      "vocab_size" => 128,
      "tie_word_embeddings" => true,
    },
    "gemma" => {
      "model_type" => "gemma",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 128,
      "vocab_size" => 128,
      "head_dim" => 32,
      "tie_word_embeddings" => true,
    },
    "gemma2" => {
      "model_type" => "gemma2",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 128,
      "vocab_size" => 128,
      "head_dim" => 32,
      "query_pre_attn_scalar" => 32.0,
    },
    "qwen2" => {
      "model_type" => "qwen2",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 128,
      "vocab_size" => 128,
      "tie_word_embeddings" => true,
    },
    "phi3" => {
      "model_type" => "phi3",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 128,
      "vocab_size" => 128,
      "tie_word_embeddings" => true,
    },
    "starcoder2" => {
      "model_type" => "starcoder2",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 128,
      "vocab_size" => 128,
      "tie_word_embeddings" => true,
    },
    "stablelm" => {
      "model_type" => "stablelm",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 128,
      "vocab_size" => 128,
    },
    "cohere" => {
      "model_type" => "cohere",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 128,
      "vocab_size" => 128,
    },
    "olmo2" => {
      "model_type" => "olmo2",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 128,
      "vocab_size" => 128,
      "tie_word_embeddings" => true,
    },
    "gpt_neox" => {
      "model_type" => "gpt_neox",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "vocab_size" => 128,
      "intermediate_size" => 256,
    },
    "mixtral" => {
      "model_type" => "mixtral",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 128,
      "vocab_size" => 128,
      "num_local_experts" => 2,
      "num_experts_per_tok" => 1,
      "tie_word_embeddings" => true,
    },
    "deepseek" => {
      "model_type" => "deepseek",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 128,
      "moe_intermediate_size" => 64,
      "vocab_size" => 128,
      "n_routed_experts" => 2,
      "num_experts_per_tok" => 1,
      "n_shared_experts" => 1,
      "moe_layer_freq" => 1,
      "first_k_dense_replace" => 1,
    },
    "internlm2" => {
      "model_type" => "internlm2",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 128,
      "vocab_size" => 128,
      "bias" => false,
      "tie_word_embeddings" => true,
    },
  }.freeze

  # ── Subprocess runner ─────────────────────────────────────────────────
  # Runs a single model's ONNX export + compat report in an isolated Ruby
  # process. Returns JSON with results. Catches segfaults gracefully.

  SUBPROCESS_SCRIPT = <<~'RUBY'
    require "json"
    require "tmpdir"

    $LOAD_PATH.unshift File.expand_path("lib", __dir__)
    $LOAD_PATH.unshift File.expand_path("mlx-ruby/lib", __dir__)
    require "mlx"
    require "mlx_lm"

    config = JSON.parse(ARGV[0])
    model_type = config["model_type"]
    result = { "model_type" => model_type }

    begin
      mx = MLX::Core
      model_class, args_class = MlxLm::Models.get_classes(config)
      args = args_class.from_dict(config)
      model = model_class.new(args)

      params = MLX::Utils.tree_flatten(model.parameters).map { |_, v| v }
      mx.eval(*params) unless params.empty?

      input = mx.array([[1, 2, 3]], dtype: mx.int32)
      fun = ->(x) { model.call(x) }

      # Run compatibility report first (does not require full lowering)
      begin
        report = MLX::ONNX.export_onnx_compatibility_report(fun, input)
        result["compat_report"] = {
          "total_nodes"      => report["total_nodes"],
          "supported_nodes"  => report["supported_nodes"],
          "unsupported_nodes"=> report["unsupported_nodes"],
          "unsupported_ops"  => report["unsupported_ops"],
          "ready"            => report["ready_for_stub_conversion"],
        }
      rescue => e
        result["compat_error"] = "#{e.class}: #{e.message}"
      end

      # Attempt full ONNX export
      begin
        Dir.mktmpdir do |dir|
          path = File.join(dir, "#{model_type}.onnx")
          MLX::ONNX.export_onnx(path, fun, input)
          result["export"] = "success"
          result["onnx_size"] = File.size(path)
        end
      rescue NotImplementedError, RuntimeError => e
        result["export"] = "failed"
        result["export_error"] = e.message
      end
    rescue => e
      result["fatal"] = "#{e.class}: #{e.message}"
    end

    puts JSON.generate(result)
  RUBY

  def run_model_in_subprocess(model_type)
    config_json = JSON.generate(TINY_CONFIGS.fetch(model_type))
    project_root = File.expand_path("../..", __dir__)

    out, err, status = Open3.capture3(
      "ruby", "-e", SUBPROCESS_SCRIPT, config_json,
      chdir: project_root
    )

    if status.signaled?
      sig = status.termsig
      signal_name = Signal.signame(sig) rescue sig.to_s
      return {
        "model_type" => model_type,
        "export" => "crashed",
        "crash_signal" => signal_name,
        "stderr" => err.lines.first(5).join,
      }
    end

    unless status.success?
      return {
        "model_type" => model_type,
        "export" => "process_error",
        "exit_code" => status.exitstatus,
        "stderr" => err.lines.first(10).join,
      }
    end

    JSON.parse(out)
  rescue JSON::ParserError
    {
      "model_type" => model_type,
      "export" => "parse_error",
      "stdout" => out.to_s[0, 500],
      "stderr" => err.to_s[0, 500],
    }
  end

  # ── Per-model export tests ────────────────────────────────────────────

  TINY_CONFIGS.each_key do |model_type|
    define_method(:"test_onnx_export_#{model_type}") do
      result = run_model_in_subprocess(model_type)

      case result["export"]
      when "success"
        assert true, "#{model_type}: ONNX export succeeded (#{result['onnx_size']} bytes)"
        report = result["compat_report"]
        if report
          puts "\n  [ONNX] #{model_type}: PASS — #{report['supported_nodes']}/#{report['total_nodes']} nodes, #{result['onnx_size']} bytes"
        end

      when "failed"
        report = result["compat_report"]
        msg = "#{model_type}: ONNX export failed — #{result['export_error']}"
        if report
          unsupported = report["unsupported_ops"] || []
          msg += "\n  Nodes: #{report['supported_nodes']}/#{report['total_nodes']} supported"
          msg += "\n  Missing ops: #{unsupported.join(', ')}"
        end
        flunk(msg)

      when "crashed"
        report = result["compat_report"]
        msg = "#{model_type}: ONNX tracing crashed with signal #{result['crash_signal']}"
        if report
          unsupported = report["unsupported_ops"] || []
          msg += "\n  Compat report (pre-crash): #{report['supported_nodes']}/#{report['total_nodes']} nodes"
          msg += "\n  Missing ops: #{unsupported.empty? ? 'none' : unsupported.join(', ')}"
        end
        msg += "\n  (MoE models crash because tolist forces data-dependent control flow during tracing)"
        flunk(msg)

      else
        flunk("#{model_type}: unexpected result — #{result.inspect}")
      end
    end
  end

  # ── Compatibility report tests (always run) ───────────────────────────

  TINY_CONFIGS.each_key do |model_type|
    define_method(:"test_onnx_compat_report_#{model_type}") do
      result = run_model_in_subprocess(model_type)

      if result["compat_error"]
        skip "#{model_type}: compat report unavailable — #{result['compat_error']}"
      end

      if result["crash_signal"]
        # Crashed before we could get a report
        if result["compat_report"]
          report = result["compat_report"]
          assert_kind_of Integer, report["total_nodes"]
          unsupported = report["unsupported_ops"] || []
          pct = report["total_nodes"] > 0 ? (report["supported_nodes"].to_f / report["total_nodes"] * 100).round(1) : 0
          puts "\n  [ONNX] #{model_type}: #{report['supported_nodes']}/#{report['total_nodes']} nodes (#{pct}%) — missing: #{unsupported.empty? ? 'none' : unsupported.join(', ')} (CRASH during export)"
        else
          skip "#{model_type}: process crashed (signal #{result['crash_signal']}) before compat report"
        end
        return
      end

      report = result["compat_report"]
      skip "#{model_type}: no compat report in result" unless report

      assert_kind_of Integer, report["total_nodes"]
      assert_kind_of Integer, report["supported_nodes"]
      assert_kind_of Integer, report["unsupported_nodes"]

      total = report["total_nodes"]
      supported = report["supported_nodes"]
      unsupported_ops = report["unsupported_ops"] || []
      pct = total > 0 ? (supported.to_f / total * 100).round(1) : 0
      status = result["export"] == "success" ? "PASS" : "FAIL"
      puts "\n  [ONNX] #{model_type}: #{status} — #{supported}/#{total} nodes (#{pct}%) — missing: #{unsupported_ops.empty? ? 'none' : unsupported_ops.join(', ')}"
    end
  end
end
