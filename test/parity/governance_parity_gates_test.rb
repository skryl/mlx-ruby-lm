# frozen_string_literal: true

require_relative "../test_helper"
require "open3"
require_relative "../../tasks/parity_inventory_task"

class GovernanceParityGatesTest < Minitest::Test
  MLX_ONNX_SUBMODULE_DIR = File.expand_path("../../mlx-ruby/submodules/mlx-onnx", __dir__)
  REQUIRED_MLX_ONNX_MIN_SHA = "33d4b2eed2aa342f0836298dda60b6c5eb011b0f"
  PARITY_CHECKLIST_PATH = File.expand_path("../../prd/2026_02_25_python_ruby_parity_checklist.md", __dir__)
  REQUIRED_MLX_ONNX_OPS = {
    "ArgPartition" => /ArgPartition|arg_partition/,
    "GatherMM" => /GatherMM|gather_mm/,
  }.freeze
  CHECKLIST_STATUSES = %w[Implemented Partial Missing].freeze

  def test_parity_inventory_snapshot_is_current
    message = <<~MSG
      parity inventory snapshot is stale.
      regenerate with: bundle exec rake parity:inventory
    MSG

    assert ParityInventoryTask.run!(check: true), message
  end

  def test_mlx_onnx_checkout_meets_minimum_commit
    mlx_onnx_dir = MLX_ONNX_SUBMODULE_DIR
    skip "mlx-onnx submodule checkout not available" unless Dir.exist?(mlx_onnx_dir)

    _, _, status = Open3.capture3(
      "git", "-C", mlx_onnx_dir, "merge-base", "--is-ancestor", REQUIRED_MLX_ONNX_MIN_SHA, "HEAD"
    )
    current_sha, = Open3.capture3("git", "-C", mlx_onnx_dir, "rev-parse", "HEAD")

    message = <<~MSG
      mlx-onnx minimum commit gate failed.
      required minimum commit: #{REQUIRED_MLX_ONNX_MIN_SHA}
      current HEAD: #{current_sha.strip}
      checkout path: #{mlx_onnx_dir}
      update submodule to a commit that includes required lowering support.
    MSG

    assert status.success?, message
  end

  def test_mlx_onnx_checkout_includes_required_ops
    mlx_onnx_dir = MLX_ONNX_SUBMODULE_DIR
    skip "mlx-onnx submodule checkout not available" unless Dir.exist?(mlx_onnx_dir)

    missing_ops = REQUIRED_MLX_ONNX_OPS.keys.reject do |op_name|
      mlx_onnx_source_includes?(mlx_onnx_dir, REQUIRED_MLX_ONNX_OPS[op_name])
    end

    current_sha, = Open3.capture3("git", "-C", mlx_onnx_dir, "rev-parse", "HEAD")

    message = <<~MSG
      mlx-onnx capability gate failed.
      required ops: #{REQUIRED_MLX_ONNX_OPS.keys.join(", ")}
      missing ops: #{missing_ops.join(", ")}
      checkout path: #{mlx_onnx_dir}
      current HEAD: #{current_sha.strip}
      ensure mlx-ruby mlx-onnx checkout includes ArgPartition/GatherMM support.
    MSG

    assert missing_ops.empty?, message
  end

  def test_class_parity_checklist_is_closed_and_consistent
    assert File.exist?(PARITY_CHECKLIST_PATH), "parity checklist missing: #{PARITY_CHECKLIST_PATH}"

    checklist = File.read(PARITY_CHECKLIST_PATH)
    assert_includes checklist, "**Status:** Completed", "parity checklist status must be Completed"

    rows = parse_checklist_rows(checklist)
    refute_empty rows, "parity checklist has no class inventory rows"

    invalid_status_rows = rows.reject { |row| CHECKLIST_STATUSES.include?(row[:status]) }
    assert invalid_status_rows.empty?, "checklist includes unknown statuses: #{invalid_status_rows.map { |r| r[:status] }.uniq.join(", ")}"

    counts_by_status = Hash.new(0)
    rows.each { |row| counts_by_status[row[:status]] += 1 }
    summary_counts = parse_summary_counts(checklist)

    required_summary_keys = ["Python classes discovered", "Implemented", "Partial", "Missing"]
    missing_summary_keys = required_summary_keys.reject { |key| summary_counts.key?(key) }
    assert missing_summary_keys.empty?, "checklist summary missing keys: #{missing_summary_keys.join(", ")}"

    assert_equal rows.length, summary_counts["Python classes discovered"], "summary class count does not match checklist rows"
    assert_equal counts_by_status["Implemented"], summary_counts["Implemented"], "summary Implemented count is stale"
    assert_equal counts_by_status["Partial"], summary_counts["Partial"], "summary Partial count is stale"
    assert_equal counts_by_status["Missing"], summary_counts["Missing"], "summary Missing count is stale"
    assert_equal 0, counts_by_status["Partial"], "checklist still has Partial rows"
    assert_equal 0, counts_by_status["Missing"], "checklist still has Missing rows"
  end

  private

  SOURCE_GLOB = "**/*.{cc,cpp,c,h,hpp,hh,mm,m,py,rb}".freeze

  def parse_checklist_rows(markdown)
    markdown.each_line.filter_map do |line|
      next unless line.start_with?("|")

      cols = line.split("|")[1..-2]&.map(&:strip)
      next if cols.nil? || cols.length < 6
      next unless cols[0].end_with?(".py")

      {python_file: cols[0], status: cols[3]}
    end
  end

  def parse_summary_counts(markdown)
    counts = {}
    markdown.each_line do |line|
      match = line.match(/^\|\s*(Python classes discovered|Implemented|Partial|Missing)\s*\|\s*(\d+)\s*\|$/)
      next unless match

      counts[match[1]] = match[2].to_i
    end
    counts
  end

  def mlx_onnx_source_includes?(root_dir, pattern)
    Dir.glob(File.join(root_dir, SOURCE_GLOB)).any? do |path|
      next false unless File.file?(path)

      begin
        File.read(path).match?(pattern)
      rescue Encoding::InvalidByteSequenceError, Encoding::UndefinedConversionError
        false
      end
    end
  end
end
