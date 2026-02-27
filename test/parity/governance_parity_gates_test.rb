# frozen_string_literal: true

require_relative "../test_helper"
require "open3"
require_relative "../../tasks/parity_inventory_task"

class Phase13GovernanceGatesTest < Minitest::Test
  MLX_ONNX_SUBMODULE_DIR = File.expand_path("../../mlx-ruby/submodules/mlx-onnx", __dir__)
  REQUIRED_MLX_ONNX_OPS = {
    "ArgPartition" => /ArgPartition|arg_partition/,
    "GatherMM" => /GatherMM|gather_mm/,
  }.freeze

  def test_parity_inventory_snapshot_is_current
    message = <<~MSG
      parity inventory snapshot is stale.
      regenerate with: bundle exec rake parity:inventory
    MSG

    assert ParityInventoryTask.run!(check: true), message
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

  private

  SOURCE_GLOB = "**/*.{cc,cpp,c,h,hpp,hh,mm,m,py,rb}".freeze

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
