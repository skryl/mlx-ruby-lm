# frozen_string_literal: true

require_relative "../test_helper"
require "open3"
require_relative "../../tasks/parity_inventory_task"

class Phase13GovernanceGatesTest < Minitest::Test
  MIN_MLX_ONNX_SHA = "33d4b2eed2aa342f0836298dda60b6c5eb011b0f"
  MLX_ONNX_DIR = File.expand_path("../../mlx-ruby/submodules/mlx-onnx", __dir__)

  def test_parity_inventory_snapshot_is_current
    message = <<~MSG
      parity inventory snapshot is stale.
      regenerate with: bundle exec rake parity:inventory
    MSG

    assert ParityInventoryTask.run!(check: true), message
  end

  def test_mlx_onnx_submodule_meets_minimum_onnx_commit
    _, _, repo_status = Open3.capture3("git", "-C", MLX_ONNX_DIR, "rev-parse", "--is-inside-work-tree")
    skip "mlx-onnx submodule checkout not available" unless repo_status.success?

    current_sha, _, current_status = Open3.capture3("git", "-C", MLX_ONNX_DIR, "rev-parse", "HEAD")
    assert current_status.success?, "failed to read mlx-onnx HEAD"

    _, err, status = Open3.capture3(
      "git", "-C", MLX_ONNX_DIR, "merge-base", "--is-ancestor", MIN_MLX_ONNX_SHA, "HEAD"
    )

    message = <<~MSG
      mlx-onnx commit gate failed.
      required minimum: #{MIN_MLX_ONNX_SHA}
      current HEAD: #{current_sha.strip}
      ensure mlx-ruby/submodules/mlx-onnx is pinned to a commit that includes ArgPartition/GatherMM support.
      git stderr: #{err}
    MSG

    assert status.success?, message
  end
end
