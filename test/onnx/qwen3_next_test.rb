# frozen_string_literal: true

require_relative "../test_helper"
require_relative "onnx_export_test"

class OnnxExportQwen3NextTest < Minitest::Test
  include OnnxExportTestHelper

  def test_onnx_export
    assert_onnx_export("qwen3_next")
  end

  def test_onnx_compat_report
    assert_onnx_compat_report("qwen3_next")
  end
end
