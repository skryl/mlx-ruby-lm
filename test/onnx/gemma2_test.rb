# frozen_string_literal: true

require_relative "../test_helper"
require_relative "onnx_export_test"

class OnnxExportGemma2Test < Minitest::Test
  include OnnxExportTestHelper

  def test_onnx_export
    assert_onnx_export("gemma2")
  end

  def test_onnx_compat_report
    assert_onnx_compat_report("gemma2")
  end
end
