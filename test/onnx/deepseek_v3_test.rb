# frozen_string_literal: true

require_relative "../test_helper"
require_relative "onnx_export_test"

class OnnxExportDeepseekV3Test < Minitest::Test
  include OnnxExportTestHelper

  def test_onnx_export
    assert_onnx_export("deepseek_v3")
  end

  def test_onnx_compat_report
    assert_onnx_compat_report("deepseek_v3")
  end
end
