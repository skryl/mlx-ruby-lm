# frozen_string_literal: true

require_relative "../test_helper"
require_relative "onnx_export_test"

class OnnxExportFalconH1Test < Minitest::Test
  include OnnxExportTestHelper

  def test_onnx_export
    assert_onnx_export("falcon_h1")
  end

  def test_onnx_compat_report
    assert_onnx_compat_report("falcon_h1")
  end
end
