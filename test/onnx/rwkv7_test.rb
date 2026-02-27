# frozen_string_literal: true

require_relative "../test_helper"
require_relative "onnx_export_test"

class OnnxExportRwkv7Test < Minitest::Test
  include OnnxExportTestHelper

  def test_onnx_export
    assert_onnx_export("rwkv7")
  end

  def test_onnx_compat_report
    assert_onnx_compat_report("rwkv7")
  end
end
