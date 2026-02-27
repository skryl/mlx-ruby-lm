# frozen_string_literal: true

require_relative "../test_helper"
require_relative "onnx_export_test"

class OnnxExportErnie45MoeTest < Minitest::Test
  include OnnxExportTestHelper

  def test_onnx_export
    assert_onnx_export("ernie4_5_moe")
  end

  def test_onnx_compat_report
    assert_onnx_compat_report("ernie4_5_moe")
  end
end
