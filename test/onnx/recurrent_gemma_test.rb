# frozen_string_literal: true

require_relative "../test_helper"
require_relative "onnx_export_test"

class OnnxExportRecurrentGemmaTest < Minitest::Test
  include OnnxExportTestHelper

  def test_onnx_export
    assert_onnx_export("recurrent_gemma")
  end

  def test_onnx_compat_report
    assert_onnx_compat_report("recurrent_gemma")
  end
end
