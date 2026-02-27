# frozen_string_literal: true

require_relative "../test_helper"
require_relative "onnx_export_test"

class OnnxExportLlama4TextTest < Minitest::Test
  include OnnxExportTestHelper

  def test_onnx_export
    assert_onnx_export("llama4_text")
  end

  def test_onnx_compat_report
    assert_onnx_compat_report("llama4_text")
  end
end
