# frozen_string_literal: true

require_relative "../test_helper"
require_relative "onnx_export_test"

class OnnxExportOlmoTest < Minitest::Test
  include OnnxExportTestHelper

  def test_onnx_export
    assert_onnx_export("olmo")
  end

  def test_onnx_compat_report
    assert_onnx_compat_report("olmo")
  end
end
