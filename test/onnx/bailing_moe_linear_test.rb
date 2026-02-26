# frozen_string_literal: true

require_relative "../test_helper"
require_relative "onnx_export_test"

class OnnxExportBailingMoeLinearTest < Minitest::Test
  include OnnxExportTestHelper

  def test_onnx_export
    assert_onnx_export("bailing_moe_linear")
  end

  def test_onnx_compat_report
    assert_onnx_compat_report("bailing_moe_linear")
  end
end
