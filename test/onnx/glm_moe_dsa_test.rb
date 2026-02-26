# frozen_string_literal: true

require_relative "../test_helper"
require_relative "onnx_export_test"

class OnnxExportGlmMoeDsaTest < Minitest::Test
  include OnnxExportTestHelper

  def test_onnx_export
    assert_onnx_export("glm_moe_dsa")
  end

  def test_onnx_compat_report
    assert_onnx_compat_report("glm_moe_dsa")
  end
end
