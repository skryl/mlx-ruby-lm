# frozen_string_literal: true

require_relative "../test_helper"
require_relative "onnx_export_test"

class OnnxExportHunyuanV1DenseTest < Minitest::Test
  include OnnxExportTestHelper

  def test_onnx_export
    assert_onnx_export("hunyuan_v1_dense")
  end

  def test_onnx_compat_report
    assert_onnx_compat_report("hunyuan_v1_dense")
  end
end
