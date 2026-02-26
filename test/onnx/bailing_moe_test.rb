# frozen_string_literal: true

require_relative "../test_helper"
require_relative "onnx_export_test"

class OnnxExportBailingMoeTest < Minitest::Test
  include OnnxExportTestHelper

  def test_onnx_export
    assert_onnx_export("bailing_moe")
  end

  def test_onnx_compat_report
    assert_onnx_compat_report("bailing_moe")
  end
end
