# frozen_string_literal: true

require_relative "../test_helper"

class OnnxRegistryCoverageTest < Minitest::Test
  def test_every_registered_model_has_onnx_wrapper_test
    expected = MlxLm::Models::REGISTRY.keys.map { |model_type| "#{normalized_model_basename(model_type)}_test.rb" }.sort
    actual = Dir.glob(File.join(__dir__, "*_test.rb")).map { |path| File.basename(path) }

    missing = expected - actual
    assert_empty missing, "Missing ONNX wrapper tests for: #{missing.join(', ')}"
  end

  private

  def normalized_model_basename(model_type)
    model_type.downcase.gsub(/[^a-z0-9]+/, "_").gsub(/_+/, "_").gsub(/^_|_$/, "")
  end
end
