# frozen_string_literal: true

require_relative "../test_helper"

class SsmGatedDeltaBitlinearPipelineIntegrationSmokeTest < Minitest::Test
  def test_new_phase15_modules_load
    assert defined?(MlxLm::Models::SSM)
    assert defined?(MlxLm::Models::GatedDelta)
    assert defined?(MlxLm::Models::Activations)
    assert defined?(MlxLm::Models::PipelineMixin)
    assert defined?(MlxLm::Models::BitLinear)
  end

  def test_basic_construction_paths
    mx = MLX::Core

    xielu = MlxLm::Models::Activations::XieLU.new
    x = mx.array([-1.0, 0.0, 1.0], dtype: mx.float32)
    y = xielu.call(x)
    assert_equal [3], y.shape

    bitlinear = MlxLm::Models::BitLinear.new(4, 3)
    out = bitlinear.call(mx.random_uniform([2, 4], -0.5, 0.5, mx.float32))
    assert_equal [2, 3], out.shape
  end
end
