require_relative "../test_helper"
require_relative "../../lib/mlx_lm/models/pipeline"

class Phase15SwitchPipelineTest < Minitest::Test
  include ParityTestHelpers

  class FakeGroup
    def initialize(rank, size)
      @rank = rank
      @size = size
    end

    def rank
      @rank
    end

    def size
      @size
    end
  end

  class DummyPipelineModel < MLX::NN::Module
    include MlxLm::Models::PipelineMixin
  end

  def setup
    @mx = MLX::Core
  end

  def test_switch_linear_to_quantized_and_forward
    @mx.random_seed(7)
    linear = MlxLm::Models::SwitchLayers::SwitchLinear.new(32, 16, 4, bias: true)
    qlinear = linear.to_quantized(group_size: 32, bits: 4, mode: "affine")

    assert_instance_of MlxLm::Models::SwitchLayers::QuantizedSwitchLinear, qlinear
    assert_equal 32, qlinear.group_size
    assert_equal 4, qlinear.bits
    assert_equal "affine", qlinear.mode

    x = @mx.random_uniform([2, 3, 1, 1, 32], -0.5, 0.5, @mx.float32)
    indices = @mx.array(
      [
        [[0, 1], [1, 2], [2, 3]],
        [[3, 2], [2, 1], [1, 0]]
      ],
      dtype: @mx.int32
    )

    expected = linear.call(x, indices)
    actual = qlinear.call(x, indices)
    @mx.eval(expected, actual)

    assert_equal expected.shape, actual.shape
    assert_arrays_close(
      expected.tolist,
      actual.tolist,
      atol: 0.35,
      msg: "QuantizedSwitchLinear output should closely track SwitchLinear"
    )
  end

  def test_switch_linear_to_quantized_rejects_quantized_input
    linear = MlxLm::Models::SwitchLayers::SwitchLinear.new(8, 4, 2)
    assert_raises(ArgumentError) do
      linear.to_quantized(quantize_input: true)
    end
  end

  def test_switch_mlp_forward_shape_with_sorted_route_path
    @mx.random_seed(123)
    mlp = MlxLm::Models::SwitchLayers::SwitchMLP.new(16, 32, 8)
    x = @mx.random_uniform([4, 8, 16], -1.0, 1.0, @mx.float32)

    route_data = Array.new(4) do
      Array.new(8) do
        [rand(8), rand(8)]
      end
    end
    indices = @mx.array(route_data, dtype: @mx.int32) # 4*8*2 = 64 routes -> sorted path

    y = mlp.call(x, indices)
    @mx.eval(y)
    assert_equal [4, 8, 2, 16], y.shape
  end

  def test_pipeline_mixin_defaults_and_partitioning
    base = DummyPipelineModel.new
    base.layers = (0...6).to_a

    assert_equal 0, base.pipeline_rank
    assert_equal 1, base.pipeline_size
    assert_equal 0, base.start_idx
    assert_nil base.end_idx
    assert_equal [0, 1, 2, 3, 4, 5], base.pipeline_layers

    rank1 = DummyPipelineModel.new
    rank1.layers = (0...10).to_a
    returned = rank1.pipeline(FakeGroup.new(1, 3))

    assert_same rank1, returned
    assert_equal 1, rank1.pipeline_rank
    assert_equal 3, rank1.pipeline_size
    assert_equal 3, rank1.start_idx
    assert_equal 6, rank1.end_idx
    assert_equal [nil, nil, nil, 3, 4, 5], rank1.layers
    assert_equal [3, 4, 5], rank1.pipeline_layers

    rank0 = DummyPipelineModel.new
    rank0.layers = (0...10).to_a
    rank0.pipeline(FakeGroup.new(0, 3))
    assert_equal 8, rank0.start_idx
    assert_equal 12, rank0.end_idx
    assert_equal [nil, nil, nil, nil, nil, nil, nil, nil, 8, 9], rank0.layers
    assert_equal [8, 9], rank0.pipeline_layers
  end
end
