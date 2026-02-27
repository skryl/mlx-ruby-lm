require_relative "../test_helper"
require_relative "../../lib/mlx_lm/models/bitlinear_layers"

class BitlinearQuantizationBlock < MLX::NN::Module
  def initialize
    super()
    self.keep = MLX::NN::Linear.new(4, 6, bias: true)
    self.skip = MLX::NN::Linear.new(6, 4, bias: false)
  end
end

class BitlinearQuantizationModel < MLX::NN::Module
  def initialize
    super()
    self.block = BitlinearQuantizationBlock.new
    self.head = MLX::NN::Linear.new(4, 3, bias: false)
  end
end

class BitlinearQuantizationTest < Minitest::Test
  include ParityTestHelpers

  def setup
    @mx = MLX::Core
  end

  def test_bitlinear_forward_fallback_matches_dense_reference
    layer = MlxLm::Models::BitLinear.new(3, 5, bias: true, invert_weight_scales: false)

    full_weights = [
      [1, 0, -1],
      [-1, 1, 0],
      [0, 1, 1],
      [1, -1, 0],
      [-1, 0, 1],
    ]
    layer.weight = @mx.array(pack_ternary_weights(full_weights), dtype: @mx.uint8)
    layer.weight_scale = @mx.array([2.0], dtype: @mx.float32)
    layer.bias = @mx.array([0.25, -0.5, 0.0, 1.0, -1.5], dtype: @mx.float32)

    x = @mx.array(
      [
        [[1.0, 2.0, -1.0], [0.5, 0.0, 1.0]],
        [[-2.0, 1.0, 3.0], [1.5, -0.5, 0.25]],
      ],
      dtype: @mx.float32
    )

    dense_weight = @mx.array(full_weights, dtype: @mx.float32)
    expected = @mx.add(@mx.multiply(@mx.matmul(x, dense_weight.T), 2.0), layer.bias)
    actual = layer.call(x)

    @mx.eval(expected, actual)
    assert_equal [2, 2, 5], actual.shape
    assert_equal expected.shape, actual.shape
    assert_arrays_close(expected.tolist, actual.tolist, atol: 1e-6, msg: "bitlinear fallback output mismatch")
  end

  def test_bitlinear_inverted_weight_scale_matches_reference
    layer = MlxLm::Models::BitLinear.new(2, 4, bias: false, invert_weight_scales: true)

    full_weights = [
      [1, -1],
      [0, 1],
      [-1, 0],
      [1, 1],
    ]
    layer.weight = @mx.array(pack_ternary_weights(full_weights), dtype: @mx.uint8)
    layer.weight_scale = @mx.array([4.0], dtype: @mx.float32)

    x = @mx.array(
      [
        [2.0, -1.0],
        [0.5, 0.25],
        [-3.0, 1.0],
      ],
      dtype: @mx.float32
    )

    dense_weight = @mx.array(full_weights, dtype: @mx.float32)
    expected = @mx.multiply(@mx.matmul(x, dense_weight.T), 0.25)
    actual = layer.call(x)

    @mx.eval(expected, actual)
    assert_equal expected.shape, actual.shape
    assert_arrays_close(expected.tolist, actual.tolist, atol: 1e-6, msg: "inverted scale mismatch")
  end

  def test_bitnet_quantize_replaces_linear_layers_with_skip_list_support
    model = BitlinearQuantizationModel.new
    converted = MlxLm::Models.bitnet_quantize(
      model,
      {
        modules_to_not_convert: ["block.skip"],
        linear_class: "autobitlinear",
      }
    )

    assert_same model, converted

    assert_instance_of MlxLm::Models::BitLinear, model.block.keep
    assert_instance_of MLX::NN::Linear, model.block.skip
    assert_instance_of MlxLm::Models::BitLinear, model.head

    assert_equal false, model.block.keep.invert_weight_scales
    assert_equal false, model.head.invert_weight_scales

    assert model.block.keep.state.key?("bias")
    assert_equal [2, 4], model.block.keep.weight.shape
    refute model.head.state.key?("bias")
  end

  def test_bitnet_quantize_defaults_to_inverted_weight_scales
    model = BitlinearQuantizationModel.new
    MlxLm::Models.bitnet_quantize(model, {})

    assert_instance_of MlxLm::Models::BitLinear, model.block.keep
    assert_equal true, model.block.keep.invert_weight_scales
  end

  private

  def pack_ternary_weights(full_weights)
    out_features = full_weights.length
    in_features = full_weights.first.length
    packed_out_features = (out_features + 3) / 4

    packed = Array.new(packed_out_features) { Array.new(in_features, 0) }
    packed_out_features.times do |packed_row|
      in_features.times do |input_col|
        byte = 0
        4.times do |group_idx|
          out_row = packed_row + (group_idx * packed_out_features)
          ternary = out_row < out_features ? full_weights[out_row][input_col] : 0
          encoded = encode_ternary(ternary)
          byte |= (encoded << (group_idx * 2))
        end
        packed[packed_row][input_col] = byte
      end
    end
    packed
  end

  def encode_ternary(value)
    case value
    when -1
      0
    when 0
      1
    when 1
      2
    else
      raise ArgumentError, "Expected ternary value in {-1, 0, 1}, got #{value.inspect}"
    end
  end
end
