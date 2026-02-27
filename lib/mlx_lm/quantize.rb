module MlxLm
  module Quantize
    module_function

    # Quantize a model's linear and embedding layers.
    #
    # @param model [nn::Module] The model to quantize
    # @param group_size [Integer] Group size for quantization (default: 64)
    # @param bits [Integer] Number of bits (default: 4)
    # @param weights [Hash] Optional weight dict; if provided, only quantize layers
    #   that have corresponding .scales keys
    # @return [Hash] The quantization config
    def quantize_model(model, group_size: 64, bits: 4, weights: nil)
      if weights
        # Auto-detect: only quantize layers that have .scales in weights
        class_predicate = ->(path, mod) {
          if mod.respond_to?(:to_quantized)
            weights.key?("#{path}.scales")
          else
            false
          end
        }
      else
        # Quantize all quantizable layers
        class_predicate = ->(_path, mod) {
          mod.respond_to?(:to_quantized)
        }
      end

      MLX::NN.quantize(model, group_size: group_size, bits: bits, class_predicate: class_predicate)

      { "group_size" => group_size, "bits" => bits }
    end

    # Dequantize a model (convert QuantizedLinear back to Linear, etc.)
    #
    # @param model [nn::Module] The quantized model to dequantize
    # @return [nn::Module] The dequantized model
    def dequantize_model(model)
      de_quantize_layers(model)
      model
    end

    # Compute bits per weight for a model
    def bits_per_weight(model)
      total_bits = 0
      total_params = 0

      model.named_modules.each do |name, mod|
        case mod
        when MLX::NN::QuantizedLinear
          # Quantized: bits per element = quantized bits
          weight = mod.instance_variable_get(:@weight)
          if weight
            num_params = weight.shape.reduce(:*)
            total_bits += num_params * 4 # approximate
            total_params += num_params
          end
        when MLX::NN::Linear
          weight = mod.instance_variable_get(:@weight)
          if weight
            num_params = weight.shape.reduce(:*)
            total_bits += num_params * 32 # float32
            total_params += num_params
          end
        end
      end

      total_params > 0 ? total_bits.to_f / total_params : 0.0
    end

    private

    def self.de_quantize_layers(mod)
      mod.instance_variables.each do |ivar|
        val = mod.instance_variable_get(ivar)
        case val
        when MLX::NN::QuantizedLinear
          # Convert back to Linear
          dequantized = linear_from_quantized(val)
          mod.instance_variable_set(ivar, dequantized)
        when MLX::NN::QuantizedEmbedding
          dequantized = embedding_from_quantized(val)
          mod.instance_variable_set(ivar, dequantized)
        when MLX::NN::Module
          de_quantize_layers(val)
        when ::Array
          val.each { |item| de_quantize_layers(item) if item.is_a?(MLX::NN::Module) }
        end
      end
    end

    def self.linear_from_quantized(qlinear)
      mx = MLX::Core
      weight = qlinear.instance_variable_get(:@weight)
      scales = qlinear.instance_variable_get(:@scales)
      biases = qlinear.instance_variable_get(:@biases)
      bias = qlinear.instance_variable_get(:@bias)
      group_size = qlinear.instance_variable_get(:@group_size) || 64
      bits = qlinear.instance_variable_get(:@bits) || 4

      # Dequantize weight
      dequantized = mx.dequantize(weight, scales, biases, group_size, bits)
      out_features = dequantized.shape[0]
      in_features = dequantized.shape[1]

      linear = MLX::NN::Linear.new(in_features, out_features, bias: !bias.nil?)
      linear.instance_variable_set(:@weight, dequantized)
      linear.instance_variable_set(:@bias, bias) if bias
      linear
    end

    def self.embedding_from_quantized(qembed)
      mx = MLX::Core
      weight = qembed.instance_variable_get(:@weight)
      scales = qembed.instance_variable_get(:@scales)
      biases = qembed.instance_variable_get(:@biases)
      group_size = qembed.instance_variable_get(:@group_size) || 64
      bits = qembed.instance_variable_get(:@bits) || 4

      dequantized = mx.dequantize(weight, scales, biases, group_size, bits)
      num_embeddings = dequantized.shape[0]
      dims = dequantized.shape[1]

      embedding = MLX::NN::Embedding.new(num_embeddings, dims)
      embedding.instance_variable_set(:@weight, dequantized)
      embedding
    end
  end
end
