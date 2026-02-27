module MlxLm
  module Tuner
    class DoRALinear < MLX::NN::Module
      attr_accessor :linear, :dropout, :lora_a, :lora_b, :scale, :m

      def self.from_base(linear, r: 8, dropout: 0.0, scale: 20.0)
        weight = dequantized_weight_from(linear)
        output_dims, input_dims = weight.shape
        dora = new(
          input_dims,
          output_dims,
          r: r,
          dropout: dropout,
          scale: scale,
          bias: has_bias?(linear)
        )
        dora.set_linear(linear)
        dora
      end

      def initialize(input_dims, output_dims, r: 8, dropout: 0.0, scale: 20.0, bias: false)
        super()
        mx = MLX::Core
        self.scale = scale
        set_linear(MLX::NN::Linear.new(input_dims, output_dims, bias: bias))
        self.dropout = MLX::NN::Dropout.new(dropout)

        init_scale = 1.0 / Math.sqrt(input_dims)
        self.lora_a = mx.random_uniform([input_dims, r], -init_scale, init_scale, mx.float32)
        self.lora_b = mx.zeros([r, output_dims])
      end

      def set_linear(linear_layer)
        self.linear = linear_layer
        self.m = row_norm(_dequantized_weight.astype(MLX::Core.float32))
      end

      def fuse(dequantize: false)
        mx = MLX::Core
        weight = _dequantized_weight
        adapted = adapted_weight(weight)
        norm_scale = m / row_norm(adapted.astype(mx.float32))
        fused_weight = adapted * norm_scale.reshape([norm_scale.shape[0], 1])

        out_features, in_features = fused_weight.shape
        fused = MLX::NN::Linear.new(in_features, out_features, bias: self.class.has_bias?(linear))
        fused.weight = fused_weight
        if self.class.has_bias?(linear) && fused.respond_to?(:bias=)
          fused.bias = linear_bias(linear)
        end

        if quantized_linear?(linear) && !dequantize
          group_size = linear.instance_variable_get(:@group_size) || 64
          bits = linear.instance_variable_get(:@bits) || 4
          fused = fused.to_quantized(group_size: group_size, bits: bits)
        end
        fused
      end

      def call(x)
        mx = MLX::Core
        weight = _dequantized_weight
        y = mx.matmul(x, mx.transpose(weight))
        z = mx.matmul(mx.matmul(dropout.call(x), lora_a), lora_b)
        out = y + (z * scale).astype(y.dtype)

        denom = mx.stop_gradient(row_norm(adapted_weight(weight).astype(mx.float32)))
        out = out * (m / denom)
        bias = linear_bias(linear)
        out = out + bias if bias
        out
      end

      private

      def adapted_weight(weight)
        mx = MLX::Core
        lora = mx.matmul(mx.transpose(lora_b * scale), mx.transpose(lora_a))
        weight + lora.astype(weight.dtype)
      end

      def linear_bias(layer)
        return layer.bias if layer.respond_to?(:bias)

        layer.instance_variable_get(:@bias)
      rescue StandardError
        nil
      end

      def _dequantized_weight
        self.class.dequantized_weight_from(linear)
      end

      def row_norm(weight)
        mx = MLX::Core
        mx.sqrt(mx.maximum(mx.sum(weight * weight, 1), 1e-12))
      end

      def quantized_linear?(layer)
        layer.class.name.to_s.end_with?("QuantizedLinear")
      end

      class << self
        def has_bias?(layer)
          if layer.respond_to?(:bias)
            !layer.bias.nil?
          else
            !layer.instance_variable_get(:@bias).nil?
          end
        rescue StandardError
          false
        end

        def dequantized_weight_from(layer)
          return layer.weight unless layer.class.name.to_s.end_with?("QuantizedLinear")

          mx = MLX::Core
          weight = layer.instance_variable_get(:@weight)
          scales = layer.instance_variable_get(:@scales)
          biases = layer.instance_variable_get(:@biases)
          group_size = layer.instance_variable_get(:@group_size) || 64
          bits = layer.instance_variable_get(:@bits) || 4
          mx.dequantize(weight, scales, biases, group_size, bits)
        end
      end
    end

    class DoRAEmbedding < MLX::NN::Module
      attr_accessor :embedding, :dropout, :lora_a, :lora_b, :scale, :m

      def self.from_base(embedding, r: 8, dropout: 0.0, scale: 20.0)
        num_embeddings, dims = embedding.weight.shape
        dora = new(num_embeddings, dims, r: r, dropout: dropout, scale: scale)
        dora.set_embedding(embedding)
        dora
      end

      def initialize(num_embeddings, dims, r: 8, dropout: 0.0, scale: 20.0)
        super()
        mx = MLX::Core
        self.scale = scale
        set_embedding(MLX::NN::Embedding.new(num_embeddings, dims))
        self.dropout = MLX::NN::Dropout.new(dropout)

        init_scale = 1.0 / Math.sqrt(num_embeddings)
        self.lora_a = mx.random_uniform([num_embeddings, r], -init_scale, init_scale, mx.float32)
        self.lora_b = mx.zeros([r, dims])
      end

      def set_embedding(embedding_layer)
        self.embedding = embedding_layer
        self.m = row_norm(embedding.weight.astype(MLX::Core.float32))
      end

      def fuse(_dequantize: false)
        mx = MLX::Core
        weight = embedding.weight
        adapted = weight + mx.matmul(lora_a * scale, lora_b).astype(weight.dtype)
        norm_scale = m / row_norm(adapted.astype(mx.float32))
        fused_weight = adapted * norm_scale.reshape([norm_scale.shape[0], 1])

        rows, cols = fused_weight.shape
        fused = MLX::NN::Embedding.new(rows, cols)
        fused.weight = fused_weight
        fused
      end

      def call(x)
        mx = MLX::Core
        y = embedding.call(x)
        z = mx.matmul(mx.take(lora_a, x, 0), lora_b) * scale
        out = y + dropout.call(z).astype(y.dtype)

        adapted = y + z
        denom = mx.stop_gradient(token_norm(adapted.astype(mx.float32)))
        m_rows = mx.take(m, x, 0)
        out * (m_rows / denom).reshape(m_rows.shape + [1])
      end

      def as_linear(x)
        mx = MLX::Core
        y = embedding.as_linear(x)
        z = mx.matmul(mx.matmul(dropout.call(x), mx.transpose(lora_b)), mx.transpose(lora_a))
        out = y + (z * scale).astype(y.dtype)

        adapted = embedding.weight + mx.matmul(lora_a * scale, lora_b)
        denom = mx.stop_gradient(row_norm(adapted.astype(mx.float32)))
        out * (m / denom)
      end

      private

      def row_norm(weight)
        mx = MLX::Core
        mx.sqrt(mx.maximum(mx.sum(weight * weight, 1), 1e-12))
      end

      def token_norm(values)
        mx = MLX::Core
        mx.sqrt(mx.maximum(mx.sum(values * values, -1), 1e-12))
      end
    end
  end
end
