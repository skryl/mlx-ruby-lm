module MlxLm
  module Tuner
    # LoRA adapter for Linear layers.
    # Forward: y = linear(x) + scale * (dropout(x) @ lora_a @ lora_b)
    class LoRALinear < MLX::NN::Module
      def self.from_base(linear, r: 8, dropout: 0.0, scale: 20.0)
        if linear.is_a?(MLX::NN::QuantizedLinear)
          input_dims = linear.instance_variable_get(:@weight).shape[1] * 32 /
                       (linear.instance_variable_get(:@bits) || 4)
          output_dims = linear.instance_variable_get(:@weight).shape[0]
          bias = !linear.instance_variable_get(:@bias).nil?
        else
          weight = linear.weight
          output_dims, input_dims = weight.shape
          bias = !linear.respond_to?(:bias) || !linear.bias.nil? rescue false
        end
        lora = new(input_dims, output_dims, r: r, dropout: dropout, scale: scale, bias: bias)
        lora.linear = linear
        lora
      end

      def initialize(input_dims, output_dims, r: 8, dropout: 0.0, scale: 20.0, bias: false)
        super()
        mx = MLX::Core
        @scale = scale
        self.linear = MLX::NN::Linear.new(input_dims, output_dims, bias: bias)
        self.dropout = MLX::NN::Dropout.new(dropout)

        # Initialize LoRA matrices
        lora_scale = 1.0 / Math.sqrt(input_dims)
        self.lora_a = mx.random_uniform(
          [input_dims, r], -lora_scale, lora_scale, mx.float32
        )
        self.lora_b = mx.zeros([r, output_dims])
      end

      def call(x)
        mx = MLX::Core
        y = linear.call(x)
        z = dropout.call(x)
        z = mx.matmul(mx.matmul(z, lora_a), lora_b)
        y + z * @scale
      end

      def fuse(dequantize: false)
        mx = MLX::Core
        lin = linear

        if dequantize && lin.is_a?(MLX::NN::QuantizedLinear)
          lin = MlxLm::Quantize.linear_from_quantized(lin)
        end

        weight = lin.weight
        bias_val = lin.respond_to?(:bias) ? lin.bias : nil

        # Fuse: W' = W + scale * (lora_a @ lora_b)^T
        lora_weight = mx.matmul(lora_a, lora_b)
        fused_weight = weight + mx.transpose(lora_weight) * @scale

        out_features, in_features = fused_weight.shape
        result = MLX::NN::Linear.new(in_features, out_features, bias: !bias_val.nil?)
        result.weight = fused_weight
        result.bias = bias_val if bias_val
        result
      end
    end

    # Compatibility wrapper for SwitchLinear LoRA adaptation paths.
    class LoRASwitchLinear < LoRALinear
      def call(x, indices = nil, **kwargs)
        mx = MLX::Core
        y = if indices.nil?
          linear.call(x)
        else
          linear.call(x, indices, **kwargs)
        end

        z = dropout.call(x)
        z = mx.matmul(mx.matmul(z, lora_a), lora_b)
        y + z * @scale
      end
    end

    # LoRA adapter for Embedding layers.
    class LoRAEmbedding < MLX::NN::Module
      def self.from_base(embedding, r: 8, dropout: 0.0, scale: 20.0)
        weight = embedding.weight
        num_embeddings, dims = weight.shape
        lora = new(num_embeddings, dims, r: r, dropout: dropout, scale: scale)
        lora.embedding = embedding
        lora
      end

      def initialize(num_embeddings, dims, r: 8, dropout: 0.0, scale: 20.0)
        super()
        mx = MLX::Core
        @scale = scale
        self.embedding = MLX::NN::Embedding.new(num_embeddings, dims)
        self.dropout = MLX::NN::Dropout.new(dropout)

        lora_scale = 1.0 / Math.sqrt(num_embeddings)
        self.lora_a = mx.random_uniform(
          [num_embeddings, r], -lora_scale, lora_scale, mx.float32
        )
        self.lora_b = mx.zeros([r, dims])
      end

      def call(x)
        mx = MLX::Core
        y = embedding.call(x)
        # LoRA for embedding: look up lora_a rows, then multiply by lora_b
        z = mx.matmul(mx.take(lora_a, x, 0), lora_b)
        z = dropout.call(z)
        y + z * @scale
      end

      def as_linear(x)
        mx = MLX::Core
        y = embedding.as_linear(x)
        z = mx.matmul(mx.matmul(dropout.call(x), mx.transpose(lora_b)), mx.transpose(lora_a))
        y + z * @scale
      end

      def fuse(dequantize: false)
        mx = MLX::Core
        embed = embedding

        if dequantize && embed.is_a?(MLX::NN::QuantizedEmbedding)
          embed = MlxLm::Quantize.embedding_from_quantized(embed)
        end

        weight = embed.weight
        lora_weight = mx.matmul(lora_a, lora_b)
        fused_weight = weight + lora_weight * @scale

        num_embeddings, dims = fused_weight.shape
        result = MLX::NN::Embedding.new(num_embeddings, dims)
        result.weight = fused_weight
        result
      end
    end

    module_function

    # Default LoRA target keys (layer names that get LoRA applied)
    DEFAULT_LORA_KEYS = %w[self_attn.q_proj self_attn.k_proj self_attn.v_proj].freeze

    # Apply LoRA layers to a model's last N layers.
    def apply_lora_layers(model, num_layers: nil, config: {})
      rank = config["rank"] || config[:rank] || 8
      scale = config["scale"] || config[:scale] || 20.0
      dropout = config["dropout"] || config[:dropout] || 0.0
      keys = config["keys"] || config[:keys] || DEFAULT_LORA_KEYS

      layers = model.layers
      num_layers ||= layers.length
      target_layers = layers.last(num_layers)

      target_layers.each do |layer|
        _apply_lora_to_module(layer, "", keys, rank: rank, scale: scale, dropout: dropout)
      end
    end

    def _apply_lora_to_module(mod, prefix, keys, rank:, scale:, dropout:)
      mod.state.each do |key, value|
        full_key = prefix.empty? ? key : "#{prefix}.#{key}"

        if value.is_a?(MLX::NN::Linear) && keys.any? { |k| full_key.end_with?(k) || full_key.include?(k) }
          lora = LoRALinear.from_base(value, r: rank, scale: scale, dropout: dropout)
          mod.state[key] = lora
        elsif value.is_a?(MLX::NN::Embedding) && keys.any? { |k| full_key.end_with?(k) || full_key.include?(k) }
          lora = LoRAEmbedding.from_base(value, r: rank, scale: scale, dropout: dropout)
          mod.state[key] = lora
        elsif value.is_a?(MLX::NN::Module)
          _apply_lora_to_module(value, full_key, keys, rank: rank, scale: scale, dropout: dropout)
        end
      end
    end
    module_function :_apply_lora_to_module
  end
end
