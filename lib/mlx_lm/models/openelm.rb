module MlxLm
  module Models
    module OpenELM
      module_function

      def make_divisible(v, divisor = 8, min_value = nil)
        min_value ||= divisor
        rounded = ((v + (divisor.to_f / 2)).to_i / divisor) * divisor
        new_v = [min_value, rounded].max
        new_v += divisor if new_v < (0.9 * v)
        new_v
      end

      class ModelArgs < BaseModelArgs
        field :model_type, default: "openelm"
        field :head_dim, default: 64
        field :num_transformer_layers, default: 12
        field :model_dim, default: 2048
        field :vocab_size, default: 32_000
        field :ffn_dim_divisor, default: 8
        field :num_query_heads, default: [32]
        field :num_kv_heads, default: []
        field :ffn_multipliers, default: [1.0]
        field :ffn_with_glu, default: true
        field :normalize_qk_projections, default: true
        field :share_input_output_layers, default: true
        field :rms_norm_eps, default: 1e-6
        field :rope_freq_constant, default: 10_000.0

        def initialize(**kwargs)
          super
          @num_query_heads = normalize_schedule(@num_query_heads, @num_transformer_layers, 1, "num_query_heads").map(&:to_i)

          if @num_kv_heads.nil? || Array(@num_kv_heads).empty?
            @num_kv_heads = @num_query_heads.dup
          else
            @num_kv_heads = normalize_schedule(@num_kv_heads, @num_transformer_layers, @num_query_heads[0], "num_kv_heads").map(&:to_i)
          end

          @ffn_multipliers = normalize_schedule(@ffn_multipliers, @num_transformer_layers, 1.0, "ffn_multipliers").map(&:to_f)
        end

        private

        def normalize_schedule(values, layers, fallback, field_name)
          items = Array(values)
          items = [fallback] if items.empty?
          items = Array.new(layers, items[0]) if items.length == 1 && layers > 1

          unless items.length == layers
            raise ArgumentError, "#{field_name} must have #{layers} entries, got #{items.length}"
          end

          items
        end
      end

      class Attention < MLX::NN::Module
        def initialize(args, layer_id:)
          super()
          @head_dim = args.head_dim
          @n_heads = args.num_query_heads[layer_id]
          @n_kv_heads = args.num_kv_heads[layer_id]
          @scale = @head_dim**(-0.5)
          @normalize_qk_projections = args.normalize_qk_projections

          op_size = (@n_heads + (2 * @n_kv_heads)) * @head_dim
          self.qkv_proj = MLX::NN::Linear.new(args.model_dim, op_size, bias: false)
          self.out_proj = MLX::NN::Linear.new(@n_heads * @head_dim, args.model_dim, bias: false)

          if @normalize_qk_projections
            self.q_norm = MLX::NN::RMSNorm.new(@head_dim, eps: args.rms_norm_eps)
            self.k_norm = MLX::NN::RMSNorm.new(@head_dim, eps: args.rms_norm_eps)
          end

          self.rope = MLX::NN::RoPE.new(@head_dim, traditional: false, base: args.rope_freq_constant)
        end

        def call(x, mask: nil, cache: nil)
          mx = MLX::Core
          b, l, _d = x.shape

          qkv = qkv_proj.call(x)
          qkv = qkv.reshape([b, l, @n_heads + (2 * @n_kv_heads), @head_dim]).transpose([0, 2, 1, 3])
          queries, keys, values = mx.split(qkv, [@n_heads, @n_heads + @n_kv_heads], 1)

          if @normalize_qk_projections
            queries = q_norm.call(queries)
            keys = k_norm.call(keys)
          end

          if cache
            queries = rope.call(queries, offset: cache.offset)
            keys = rope.call(keys, offset: cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
          else
            queries = rope.call(queries)
            keys = rope.call(keys)
          end

          output = mx.scaled_dot_product_attention(queries, keys, values, @scale, mask)
          output = output.transpose([0, 2, 1, 3]).reshape([b, l, @n_heads * @head_dim])
          out_proj.call(output)
        end
      end

      class MLP < MLX::NN::Module
        def initialize(args, layer_id:)
          super()
          @ffn_with_glu = args.ffn_with_glu
          dim = args.model_dim
          multiplier = args.ffn_multipliers[layer_id]
          @intermediate_dim = OpenELM.make_divisible(multiplier * dim, args.ffn_dim_divisor).to_i

          proj_1_dim = @ffn_with_glu ? (2 * @intermediate_dim) : @intermediate_dim
          self.proj_1 = MLX::NN::Linear.new(dim, proj_1_dim, bias: false)
          self.proj_2 = MLX::NN::Linear.new(@intermediate_dim, dim, bias: false)
        end

        def call(x)
          x = proj_1.call(x)
          x = if @ffn_with_glu
            gate, value = MLX::Core.split(x, [@intermediate_dim], -1)
            Activations.swiglu(gate, value)
          else
            MLX::NN.gelu_approx(x)
          end
          proj_2.call(x)
        end
      end

      class TransformerBlock < MLX::NN::Module
        def initialize(args, layer_id:)
          super()
          self.attn = Attention.new(args, layer_id: layer_id)
          self.ffn = MLP.new(args, layer_id: layer_id)
          self.attn_norm = MLX::NN::RMSNorm.new(args.model_dim, eps: args.rms_norm_eps)
          self.ffn_norm = MLX::NN::RMSNorm.new(args.model_dim, eps: args.rms_norm_eps)
        end

        def call(x, mask: nil, cache: nil)
          r = attn.call(attn_norm.call(x), mask: mask, cache: cache)
          h = x + r
          r = ffn.call(ffn_norm.call(h))
          h + r
        end
      end

      class OpenELMModel < MLX::NN::Module
        def initialize(args)
          super()
          self.token_embeddings = MLX::NN::Embedding.new(args.vocab_size, args.model_dim)
          self.layers = Array.new(args.num_transformer_layers) do |layer_id|
            TransformerBlock.new(args, layer_id: layer_id)
          end
          self.norm = MLX::NN::RMSNorm.new(args.model_dim, eps: args.rms_norm_eps)
        end

        def call(inputs, cache: nil)
          h = token_embeddings.call(inputs)
          layer_cache = cache || [nil] * layers.length
          mask = _create_attention_mask(h, layer_cache[0])

          layers.each_with_index do |layer, i|
            h = layer.call(h, mask: mask, cache: layer_cache[i])
          end
          norm.call(h)
        end

        private

        def _create_attention_mask(h, cache)
          return cache.make_mask(h.shape[1]) if cache && cache.respond_to?(:make_mask)
          return nil if h.shape[1] == 1

          "causal"
        end
      end

      class Model < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.model_type = args.model_type
          self.transformer = OpenELMModel.new(args)
          unless args.share_input_output_layers
            self.lm_head = MLX::NN::Linear.new(args.model_dim, args.vocab_size, bias: false)
          end
        end

        def call(inputs, cache: nil)
          out = transformer.call(inputs, cache: cache)
          if @args.share_input_output_layers
            transformer.token_embeddings.as_linear(out)
          else
            lm_head.call(out)
          end
        end

        def layers
          transformer.layers
        end
      end

      Models.register("openelm", Model, ModelArgs)
    end
  end
end
