module MlxLm
  module Models
    module Gemma2
      class ModelArgs < BaseModelArgs
        field :model_type, default: "gemma2"
        field :hidden_size, default: 3072
        field :num_hidden_layers, default: 28
        field :num_attention_heads, default: 16
        field :num_key_value_heads, default: 16
        field :intermediate_size, default: 24576
        field :vocab_size, default: 256000
        field :head_dim, default: 256
        field :rms_norm_eps, default: 1e-6
        field :rope_theta, default: 10000.0
        field :rope_traditional, default: false
        field :attn_logit_softcapping, default: 50.0
        field :final_logit_softcapping, default: 30.0
        field :query_pre_attn_scalar, default: 144.0

        def initialize(**kwargs)
          super
          @num_key_value_heads ||= @num_attention_heads
        end
      end

      # Gemma2 custom RMSNorm: uses (1 + weight) instead of weight
      class Gemma2RMSNorm < MLX::NN::Module
        def initialize(dims, eps: 1e-6)
          super()
          self.weight = MLX::Core.ones([dims])
          @eps = eps
        end

        def call(x)
          mx = MLX::Core
          # RMS normalization: x / sqrt(mean(x^2) + eps) * (1 + weight)
          x_sq = x * x
          mean_sq = mx.mean(x_sq, -1, keepdims: true)
          norm = x * mx.rsqrt(mean_sq + @eps)
          norm * (weight + 1.0)
        end
      end

      class Attention < MLX::NN::Module
        def initialize(args)
          super()
          dim = args.hidden_size
          @n_heads = args.num_attention_heads
          @n_kv_heads = args.num_key_value_heads
          @head_dim = args.head_dim
          @scale = 1.0 / (args.query_pre_attn_scalar**0.5)
          @attn_logit_softcapping = args.attn_logit_softcapping

          self.q_proj = MLX::NN::Linear.new(dim, @n_heads * @head_dim, bias: false)
          self.k_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: false)
          self.v_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: false)
          self.o_proj = MLX::NN::Linear.new(@n_heads * @head_dim, dim, bias: false)

          self.rope = MLX::NN::RoPE.new(
            @head_dim,
            traditional: args.rope_traditional,
            base: args.rope_theta
          )
        end

        def call(x, mask: nil, cache: nil)
          mx = MLX::Core
          b, l, _d = x.shape

          queries = q_proj.call(x).reshape([b, l, @n_heads, @head_dim]).transpose([0, 2, 1, 3])
          keys = k_proj.call(x).reshape([b, l, @n_kv_heads, @head_dim]).transpose([0, 2, 1, 3])
          values = v_proj.call(x).reshape([b, l, @n_kv_heads, @head_dim]).transpose([0, 2, 1, 3])

          if cache
            queries = rope.call(queries, offset: cache.offset)
            keys = rope.call(keys, offset: cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
          else
            queries = rope.call(queries)
            keys = rope.call(keys)
          end

          # Custom attention with softcapping
          queries = queries * @scale

          # Manual attention computation for softcapping
          scores = mx.matmul(queries, mx.transpose(keys, [0, 1, 3, 2]))

          # Apply attention logit softcapping
          scores = mx.tanh(scores / @attn_logit_softcapping) * @attn_logit_softcapping

          # Apply causal mask
          if mask == "causal"
            n = scores.shape[-1]
            causal_mask = mx.triu(mx.full([n, n], -Float::INFINITY), 1)
            scores = scores + causal_mask
          end

          scores = mx.softmax(scores, -1)
          output = mx.matmul(scores, values)

          output = output.transpose([0, 2, 1, 3]).reshape([b, l, @n_heads * @head_dim])
          o_proj.call(output)
        end
      end

      class MLP < MLX::NN::Module
        def initialize(dim, hidden_dim)
          super()
          self.gate_proj = MLX::NN::Linear.new(dim, hidden_dim, bias: false)
          self.down_proj = MLX::NN::Linear.new(hidden_dim, dim, bias: false)
          self.up_proj = MLX::NN::Linear.new(dim, hidden_dim, bias: false)
        end

        def call(x)
          # Gemma2 uses gelu_approx instead of silu
          down_proj.call(MLX::NN.gelu_approx(gate_proj.call(x)) * up_proj.call(x))
        end
      end

      class TransformerBlock < MLX::NN::Module
        def initialize(args)
          super()
          self.self_attn = Attention.new(args)
          self.mlp = MLP.new(args.hidden_size, args.intermediate_size)
          # Gemma2 has 4 norms per block
          self.input_layernorm = Gemma2RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
          self.post_attention_layernorm = Gemma2RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
          self.pre_feedforward_layernorm = Gemma2RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
          self.post_feedforward_layernorm = Gemma2RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
        end

        def call(x, mask: nil, cache: nil)
          r = self_attn.call(input_layernorm.call(x), mask: mask, cache: cache)
          h = x + post_attention_layernorm.call(r)
          r = mlp.call(pre_feedforward_layernorm.call(h))
          h + post_feedforward_layernorm.call(r)
        end
      end

      class Gemma2Model < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.embed_tokens = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
          self.layers = Array.new(args.num_hidden_layers) { TransformerBlock.new(args) }
          self.norm = Gemma2RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
        end

        def call(inputs, cache: nil)
          mx = MLX::Core
          h = embed_tokens.call(inputs)
          # Gemma2 scales embeddings by sqrt(hidden_size)
          h = h * Math.sqrt(@args.hidden_size)
          layer_cache = cache || [nil] * layers.length

          mask = nil
          mask = "causal" if h.shape[1] > 1

          layers.each_with_index do |layer, i|
            h = layer.call(h, mask: mask, cache: layer_cache[i])
          end

          norm.call(h)
        end
      end

      class Model < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          @final_logit_softcapping = args.final_logit_softcapping
          self.model = Gemma2Model.new(args)
        end

        def call(inputs, cache: nil)
          mx = MLX::Core
          out = model.call(inputs, cache: cache)
          # Tied embeddings
          out = model.embed_tokens.as_linear(out)
          # Final logit softcapping
          out = mx.tanh(out / @final_logit_softcapping) * @final_logit_softcapping
          out
        end

        def sanitize(weights)
          weights.reject { |k, _| k.include?("self_attn.rotary_emb.inv_freq") }
        end

        def layers
          model.layers
        end
      end

      Models.register("gemma2", Model, ModelArgs)
    end
  end
end
