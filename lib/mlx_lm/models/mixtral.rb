module MlxLm
  module Models
    module Mixtral
      class ModelArgs < BaseModelArgs
        field :model_type, default: "mixtral"
        field :hidden_size, default: 4096
        field :num_hidden_layers, default: 32
        field :num_attention_heads, default: 32
        field :num_key_value_heads, default: 8
        field :intermediate_size, default: 14336
        field :vocab_size, default: 32000
        field :rms_norm_eps, default: 1e-5
        field :rope_theta, default: 1e6
        field :rope_traditional, default: false
        field :rope_scaling, default: nil
        field :num_local_experts, default: 8
        field :num_experts_per_tok, default: 2
        field :tie_word_embeddings, default: false

        def initialize(**kwargs)
          super
          @num_key_value_heads ||= @num_attention_heads
        end
      end

      class Attention < MLX::NN::Module
        def initialize(args)
          super()
          dim = args.hidden_size
          @n_heads = args.num_attention_heads
          @n_kv_heads = args.num_key_value_heads
          @head_dim = dim / @n_heads
          @scale = @head_dim**(-0.5)

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

          output = mx.scaled_dot_product_attention(queries, keys, values, @scale, mask)
          output = output.transpose([0, 2, 1, 3]).reshape([b, l, @n_heads * @head_dim])
          o_proj.call(output)
        end
      end

      class SparseMoeBlock < MLX::NN::Module
        def initialize(args)
          super()
          @num_experts = args.num_local_experts
          @num_experts_per_tok = args.num_experts_per_tok
          dim = args.hidden_size
          hidden_dim = args.intermediate_size

          self.gate = MLX::NN::Linear.new(dim, @num_experts, bias: false)
          self.switch_mlp = SwitchLayers::SwitchGLU.new(dim, hidden_dim, @num_experts)
        end

        def call(x)
          mx = MLX::Core
          k = @num_experts_per_tok

          gates = gate.call(x)
          inds = mx.stop_gradient(mx.argpartition(gates * -1.0, k - 1, -1))
          take_ids = mx.array((0...k).to_a, dtype: mx.int32)
          inds = mx.take(inds, take_ids, -1)

          scores = mx.take_along_axis(gates, inds, -1)
          scores = mx.softmax(scores.astype(mx.float32), -1).astype(gates.dtype)

          y = switch_mlp.call(x, inds)
          y = mx.sum(y * mx.expand_dims(scores, -1), -2)
          y
        end
      end

      class MixtralDecoderLayer < MLX::NN::Module
        def initialize(args)
          super()
          self.self_attn = Attention.new(args)
          self.block_sparse_moe = SparseMoeBlock.new(args)
          self.input_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
          self.post_attention_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
        end

        def call(x, mask: nil, cache: nil)
          r = self_attn.call(input_layernorm.call(x), mask: mask, cache: cache)
          h = x + r
          r = block_sparse_moe.call(post_attention_layernorm.call(h))
          h + r
        end
      end

      class MixtralModel < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.embed_tokens = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
          self.layers = Array.new(args.num_hidden_layers) { MixtralDecoderLayer.new(args) }
          self.norm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
        end

        def call(inputs, cache: nil)
          h = embed_tokens.call(inputs)
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
          self.model = MixtralModel.new(args)
          unless args.tie_word_embeddings
            self.lm_head = MLX::NN::Linear.new(args.hidden_size, args.vocab_size, bias: false)
          end
        end

        def call(inputs, cache: nil)
          out = model.call(inputs, cache: cache)
          if @args.tie_word_embeddings
            model.embed_tokens.as_linear(out)
          else
            lm_head.call(out)
          end
        end

        def sanitize(weights)
          mx = MLX::Core
          result = weights.reject { |k, _| k.include?("self_attn.rotary_emb.inv_freq") }
          result.delete("lm_head.weight") if @args.tie_word_embeddings

          # Convert per-expert weights to stacked SwitchGLU format
          @args.num_hidden_layers.times do |l|
            prefix = "model.layers.#{l}"
            [["w1", "gate_proj"], ["w2", "down_proj"], ["w3", "up_proj"]].each do |n, m|
              ["weight", "scales", "biases"].each do |k|
                key0 = "#{prefix}.block_sparse_moe.experts.0.#{n}.#{k}"
                if result.key?(key0)
                  to_join = (0...@args.num_local_experts).map { |e|
                    result.delete("#{prefix}.block_sparse_moe.experts.#{e}.#{n}.#{k}")
                  }
                  result["#{prefix}.block_sparse_moe.switch_mlp.#{m}.#{k}"] = mx.stack(to_join)
                end
              end
            end
          end

          result
        end

        def layers
          model.layers
        end
      end

      Models.register("mixtral", Model, ModelArgs)
    end
  end
end
