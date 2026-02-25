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

      class Expert < MLX::NN::Module
        def initialize(dim, hidden_dim)
          super()
          self.gate_proj = MLX::NN::Linear.new(dim, hidden_dim, bias: false)
          self.down_proj = MLX::NN::Linear.new(hidden_dim, dim, bias: false)
          self.up_proj = MLX::NN::Linear.new(dim, hidden_dim, bias: false)
        end

        def call(x)
          down_proj.call(MLX::NN.silu(gate_proj.call(x)) * up_proj.call(x))
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
          self.experts = Array.new(@num_experts) { Expert.new(dim, hidden_dim) }
        end

        def call(x)
          mx = MLX::Core
          ne = @num_experts_per_tok
          orig_shape = x.shape
          dims = x.shape[-1]
          tokens = x.size / dims
          x_flat = x.reshape([tokens, dims])

          # Route tokens to experts
          gates = gate.call(x_flat)
          inds = mx.argpartition(gates * -1.0, ne - 1, -1)
          take_ids = mx.array((0...ne).to_a, mx.int32)
          inds = mx.take(inds, take_ids, 1)

          scores = mx.take_along_axis(gates, inds, -1)
          scores = mx.softmax(scores.astype(mx.float32), -1).astype(gates.dtype)

          # Evaluate experts per token
          inds_list = inds.tolist
          y_rows = []
          (0...x_flat.shape[0]).each do |i|
            xt = x_flat[i]
            selected = inds_list[i]
            selected = [selected].flatten
            expert_outs = selected.map { |eidx|
              mx.expand_dims(experts[eidx].call(xt), 0)
            }
            yt = mx.concatenate(expert_outs, 0)
            # Weighted sum: yt shape [ne, dim], scores[i] shape [ne]
            st = scores[i]
            weighted = yt * mx.expand_dims(st, -1)
            summed = mx.sum(weighted, 0)
            y_rows << mx.expand_dims(summed, 0)
          end

          y = mx.concatenate(y_rows, 0)
          y.reshape(orig_shape)
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
          result = weights.reject { |k, _| k.include?("self_attn.rotary_emb.inv_freq") }
          result.delete("lm_head.weight") if @args.tie_word_embeddings
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
