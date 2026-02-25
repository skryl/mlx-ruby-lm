module MlxLm
  module Models
    module DeepSeek
      class ModelArgs < BaseModelArgs
        field :model_type, default: "deepseek"
        field :hidden_size, default: 4096
        field :num_hidden_layers, default: 30
        field :num_attention_heads, default: 32
        field :num_key_value_heads, default: 32
        field :intermediate_size, default: 11008
        field :moe_intermediate_size, default: 1407
        field :vocab_size, default: 102400
        field :rms_norm_eps, default: 1e-6
        field :rope_theta, default: 10000.0
        field :rope_scaling, default: nil
        field :attention_bias, default: false
        field :n_shared_experts, default: nil
        field :n_routed_experts, default: nil
        field :num_experts_per_tok, default: nil
        field :moe_layer_freq, default: 1
        field :first_k_dense_replace, default: 0
        field :max_position_embeddings, default: 2048

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

          bias = args.attention_bias
          self.q_proj = MLX::NN::Linear.new(dim, @n_heads * @head_dim, bias: bias)
          self.k_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: bias)
          self.v_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: bias)
          self.o_proj = MLX::NN::Linear.new(@n_heads * @head_dim, dim, bias: bias)

          rope_scale = 1.0
          if args.rope_scaling && args.rope_scaling["type"] == "linear"
            rope_scale = 1.0 / args.rope_scaling["factor"]
          end

          self.rope = MLX::NN::RoPE.new(
            @head_dim,
            traditional: false,
            base: args.rope_theta,
            scale: rope_scale
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

      class DeepseekMLP < MLX::NN::Module
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

      class MoEGate < MLX::NN::Module
        def initialize(args)
          super()
          @top_k = args.num_experts_per_tok
          self.weight = MLX::Core.zeros([args.n_routed_experts, args.hidden_size])
        end

        def call(x)
          mx = MLX::Core
          gates = mx.matmul(x, mx.transpose(weight))
          scores = mx.softmax(gates.astype(mx.float32), -1).astype(gates.dtype)
          k = @top_k
          inds = mx.stop_gradient(mx.argpartition(scores * -1.0, k - 1, -1))
          take_ids = mx.array((0...k).to_a, dtype: mx.int32)
          inds = mx.take(inds, take_ids, -1)
          scores = mx.take_along_axis(scores, inds, -1)
          [inds, scores]
        end
      end

      class DeepseekMoE < MLX::NN::Module
        def initialize(args)
          super()
          @n_shared_experts = args.n_shared_experts
          dim = args.hidden_size
          moe_dim = args.moe_intermediate_size

          self.switch_mlp = SwitchLayers::SwitchGLU.new(dim, moe_dim, args.n_routed_experts)
          self.gate = MoEGate.new(args)

          if args.n_shared_experts && args.n_shared_experts > 0
            shared_dim = moe_dim * args.n_shared_experts
            self.shared_experts = DeepseekMLP.new(dim, shared_dim)
          end
        end

        def call(x)
          mx = MLX::Core
          inds, scores = gate.call(x)
          y = switch_mlp.call(x, inds)
          y = mx.sum(y * mx.expand_dims(scores, -1), -2)

          if @n_shared_experts && @n_shared_experts > 0
            y = y + shared_experts.call(x)
          end

          y
        end
      end

      class DecoderLayer < MLX::NN::Module
        def initialize(args, layer_idx)
          super()
          self.self_attn = Attention.new(args)
          self.input_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
          self.post_attention_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)

          # Determine if this layer uses MoE or dense
          use_moe = args.n_routed_experts &&
                    layer_idx >= args.first_k_dense_replace &&
                    layer_idx % args.moe_layer_freq == 0

          if use_moe
            self.mlp = DeepseekMoE.new(args)
          else
            self.mlp = DeepseekMLP.new(args.hidden_size, args.intermediate_size)
          end
        end

        def call(x, mask: nil, cache: nil)
          r = self_attn.call(input_layernorm.call(x), mask: mask, cache: cache)
          h = x + r
          r = mlp.call(post_attention_layernorm.call(h))
          h + r
        end
      end

      class DeepseekModel < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.embed_tokens = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
          self.layers = Array.new(args.num_hidden_layers) { |i| DecoderLayer.new(args, i) }
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
          self.model = DeepseekModel.new(args)
          self.lm_head = MLX::NN::Linear.new(args.hidden_size, args.vocab_size, bias: false)
        end

        def call(inputs, cache: nil)
          out = model.call(inputs, cache: cache)
          lm_head.call(out)
        end

        def sanitize(weights)
          mx = MLX::Core
          result = weights.reject { |k, _| k.include?("self_attn.rotary_emb.inv_freq") }

          # Convert per-expert weights to stacked SwitchGLU format
          @args.num_hidden_layers.times do |l|
            prefix = "model.layers.#{l}"
            ["gate_proj", "down_proj", "up_proj"].each do |m|
              ["weight", "scales", "biases"].each do |k|
                key0 = "#{prefix}.mlp.experts.0.#{m}.#{k}"
                if result.key?(key0)
                  to_join = (0...@args.n_routed_experts).map { |e|
                    result.delete("#{prefix}.mlp.experts.#{e}.#{m}.#{k}")
                  }
                  result["#{prefix}.mlp.switch_mlp.#{m}.#{k}"] = mx.stack(to_join)
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

      Models.register("deepseek", Model, ModelArgs)
    end
  end
end
