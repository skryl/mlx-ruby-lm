require_relative "activations"
require_relative "rope_utils"
require_relative "switch_layers"

module MlxLm
  module Models
    module Dots1
      class ModelArgs < BaseModelArgs
        field :model_type, default: "dots1"
        field :hidden_size
        field :num_hidden_layers
        field :intermediate_size
        field :num_attention_heads
        field :rms_norm_eps
        field :vocab_size
        field :max_position_embeddings, default: nil
        field :num_key_value_heads
        field :first_k_dense_replace
        field :moe_intermediate_size
        field :n_routed_experts
        field :n_shared_experts
        field :norm_topk_prob
        field :num_experts_per_tok
        field :rope_theta
        field :routed_scaling_factor
        field :head_dim, default: nil
        field :scoring_func, default: "noaux_tc"
        field :n_group, default: 1
        field :topk_group, default: 1
        field :attention_bias, default: false
        field :mlp_bias, default: false
        field :rope_scaling, default: nil
        field :tie_word_embeddings, default: false

        def initialize(**kwargs)
          super
          @num_key_value_heads ||= @num_attention_heads
          @head_dim ||= @hidden_size / @num_attention_heads
          @n_group ||= 1
          @topk_group ||= 1
        end
      end

      class Dots1Attention < MLX::NN::Module
        def initialize(args)
          super()

          dim = args.hidden_size
          @n_heads = args.num_attention_heads
          @n_kv_heads = args.num_key_value_heads
          @head_dim = args.head_dim
          @scale = @head_dim**(-0.5)

          self.q_proj = MLX::NN::Linear.new(dim, @n_heads * @head_dim, bias: args.attention_bias)
          self.k_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: args.attention_bias)
          self.v_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: args.attention_bias)
          self.o_proj = MLX::NN::Linear.new(@n_heads * @head_dim, dim, bias: args.attention_bias)

          self.q_norm = MLX::NN::RMSNorm.new(@head_dim, eps: args.rms_norm_eps)
          self.k_norm = MLX::NN::RMSNorm.new(@head_dim, eps: args.rms_norm_eps)
          self.rope = MlxLm::Models.initialize_rope(
            @head_dim,
            args.rope_theta,
            false,
            args.rope_scaling,
            max_position_embeddings: args.max_position_embeddings
          )
        end

        def call(x, mask: nil, cache: nil)
          mx = MLX::Core
          b, l, _d = x.shape

          queries = q_proj.call(x)
          keys = k_proj.call(x)
          values = v_proj.call(x)

          queries = q_norm.call(queries.reshape([b, l, @n_heads, @head_dim])).transpose([0, 2, 1, 3])
          keys = k_norm.call(keys.reshape([b, l, @n_kv_heads, @head_dim])).transpose([0, 2, 1, 3])
          values = values.reshape([b, l, @n_kv_heads, @head_dim]).transpose([0, 2, 1, 3])

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

      class Dots1TopkRouter < MLX::NN::Module
        def initialize(args)
          super()
          mx = MLX::Core
          @top_k = args.num_experts_per_tok
          @norm_topk_prob = args.norm_topk_prob
          @n_routed_experts = args.n_routed_experts
          @routed_scaling_factor = args.routed_scaling_factor
          @n_group = args.n_group
          @topk_group = args.topk_group
          self.weight = mx.zeros([@n_routed_experts, args.hidden_size]).astype(mx.float32)
          self.e_score_correction_bias = mx.zeros([@n_routed_experts]).astype(mx.float32)
        end

        def call(x)
          mx = MLX::Core

          gates = mx.matmul(x, mx.transpose(weight))
          scores = mx.sigmoid(gates.astype(mx.float32))
          scores = scores + e_score_correction_bias.reshape([1, 1, @n_routed_experts])

          k = [[@top_k.to_i, 1].max, @n_routed_experts].min
          inds = mx.stop_gradient(mx.argpartition(scores * -1.0, k - 1, -1))
          take_ids = mx.array((0...k).to_a, dtype: mx.int32)
          inds = mx.take(inds, take_ids, -1)

          selected_scores = mx.take_along_axis(mx.sigmoid(gates.astype(mx.float32)), inds, -1)
          if k > 1 && @norm_topk_prob
            denom = mx.expand_dims(mx.sum(selected_scores, -1), -1)
            selected_scores = selected_scores / denom
          end
          selected_scores = selected_scores * @routed_scaling_factor.to_f

          [inds, selected_scores.astype(gates.dtype)]
        end
      end

      class Dots1MLP < MLX::NN::Module
        def initialize(args, hidden_size: nil, intermediate_size: nil)
          super()
          @hidden_size = hidden_size || args.hidden_size
          @intermediate_size = intermediate_size || args.intermediate_size

          self.gate_proj = MLX::NN::Linear.new(@hidden_size, @intermediate_size, bias: args.mlp_bias)
          self.up_proj = MLX::NN::Linear.new(@hidden_size, @intermediate_size, bias: args.mlp_bias)
          self.down_proj = MLX::NN::Linear.new(@intermediate_size, @hidden_size, bias: args.mlp_bias)
        end

        def call(x)
          down_proj.call(Activations.swiglu(gate_proj.call(x), up_proj.call(x)))
        end
      end

      class Dots1MoE < MLX::NN::Module
        def initialize(args)
          super()
          @n_shared_experts = args.n_shared_experts
          self.experts = SwitchLayers::SwitchGLU.new(
            args.hidden_size,
            args.moe_intermediate_size,
            args.n_routed_experts,
            bias: args.mlp_bias
          )
          self.gate = Dots1TopkRouter.new(args)

          if @n_shared_experts && @n_shared_experts > 0
            self.shared_experts = Dots1MLP.new(
              args,
              intermediate_size: args.moe_intermediate_size * @n_shared_experts
            )
          end
        end

        def call(x)
          mx = MLX::Core
          inds, scores = gate.call(x)
          y = experts.call(x, inds)
          y = mx.sum(y * mx.expand_dims(scores.astype(y.dtype), -1), -2)

          y = y + shared_experts.call(x) if @n_shared_experts && @n_shared_experts > 0
          y
        end
      end

      class Dots1DecoderLayer < MLX::NN::Module
        def initialize(args, layer_idx)
          super()
          self.self_attn = Dots1Attention.new(args)
          if layer_idx >= args.first_k_dense_replace
            self.mlp = Dots1MoE.new(args)
          else
            self.mlp = Dots1MLP.new(args)
          end
          self.input_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
          self.post_attention_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
        end

        def call(x, mask: nil, cache: nil)
          r = self_attn.call(input_layernorm.call(x), mask: mask, cache: cache)
          h = x + r
          r = mlp.call(post_attention_layernorm.call(h))
          h + r
        end
      end

      class Dots1Model < MLX::NN::Module
        def initialize(args)
          super()
          self.embed_tokens = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
          self.layers = Array.new(args.num_hidden_layers) { |layer_idx| Dots1DecoderLayer.new(args, layer_idx) }
          self.norm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
        end

        def call(inputs, cache: nil)
          h = embed_tokens.call(inputs)
          layer_cache = cache || [nil] * layers.length
          mask = _create_attention_mask(h, layer_cache[0])

          layers.each_with_index do |layer, layer_idx|
            h = layer.call(h, mask: mask, cache: layer_cache[layer_idx])
          end

          norm.call(h)
        end

        private

        def _create_attention_mask(h, cache)
          n = h.shape[1]
          return cache.make_mask(n) if cache && cache.respond_to?(:make_mask)
          return nil if n == 1

          "causal"
        end
      end

      class Model < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.model_type = args.model_type
          self.model = Dots1Model.new(args)
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
          result = weights.dup
          result.delete("lm_head.weight") if @args.tie_word_embeddings

          experts_count = @args.n_routed_experts.to_i
          if experts_count > 0
            mx = MLX::Core
            @args.num_hidden_layers.times do |layer_idx|
              next if layer_idx < @args.first_k_dense_replace

              prefix = "model.layers.#{layer_idx}.mlp"
              %w[gate_proj down_proj up_proj].each do |projection|
                %w[weight scales biases].each do |param|
                  first_key = "#{prefix}.experts.0.#{projection}.#{param}"
                  next unless result.key?(first_key)

                  expert_keys = (0...experts_count).map do |expert_idx|
                    "#{prefix}.experts.#{expert_idx}.#{projection}.#{param}"
                  end
                  next unless expert_keys.all? { |key| result.key?(key) }

                  stacked = expert_keys.map { |key| result.delete(key) }
                  result["#{prefix}.experts.#{projection}.#{param}"] = mx.stack(stacked)
                end
              end
            end
          end

          result.reject { |k, _| k.include?("rotary_emb.inv_freq") }
        end

        def layers
          model.layers
        end
      end

      Models.register("dots1", Model, ModelArgs)
    end
  end
end
