require_relative "activations"
require_relative "switch_layers"

module MlxLm
  module Models
    module Klear
      class ModelArgs < BaseModelArgs
        field :model_type, default: "Klear"
        field :hidden_size
        field :num_hidden_layers
        field :intermediate_size
        field :num_attention_heads
        field :attention_bias
        field :mlp_only_layers
        field :num_experts
        field :num_experts_per_tok
        field :decoder_sparse_step
        field :n_shared_experts
        field :moe_intermediate_size
        field :rms_norm_eps
        field :vocab_size
        field :num_key_value_heads
        field :rope_theta
        field :max_position_embeddings
        field :norm_topk_prob

        def initialize(**kwargs)
          super
          @mlp_only_layers ||= []
          @num_key_value_heads ||= @num_attention_heads
        end
      end

      class KlearAttention < MLX::NN::Module
        def initialize(args)
          super()
          @num_attention_heads = args.num_attention_heads
          @num_key_value_heads = args.num_key_value_heads
          @head_dim = args.hidden_size / args.num_attention_heads
          @scale = @head_dim**(-0.5)

          self.q_proj = MLX::NN::Linear.new(
            args.hidden_size,
            @num_attention_heads * @head_dim,
            bias: args.attention_bias
          )
          self.k_proj = MLX::NN::Linear.new(
            args.hidden_size,
            @num_key_value_heads * @head_dim,
            bias: args.attention_bias
          )
          self.v_proj = MLX::NN::Linear.new(
            args.hidden_size,
            @num_key_value_heads * @head_dim,
            bias: args.attention_bias
          )
          self.o_proj = MLX::NN::Linear.new(
            @num_attention_heads * @head_dim,
            args.hidden_size,
            bias: args.attention_bias
          )

          self.q_norm = MLX::NN::RMSNorm.new(@head_dim, eps: args.rms_norm_eps)
          self.k_norm = MLX::NN::RMSNorm.new(@head_dim, eps: args.rms_norm_eps)
          self.rope = MLX::NN::RoPE.new(@head_dim, traditional: false, base: args.rope_theta)
        end

        def call(x, mask: nil, cache: nil)
          mx = MLX::Core
          b, l, _d = x.shape

          queries = q_proj.call(x)
          keys = k_proj.call(x)
          values = v_proj.call(x)

          queries = q_norm.call(queries.reshape([b, l, @num_attention_heads, @head_dim])).transpose([0, 2, 1, 3])
          keys = k_norm.call(keys.reshape([b, l, @num_key_value_heads, @head_dim])).transpose([0, 2, 1, 3])
          values = values.reshape([b, l, @num_key_value_heads, @head_dim]).transpose([0, 2, 1, 3])

          if cache
            queries = rope.call(queries, offset: cache.offset)
            keys = rope.call(keys, offset: cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
          else
            queries = rope.call(queries)
            keys = rope.call(keys)
          end

          output = mx.scaled_dot_product_attention(queries, keys, values, @scale, mask)
          output = output.transpose([0, 2, 1, 3]).reshape([b, l, @num_attention_heads * @head_dim])
          o_proj.call(output)
        end
      end

      class KlearMLP < MLX::NN::Module
        def initialize(dim, hidden_dim)
          super()
          self.gate_proj = MLX::NN::Linear.new(dim, hidden_dim, bias: false)
          self.down_proj = MLX::NN::Linear.new(hidden_dim, dim, bias: false)
          self.up_proj = MLX::NN::Linear.new(dim, hidden_dim, bias: false)
        end

        def call(x)
          down_proj.call(Activations.swiglu(gate_proj.call(x), up_proj.call(x)))
        end
      end

      class KlearSparseMoeBlock < MLX::NN::Module
        def initialize(args)
          super()
          @norm_topk_prob = args.norm_topk_prob
          @num_experts = args.num_experts
          @top_k = [args.num_experts_per_tok.to_i, 1].max

          self.gate = MLX::NN::Linear.new(args.hidden_size, @num_experts, bias: false)
          self.experts = SwitchLayers::SwitchGLU.new(
            args.hidden_size,
            args.moe_intermediate_size,
            @num_experts
          )
          self.shared_experts = KlearMLP.new(
            args.hidden_size,
            args.moe_intermediate_size * args.n_shared_experts
          )
          self.coefficient = MLX::NN::Linear.new(args.hidden_size, 2)

          mx = MLX::Core
          self.expert_bias = mx.zeros([@num_experts]).astype(mx.float32)
        end

        def call(x)
          mx = MLX::Core

          routing_weights = mx.sigmoid(gate.call(x).astype(mx.float32))
          biased_weights = routing_weights + expert_bias.reshape([1, 1, @num_experts])

          k = [@top_k, @num_experts].min
          inds = mx.argpartition(biased_weights * -1.0, k - 1, -1)
          take_ids = mx.array((0...k).to_a, dtype: mx.int32)
          inds = mx.take(inds, take_ids, -1)

          scores = mx.take_along_axis(routing_weights, inds, -1)
          if @norm_topk_prob
            denom = mx.expand_dims(mx.sum(scores, -1), -1)
            scores = scores / denom
          end

          scores = scores.astype(x.dtype)
          expert_out = experts.call(x, inds)
          y_experts = mx.sum(expert_out * mx.expand_dims(scores, -1), -2)

          coef = mx.softmax(coefficient.call(x).astype(mx.float32), -1).astype(x.dtype)
          coef_expert, coef_shared = mx.split(coef, [1], -1)
          shared = shared_experts.call(x)

          y_experts * coef_expert + shared * coef_shared
        end
      end

      class KlearDecoderLayer < MLX::NN::Module
        def initialize(args, layer_idx:)
          super()
          self.self_attn = KlearAttention.new(args)
          self.input_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
          self.post_attention_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)

          if _use_sparse_moe_layer?(args, layer_idx)
            self.mlp = KlearSparseMoeBlock.new(args)
          else
            self.mlp = KlearMLP.new(args.hidden_size, args.intermediate_size)
          end
        end

        def call(x, mask: nil, cache: nil)
          r = self_attn.call(input_layernorm.call(x), mask: mask, cache: cache)
          h = x + r
          r = mlp.call(post_attention_layernorm.call(h))
          h + r
        end

        private

        def _use_sparse_moe_layer?(args, layer_idx)
          sparse_step = [args.decoder_sparse_step.to_i, 1].max
          mlp_only_layers = args.mlp_only_layers || []

          !mlp_only_layers.include?(layer_idx) &&
            args.num_experts.to_i > 0 &&
            ((layer_idx + 1) % sparse_step).zero?
        end
      end

      class KlearModel < MLX::NN::Module
        def initialize(args)
          super()
          self.embed_tokens = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
          self.layers = Array.new(args.num_hidden_layers) do |layer_idx|
            KlearDecoderLayer.new(args, layer_idx: layer_idx)
          end
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
          self.model = KlearModel.new(args)
          self.lm_head = MLX::NN::Linear.new(args.hidden_size, args.vocab_size, bias: false)
        end

        def call(inputs, cache: nil)
          lm_head.call(model.call(inputs, cache: cache))
        end

        def sanitize(weights)
          return weights unless weights.key?("model.layers.0.mlp.experts.0.gate_proj.weight")

          mx = MLX::Core
          result = weights.dup

          @args.num_hidden_layers.times do |layer_idx|
            prefix = "model.layers.#{layer_idx}.mlp.experts"
            %w[gate_proj up_proj down_proj].each do |name|
              expert_keys = (0...@args.num_experts).map do |expert_idx|
                "#{prefix}.#{expert_idx}.#{name}.weight"
              end
              next unless expert_keys.all? { |key| result.key?(key) }

              stacked = expert_keys.map { |key| result.delete(key) }
              result["#{prefix}.#{name}.weight"] = mx.stack(stacked)
            end
          end

          result
        end

        def layers
          model.layers
        end

        def quant_predicate
          lambda do |path, _module|
            if path.to_s.end_with?("mlp.gate")
              { "group_size" => 64, "bits" => 8 }
            else
              true
            end
          end
        end

        def cast_predicate
          lambda { |key| !key.to_s.include?("expert_bias") }
        end
      end

      Models.register("Klear", Model, ModelArgs)
    end
  end
end
