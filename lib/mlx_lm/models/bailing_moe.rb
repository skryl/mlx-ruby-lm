require_relative "activations"
require_relative "rope_utils"
require_relative "switch_layers"

module MlxLm
  module Models
    module BailingMoe
      class ModelArgs < BaseModelArgs
        field :model_type
        field :hidden_size
        field :intermediate_size
        field :max_position_embeddings
        field :moe_intermediate_size
        field :num_experts
        field :num_shared_experts
        field :norm_topk_prob
        field :num_attention_heads
        field :num_experts_per_tok
        field :num_hidden_layers
        field :num_key_value_heads
        field :rms_norm_eps
        field :rope_theta
        field :vocab_size
        field :first_k_dense_replace
        field :rope_scaling, default: nil
        field :use_bias, default: false
        field :use_qkv_bias, default: false
        field :norm_head, default: false
        field :norm_softmax, default: false
        field :use_qk_norm, default: false
        field :tie_word_embeddings, default: false
        field :partial_rotary_factor, default: 1.0
        field :rotary_dim, default: nil
        field :moe_router_enable_expert_bias, default: false
        field :moe_router_enable_routed_scaling, default: true
        field :routed_scaling_factor, default: 1.0
        field :score_function, default: "softmax"
        field :n_group, default: 1
        field :topk_group, default: 4
        field :moe_shared_expert_intermediate_size, default: nil
        field :moe_router_enable_shared_expert, default: true

        def initialize(**kwargs)
          super
          @num_key_value_heads ||= @num_attention_heads
        end
      end

      module_function

      def aggregate_expert_outputs(expert_outputs, scores)
        mx = MLX::Core
        mx.sum(expert_outputs * mx.expand_dims(scores, -1), -2).astype(expert_outputs.dtype)
      end

      def group_expert_select(
        gates,
        e_score_correction_bias,
        top_k,
        n_group,
        topk_group,
        routed_scaling_factor,
        norm_topk_prob,
        score_function
      )
        mx = MLX::Core
        in_type = gates.dtype

        scores = if score_function == "sigmoid"
          mx.sigmoid(gates.astype(mx.float32))
        else
          mx.softmax(gates.astype(mx.float32), -1)
        end
        orig_scores = scores
        scores = scores + e_score_correction_bias if e_score_correction_bias

        if n_group.to_i > 1
          experts_per_group = scores.shape[-1] / n_group
          scores = mx.unflatten(scores, -1, [n_group, experts_per_group])
          group_scores = mx.topk(scores, 2, -1)
          group_scores = mx.expand_dims(mx.sum(group_scores, -1), -1)

          drop_count = n_group - topk_group.to_i
          if drop_count > 0
            group_idx = mx.argpartition(group_scores, drop_count - 1, -2)
            take_ids = mx.array((0...drop_count).to_a, dtype: mx.int32)
            group_idx = mx.take(group_idx, take_ids, -2)
            scores = mx.put_along_axis(
              scores,
              mx.stop_gradient(group_idx),
              mx.array(0.0),
              -2
            )
          end

          scores = mx.flatten(scores, -2, -1)
        end

        k = [top_k.to_i, scores.shape[-1]].min
        inds = mx.argpartition(scores * -1.0, k - 1, -1)
        take_ids = mx.array((0...k).to_a, dtype: mx.int32)
        inds = mx.take(inds, take_ids, -1)

        selected_scores = mx.take_along_axis(orig_scores, inds, -1)
        if k > 1 && norm_topk_prob
          denominator = mx.expand_dims(mx.sum(selected_scores, -1), -1) + 1e-20
          selected_scores = selected_scores / denominator
        end

        selected_scores = selected_scores * routed_scaling_factor.to_f
        [inds, selected_scores.astype(in_type)]
      end

      class BailingMoeMLP < MLX::NN::Module
        def initialize(args, intermediate_size: nil)
          super()
          hidden_dim = intermediate_size || args.intermediate_size

          self.gate_proj = MLX::NN::Linear.new(args.hidden_size, hidden_dim, bias: args.use_bias)
          self.down_proj = MLX::NN::Linear.new(hidden_dim, args.hidden_size, bias: args.use_bias)
          self.up_proj = MLX::NN::Linear.new(args.hidden_size, hidden_dim, bias: args.use_bias)
        end

        def call(x)
          down_proj.call(Activations.swiglu(gate_proj.call(x), up_proj.call(x)))
        end
      end

      class BailingMoeAttention < MLX::NN::Module
        def initialize(args)
          super()
          @use_qk_norm = args.use_qk_norm
          @num_attention_heads = args.num_attention_heads
          @num_key_value_heads = args.num_key_value_heads
          @head_dim = args.hidden_size / @num_attention_heads
          @scale = @head_dim**(-0.5)

          self.query_key_value = MLX::NN::Linear.new(
            args.hidden_size,
            (@num_attention_heads + 2 * @num_key_value_heads) * @head_dim,
            bias: args.use_qkv_bias
          )
          self.dense = MLX::NN::Linear.new(
            @num_attention_heads * @head_dim,
            args.hidden_size,
            bias: args.use_bias
          )

          if @use_qk_norm
            self.key_layernorm = MLX::NN::RMSNorm.new(@head_dim, eps: args.rms_norm_eps)
            self.query_layernorm = MLX::NN::RMSNorm.new(@head_dim, eps: args.rms_norm_eps)
          end

          rope_dim = args.rotary_dim || (@head_dim * args.partial_rotary_factor.to_f).to_i
          rope_dim = [rope_dim, 1].max
          self.rope = MlxLm::Models.initialize_rope(
            rope_dim,
            args.rope_theta,
            false,
            args.rope_scaling,
            max_position_embeddings: args.max_position_embeddings
          )
        end

        def call(x, mask: nil, cache: nil)
          mx = MLX::Core
          b, l, _d = x.shape

          qkv = query_key_value.call(x)

          q_size = @num_attention_heads * @head_dim
          kv_size = @num_key_value_heads * @head_dim
          q, k, v = mx.split(qkv, [q_size, q_size + kv_size], -1)

          queries = q.reshape([b, l, @num_attention_heads, @head_dim]).transpose([0, 2, 1, 3])
          keys = k.reshape([b, l, @num_key_value_heads, @head_dim]).transpose([0, 2, 1, 3])
          values = v.reshape([b, l, @num_key_value_heads, @head_dim]).transpose([0, 2, 1, 3])

          if @use_qk_norm
            queries = query_layernorm.call(queries)
            keys = key_layernorm.call(keys)
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
          output = output.transpose([0, 2, 1, 3]).reshape([b, l, @num_attention_heads * @head_dim])
          dense.call(output)
        end
      end

      class BailingMoeGate < MLX::NN::Module
        def initialize(args)
          super()
          @norm_topk_prob = args.norm_topk_prob
          @top_k = args.num_experts_per_tok
          @n_group = args.n_group
          @topk_group = args.topk_group
          @routed_scaling_factor = args.routed_scaling_factor
          @score_function = args.score_function

          self.gate_proj = MLX::NN::Linear.new(args.hidden_size, args.num_experts, bias: false)
          self.expert_bias = if args.moe_router_enable_expert_bias
            MLX::Core.zeros([args.num_experts])
          else
            nil
          end
        end

        def call(x)
          BailingMoe.group_expert_select(
            gate_proj.call(x),
            expert_bias,
            @top_k,
            @n_group,
            @topk_group,
            @routed_scaling_factor,
            @norm_topk_prob,
            @score_function
          )
        end
      end

      class BailingMoeSparseMoeBlock < MLX::NN::Module
        def initialize(args)
          super()
          self.switch_mlp = SwitchLayers::SwitchGLU.new(
            args.hidden_size,
            args.moe_intermediate_size,
            args.num_experts,
            bias: args.use_bias
          )
          self.gate = BailingMoeGate.new(args)

          shared_dim = args.moe_shared_expert_intermediate_size || args.moe_intermediate_size
          self.shared_experts = if args.num_shared_experts.to_i > 0 && args.moe_router_enable_shared_expert
            BailingMoeMLP.new(
              args,
              intermediate_size: shared_dim * args.num_shared_experts
            )
          end
        end

        def call(x)
          topk_idx, topk_weight = gate.call(x)
          out = switch_mlp.call(x, topk_idx)
          out = BailingMoe.aggregate_expert_outputs(out, topk_weight)
          out = out + shared_experts.call(x) if respond_to?(:shared_experts)
          out
        end
      end

      class BailingMoeDecoderLayer < MLX::NN::Module
        def initialize(args, layer_idx:)
          super()
          self.attention = BailingMoeAttention.new(args)
          self.mlp = if !args.num_experts.nil? && layer_idx >= args.first_k_dense_replace
            BailingMoeSparseMoeBlock.new(args)
          else
            BailingMoeMLP.new(args)
          end
          self.input_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
          self.post_attention_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
        end

        def call(x, mask: nil, cache: nil)
          r = attention.call(input_layernorm.call(x), mask: mask, cache: cache)
          h = x + r
          r = mlp.call(post_attention_layernorm.call(h))
          h + r
        end
      end

      class BailingMoeModel < MLX::NN::Module
        def initialize(args)
          super()
          self.word_embeddings = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
          self.layers = Array.new(args.num_hidden_layers) do |layer_idx|
            BailingMoeDecoderLayer.new(args, layer_idx: layer_idx)
          end
          self.norm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
        end

        def call(inputs, cache: nil)
          h = word_embeddings.call(inputs)
          layer_cache = cache || [nil] * layers.length
          mask = _create_attention_mask(h, layer_cache[0])

          layers.each_with_index do |layer, layer_idx|
            h = layer.call(h, mask: mask, cache: layer_cache[layer_idx])
          end
          norm.call(h)
        end

        private

        def _create_attention_mask(hidden, cache)
          return cache.make_mask(hidden.shape[1]) if cache && cache.respond_to?(:make_mask)
          return nil if hidden.shape[1] == 1

          "causal"
        end
      end

      class Model < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          @norm_head = args.norm_head
          self.model_type = args.model_type
          self.model = BailingMoeModel.new(args)
          unless args.tie_word_embeddings
            self.lm_head = MLX::NN::Linear.new(args.hidden_size, args.vocab_size, bias: false)
          end
        end

        def call(inputs, cache: nil)
          out = model.call(inputs, cache: cache)
          if @args.tie_word_embeddings
            model.word_embeddings.as_linear(out)
          else
            lm_head.call(out)
          end
        end

        def sanitize(weights)
          mx = MLX::Core
          result = weights.dup

          result.delete("lm_head.weight") if @args.tie_word_embeddings

          if @norm_head && result.key?("lm_head.weight")
            w = result["lm_head.weight"]
            dtype = w.dtype
            w_fp32 = w.astype(mx.float32)
            weight_norm = mx.sqrt(mx.sum(mx.square(w_fp32), 0, true)) + 1e-7
            result["lm_head.weight"] = (w_fp32 / weight_norm).astype(dtype)
          end

          @args.num_hidden_layers.times do |layer_idx|
            next if layer_idx < @args.first_k_dense_replace.to_i

            prefix = "model.layers.#{layer_idx}"
            %w[gate_proj down_proj up_proj].each do |projection|
              %w[weight scales biases].each do |param|
                first_key = "#{prefix}.mlp.experts.0.#{projection}.#{param}"
                next unless result.key?(first_key)

                expert_keys = (0...@args.num_experts).map do |expert_idx|
                  "#{prefix}.mlp.experts.#{expert_idx}.#{projection}.#{param}"
                end
                next unless expert_keys.all? { |key| result.key?(key) }

                stacked = expert_keys.map { |key| result.delete(key) }
                result["#{prefix}.mlp.switch_mlp.#{projection}.#{param}"] = mx.stack(stacked)
              end
            end

            if result.key?("#{prefix}.mlp.gate.weight")
              result["#{prefix}.mlp.gate.gate_proj.weight"] = result.delete("#{prefix}.mlp.gate.weight")
            end
            if result.key?("#{prefix}.mlp.gate.bias")
              result["#{prefix}.mlp.gate.gate_proj.bias"] = result.delete("#{prefix}.mlp.gate.bias")
            end
          end

          result
        end

        def quant_predicate
          lambda do |path, _|
            if path.to_s.end_with?("mlp.gate.gate_proj")
              { group_size: 64, bits: 8 }
            else
              true
            end
          end
        end

        def cast_predicate
          lambda { |key| !key.to_s.include?("expert_bias") }
        end

        def layers
          model.layers
        end
      end

      Models.register("bailing_moe", Model, ModelArgs)
    end
  end
end
