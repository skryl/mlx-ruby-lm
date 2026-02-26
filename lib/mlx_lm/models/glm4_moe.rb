require_relative "activations"
require_relative "pipeline"
require_relative "switch_layers"

module MlxLm
  module Models
    module Glm4Moe
      class ModelArgs < BaseModelArgs
        field :model_type, default: "glm4_moe"
        field :vocab_size
        field :hidden_size
        field :intermediate_size
        field :max_position_embeddings
        field :moe_intermediate_size
        field :norm_topk_prob
        field :num_attention_heads
        field :n_group
        field :head_dim, default: nil
        field :topk_group
        field :n_shared_experts
        field :n_routed_experts
        field :routed_scaling_factor
        field :num_experts_per_tok
        field :first_k_dense_replace
        field :num_hidden_layers
        field :num_key_value_heads, default: nil
        field :rms_norm_eps
        field :rope_theta
        field :rope_scaling, default: nil
        field :use_qk_norm
        field :tie_word_embeddings
        field :attention_bias
        field :partial_rotary_factor
        field :scoring_func, default: "sigmoid"
        field :topk_method, default: "noaux_tc"

        def initialize(**kwargs)
          super
          @num_key_value_heads ||= @num_attention_heads
          @head_dim ||= @hidden_size / @num_attention_heads
        end
      end

      class Attention < MLX::NN::Module
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
          self.o_proj = MLX::NN::Linear.new(@n_heads * @head_dim, dim, bias: false)

          @use_qk_norm = args.use_qk_norm
          if @use_qk_norm
            self.q_norm = MLX::NN::RMSNorm.new(@head_dim, eps: args.rms_norm_eps)
            self.k_norm = MLX::NN::RMSNorm.new(@head_dim, eps: args.rms_norm_eps)
          end

          rope_dims = [(@head_dim * args.partial_rotary_factor.to_f).to_i, 1].max
          self.rope = MLX::NN::RoPE.new(
            rope_dims,
            traditional: false,
            base: args.rope_theta
          )
        end

        def call(x, mask: nil, cache: nil)
          mx = MLX::Core
          b, l, _d = x.shape

          queries = q_proj.call(x).reshape([b, l, @n_heads, @head_dim])
          keys = k_proj.call(x).reshape([b, l, @n_kv_heads, @head_dim])
          values = v_proj.call(x).reshape([b, l, @n_kv_heads, @head_dim])

          if @use_qk_norm
            queries = q_norm.call(queries)
            keys = k_norm.call(keys)
          end

          queries = queries.transpose([0, 2, 1, 3])
          keys = keys.transpose([0, 2, 1, 3])
          values = values.transpose([0, 2, 1, 3])

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

      class MLP < MLX::NN::Module
        def initialize(config, hidden_size: nil, intermediate_size: nil)
          super()
          hidden_size ||= config.hidden_size
          intermediate_size ||= config.intermediate_size

          self.gate_proj = MLX::NN::Linear.new(hidden_size, intermediate_size, bias: false)
          self.up_proj = MLX::NN::Linear.new(hidden_size, intermediate_size, bias: false)
          self.down_proj = MLX::NN::Linear.new(intermediate_size, hidden_size, bias: false)
        end

        def call(x)
          down_proj.call(Activations.swiglu(gate_proj.call(x), up_proj.call(x)))
        end
      end

      module_function

      def group_expert_select(
        gates,
        e_score_correction_bias,
        top_k,
        n_group,
        topk_group,
        routed_scaling_factor,
        norm_topk_prob
      )
        mx = MLX::Core

        scores = mx.sigmoid(gates.astype(mx.float32))
        orig_scores = scores
        scores = scores + e_score_correction_bias

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

        k = [[top_k.to_i, 1].max, scores.shape[-1]].min
        inds = mx.argpartition(scores * -1.0, k - 1, -1)
        take_ids = mx.array((0...k).to_a, dtype: mx.int32)
        inds = mx.take(inds, take_ids, -1)

        selected_scores = mx.take_along_axis(orig_scores, inds, -1)
        if k > 1 && norm_topk_prob
          denominator = mx.expand_dims(mx.sum(selected_scores, -1), -1)
          selected_scores = selected_scores / (denominator + 1e-20)
        end

        selected_scores = selected_scores * routed_scaling_factor.to_f
        [inds, selected_scores]
      end

      class MoEGate < MLX::NN::Module
        def initialize(config)
          super()
          @top_k = config.num_experts_per_tok
          @norm_topk_prob = config.norm_topk_prob
          @n_routed_experts = config.n_routed_experts
          @routed_scaling_factor = config.routed_scaling_factor
          @n_group = config.n_group
          @topk_group = config.topk_group

          raise ArgumentError, "Unsupported topk method: #{config.topk_method}" unless config.topk_method == "noaux_tc"

          mx = MLX::Core
          self.weight = mx.zeros([@n_routed_experts, config.hidden_size])
          self.e_score_correction_bias = mx.zeros([@n_routed_experts])
        end

        def call(x)
          mx = MLX::Core
          gates = mx.matmul(x, mx.transpose(weight))
          Glm4Moe.group_expert_select(
            gates,
            e_score_correction_bias,
            @top_k,
            @n_group,
            @topk_group,
            @routed_scaling_factor,
            @norm_topk_prob
          )
        end
      end

      class MoE < MLX::NN::Module
        def initialize(config)
          super()
          @config = config

          self.switch_mlp = SwitchLayers::SwitchGLU.new(
            config.hidden_size,
            config.moe_intermediate_size,
            config.n_routed_experts
          )

          self.gate = MoEGate.new(config)
          unless config.n_shared_experts.nil?
            shared_intermediate = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = MLP.new(config, intermediate_size: shared_intermediate)
          end
        end

        def call(x)
          mx = MLX::Core
          inds, scores = gate.call(x)
          y = switch_mlp.call(x, inds)
          y = mx.sum(y * mx.expand_dims(scores, -1), -2).astype(y.dtype)
          y = y + shared_experts.call(x) unless @config.n_shared_experts.nil?
          y
        end
      end

      class DecoderLayer < MLX::NN::Module
        def initialize(config, layer_idx)
          super()
          self.self_attn = Attention.new(config)
          self.mlp = if !config.n_routed_experts.nil? && layer_idx >= config.first_k_dense_replace
            MoE.new(config)
          else
            MLP.new(config)
          end

          self.input_layernorm = MLX::NN::RMSNorm.new(config.hidden_size, eps: config.rms_norm_eps)
          self.post_attention_layernorm = MLX::NN::RMSNorm.new(config.hidden_size, eps: config.rms_norm_eps)
        end

        def call(x, mask: nil, cache: nil)
          r = self_attn.call(input_layernorm.call(x), mask: mask, cache: cache)
          h = x + r
          r = mlp.call(post_attention_layernorm.call(h))
          h + r
        end
      end

      class LanguageModel < MLX::NN::Module
        include PipelineMixin

        def initialize(config)
          super()
          self.embed_tokens = MLX::NN::Embedding.new(config.vocab_size, config.hidden_size)
          self.layers = Array.new(config.num_hidden_layers) { |idx| DecoderLayer.new(config, idx) }
          self.norm = MLX::NN::RMSNorm.new(config.hidden_size, eps: config.rms_norm_eps)
        end

        def call(x, cache: nil)
          h = embed_tokens.call(x)
          active_layers = pipeline_layers
          layer_cache = cache || [nil] * active_layers.length
          mask = _create_attention_mask(h, layer_cache[0])

          active_layers.each_with_index do |layer, idx|
            h = layer.call(h, mask: mask, cache: layer_cache[idx])
          end

          norm.call(h)
        end

        private

        def _create_attention_mask(h, cache = nil)
          n = h.shape[1]
          return cache.make_mask(n) if cache && cache.respond_to?(:make_mask)
          return nil if n == 1

          "causal"
        end
      end

      class Model < MLX::NN::Module
        def initialize(config)
          super()
          @args = config
          self.model_type = config.model_type
          self.model = LanguageModel.new(config)
          self.lm_head = MLX::NN::Linear.new(config.hidden_size, config.vocab_size, bias: false)
        end

        def call(inputs, cache: nil)
          out = model.call(inputs, cache: cache)
          lm_head.call(out)
        end

        def sanitize(weights)
          mx = MLX::Core
          result = weights.dup
          mpt_layer = @args.num_hidden_layers.to_i

          @args.num_hidden_layers.to_i.times do |layer_idx|
            prefix = "model.layers.#{layer_idx}.mlp"
            %w[gate_proj down_proj up_proj].each do |proj_name|
              %w[weight scales biases].each do |param_name|
                first_key = "#{prefix}.experts.0.#{proj_name}.#{param_name}"
                next unless result.key?(first_key)

                expert_keys = (0...@args.n_routed_experts.to_i).map do |expert_idx|
                  "#{prefix}.experts.#{expert_idx}.#{proj_name}.#{param_name}"
                end
                next unless expert_keys.all? { |key| result.key?(key) }

                stacked = expert_keys.map { |key| result.delete(key) }
                result["#{prefix}.switch_mlp.#{proj_name}.#{param_name}"] = mx.stack(stacked)
              end
            end
          end

          result.reject { |key, _| key.start_with?("model.layers.#{mpt_layer}") }
        end

        def layers
          model.pipeline_layers
        end

        def cast_predicate
          lambda { |key| !key.include?("e_score_correction_bias") }
        end
      end

      Models.register("glm4_moe", Model, ModelArgs)
    end
  end
end
