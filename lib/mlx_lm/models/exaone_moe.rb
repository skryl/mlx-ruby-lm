require_relative "activations"
require_relative "cache"
require_relative "rope_utils"
require_relative "switch_layers"

module MlxLm
  module Models
    module ExaoneMoe
      class ModelArgs < BaseModelArgs
        field :model_type, default: "exaone_moe"
        field :vocab_size
        field :hidden_size
        field :intermediate_size
        field :moe_intermediate_size
        field :num_hidden_layers
        field :num_attention_heads
        field :num_key_value_heads, default: nil
        field :head_dim, default: nil
        field :num_experts
        field :num_experts_per_tok
        field :num_shared_experts
        field :rms_norm_eps
        field :max_position_embeddings
        field :sliding_window
        field :layer_types, default: nil
        field :is_moe_layer, default: nil
        field :n_group, default: 1
        field :topk_group, default: 1
        field :routed_scaling_factor, default: 2.5
        field :norm_topk_prob, default: true
        field :scoring_func, default: "sigmoid"
        field :topk_method, default: "noaux_tc"
        field :rope_theta, default: 1_000_000.0
        field :rope_scaling, default: nil
        field :rope_parameters, default: nil
        field :tie_word_embeddings, default: false

        def initialize(**kwargs)
          super
          @num_key_value_heads ||= @num_attention_heads
          @head_dim ||= @hidden_size / @num_attention_heads
          @layer_types ||= Array.new(@num_hidden_layers) { "full_attention" }
          @is_moe_layer ||= Array.new(@num_hidden_layers, false)

          return unless @rope_parameters.respond_to?(:[])

          rope_theta = @rope_parameters["rope_theta"] || @rope_parameters[:rope_theta]
          @rope_theta = rope_theta unless rope_theta.nil?
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
        def initialize(args)
          super()
          @top_k = args.num_experts_per_tok
          @norm_topk_prob = args.norm_topk_prob
          @n_routed_experts = args.num_experts
          @routed_scaling_factor = args.routed_scaling_factor
          @n_group = args.n_group
          @topk_group = args.topk_group

          raise ArgumentError, "Unsupported topk method: #{args.topk_method}" unless args.topk_method == "noaux_tc"

          mx = MLX::Core
          self.weight = mx.zeros([@n_routed_experts, args.hidden_size])
          self.e_score_correction_bias = mx.zeros([@n_routed_experts])
        end

        def call(x)
          mx = MLX::Core
          gates = mx.matmul(x, mx.transpose(weight))
          ExaoneMoe.group_expert_select(
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

      class MLP < MLX::NN::Module
        def initialize(args, intermediate_size: nil)
          super()
          hidden_size = args.hidden_size
          intermediate_size ||= args.intermediate_size

          self.gate_proj = MLX::NN::Linear.new(hidden_size, intermediate_size, bias: false)
          self.up_proj = MLX::NN::Linear.new(hidden_size, intermediate_size, bias: false)
          self.down_proj = MLX::NN::Linear.new(intermediate_size, hidden_size, bias: false)
        end

        def call(x)
          down_proj.call(Activations.swiglu(gate_proj.call(x), up_proj.call(x)))
        end
      end

      class MoE < MLX::NN::Module
        def initialize(args)
          super()
          @num_shared_experts = args.num_shared_experts

          self.switch_mlp = SwitchLayers::SwitchGLU.new(
            args.hidden_size,
            args.moe_intermediate_size,
            args.num_experts
          )
          self.gate = MoEGate.new(args)

          if !@num_shared_experts.nil? && @num_shared_experts > 0
            shared_intermediate = args.moe_intermediate_size * @num_shared_experts
            self.shared_experts = MLP.new(args, intermediate_size: shared_intermediate)
          end
        end

        def call(x)
          mx = MLX::Core
          inds, scores = gate.call(x)
          y = switch_mlp.call(x, inds)
          y = mx.sum(y * mx.expand_dims(scores, -1), -2).astype(y.dtype)
          y = y + shared_experts.call(x) if respond_to?(:shared_experts)
          y
        end
      end

      class Attention < MLX::NN::Module
        attr_reader :is_sliding_window

        def initialize(args, layer_idx)
          super()

          @hidden_size = args.hidden_size
          @n_heads = args.num_attention_heads
          @n_kv_heads = args.num_key_value_heads
          @head_dim = args.head_dim
          @scale = @head_dim**(-0.5)

          self.q_proj = MLX::NN::Linear.new(@hidden_size, @n_heads * @head_dim, bias: false)
          self.k_proj = MLX::NN::Linear.new(@hidden_size, @n_kv_heads * @head_dim, bias: false)
          self.v_proj = MLX::NN::Linear.new(@hidden_size, @n_kv_heads * @head_dim, bias: false)
          self.o_proj = MLX::NN::Linear.new(@n_heads * @head_dim, @hidden_size, bias: false)

          self.q_norm = MLX::NN::RMSNorm.new(@head_dim, eps: args.rms_norm_eps)
          self.k_norm = MLX::NN::RMSNorm.new(@head_dim, eps: args.rms_norm_eps)

          @is_sliding_window = args.layer_types[layer_idx] == "sliding_attention"
          apply_rope_all_layers = !args.layer_types.include?("sliding_attention")
          @use_rope = @is_sliding_window || apply_rope_all_layers

          if @use_rope
            self.rope = MlxLm::Models.initialize_rope(
              @head_dim,
              args.rope_theta,
              false,
              args.rope_scaling,
              max_position_embeddings: args.max_position_embeddings
            )
          end
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
            if @use_rope
              queries = rope.call(queries, offset: cache.offset)
              keys = rope.call(keys, offset: cache.offset)
            end
            keys, values = cache.update_and_fetch(keys, values)
          elsif @use_rope
            queries = rope.call(queries)
            keys = rope.call(keys)
          end

          output = mx.scaled_dot_product_attention(queries, keys, values, @scale, mask)
          output = output.transpose([0, 2, 1, 3]).reshape([b, l, @n_heads * @head_dim])
          o_proj.call(output)
        end
      end

      class DecoderLayer < MLX::NN::Module
        attr_reader :is_sliding_window

        def initialize(args, layer_idx)
          super()

          self.self_attn = Attention.new(args, layer_idx)
          self.mlp = args.is_moe_layer[layer_idx] ? MoE.new(args) : MLP.new(args)
          @is_sliding_window = self_attn.is_sliding_window

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

      class ExaoneMoeModel < MLX::NN::Module
        def initialize(args)
          super()
          @window_size = args.sliding_window

          self.embed_tokens = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
          self.layers = Array.new(args.num_hidden_layers) { |idx| DecoderLayer.new(args, idx) }
          self.norm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)

          self.swa_idx = nil
          self.ga_idx = nil
          layers.each_with_index do |layer, idx|
            self.swa_idx = idx if swa_idx.nil? && layer.is_sliding_window
            self.ga_idx = idx if ga_idx.nil? && !layer.is_sliding_window
            break unless swa_idx.nil? || ga_idx.nil?
          end
        end

        def call(inputs, cache: nil)
          h = embed_tokens.call(inputs)
          layer_cache = cache || [nil] * layers.length

          global_cache = ga_idx.nil? ? layer_cache[0] : layer_cache[ga_idx]
          swa_cache = swa_idx.nil? ? layer_cache[0] : layer_cache[swa_idx]

          global_mask = _create_attention_mask(h, global_cache)
          swa_mask = _create_attention_mask(h, swa_cache, window_size: @window_size)

          layers.each_with_index do |layer, idx|
            mask = layer.is_sliding_window ? swa_mask : global_mask
            h = layer.call(h, mask: mask, cache: layer_cache[idx])
          end

          norm.call(h)
        end

        private

        def _create_attention_mask(h, cache = nil, window_size: nil)
          n = h.shape[1]
          if cache && cache.respond_to?(:make_mask)
            return cache.make_mask(n, window_size: window_size)
          end

          if window_size
            offset = 0
            if cache
              offset = cache.offset if cache.respond_to?(:offset)
              if cache.instance_variable_defined?(:@max_size)
                max_size = cache.instance_variable_get(:@max_size)
                offset = [max_size - 1, offset].min if max_size && max_size > 0
              end
            end
            return _create_causal_mask(n, offset: offset, window_size: window_size) if offset + n > window_size
          end

          return nil if n == 1

          "causal"
        end

        def _create_causal_mask(n, offset: 0, window_size: nil)
          mx = MLX::Core
          rinds = mx.arange(0, offset + n, 1, mx.int32).reshape([1, offset + n])
          linds = mx.arange(offset, offset + n, 1, mx.int32).reshape([n, 1])

          mask = mx.greater_equal(linds, rinds)
          if window_size
            mask = mx.logical_and(mask, mx.less(linds, mx.add(rinds, window_size)))
          end
          mask
        end
      end

      class Model < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.model_type = args.model_type
          self.model = ExaoneMoeModel.new(args)

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
          result = weights.reject { |k, _| k.start_with?("mtp.") }
          num_experts = @args.num_experts.to_i

          @args.num_hidden_layers.to_i.times do |layer_idx|
            next unless @args.is_moe_layer[layer_idx]

            prefix = "model.layers.#{layer_idx}.mlp"
            bias_key = "#{prefix}.e_score_correction_bias"
            if result.key?(bias_key)
              result["#{prefix}.gate.e_score_correction_bias"] = result.delete(bias_key)
            end

            %w[gate_proj down_proj up_proj].each do |proj_name|
              %w[weight scales biases].each do |param_name|
                first_key = "#{prefix}.experts.0.#{proj_name}.#{param_name}"
                last_key = "#{prefix}.experts.#{num_experts - 1}.#{proj_name}.#{param_name}"
                next unless result.key?(first_key) && result.key?(last_key)

                expert_keys = (0...num_experts).map do |expert_idx|
                  "#{prefix}.experts.#{expert_idx}.#{proj_name}.#{param_name}"
                end
                next unless expert_keys.all? { |key| result.key?(key) }

                stacked = expert_keys.map { |key| result.delete(key) }
                result["#{prefix}.switch_mlp.#{proj_name}.#{param_name}"] = mx.stack(stacked)
              end
            end
          end

          result.delete("lm_head.weight") if @args.tie_word_embeddings
          result
        end

        def layers
          model.layers
        end

        def cast_predicate
          lambda { |key| !key.include?("e_score_correction_bias") }
        end

        def make_cache
          max_window = @args.sliding_window || @args.max_position_embeddings || 1
          layers.map do |layer|
            if layer.is_sliding_window
              RotatingKVCache.new(max_size: max_window, keep: 0)
            else
              KVCache.new
            end
          end
        end
      end

      Models.register("exaone_moe", Model, ModelArgs)
    end
  end
end
