require_relative "activations"
require_relative "cache"
require_relative "rope_utils"
require_relative "switch_layers"

module MlxLm
  module Models
    module Afmoe
      class ModelArgs < BaseModelArgs
        field :model_type
        field :layer_types
        field :vocab_size, default: 200_192
        field :hidden_size, default: 2048
        field :intermediate_size, default: 6144
        field :moe_intermediate_size, default: 1024
        field :num_hidden_layers, default: 32
        field :num_attention_heads, default: 32
        field :num_key_value_heads, default: 4
        field :head_dim, default: 64
        field :max_position_embeddings, default: 131_072
        field :rms_norm_eps, default: 1e-5
        field :rope_theta, default: 10_000.0
        field :rope_scaling, default: nil
        field :tie_word_embeddings, default: false
        field :num_experts, default: 128
        field :num_experts_per_tok, default: 8
        field :num_shared_experts, default: 1
        field :num_dense_layers, default: 2
        field :route_norm, default: true
        field :route_scale, default: 2.826
        field :score_func, default: "sigmoid"
        field :n_group, default: 1
        field :topk_group, default: 1
        field :sliding_window, default: 2048
        field :mup_enabled, default: true

        def initialize(**kwargs)
          super
          @num_key_value_heads ||= @num_attention_heads
          @layer_types ||= Array.new(@num_hidden_layers) { "full_attention" }
        end
      end

      class Attention < MLX::NN::Module
        def initialize(args, is_local_attention: false)
          super()
          @hidden_size = args.hidden_size
          @num_attention_heads = args.num_attention_heads
          @num_key_value_heads = args.num_key_value_heads
          @head_dim = args.head_dim
          @is_local_attention = is_local_attention
          @scale = @head_dim**(-0.5)

          self.q_proj = MLX::NN::Linear.new(
            @hidden_size,
            @num_attention_heads * @head_dim,
            bias: false
          )
          self.k_proj = MLX::NN::Linear.new(
            @hidden_size,
            @num_key_value_heads * @head_dim,
            bias: false
          )
          self.v_proj = MLX::NN::Linear.new(
            @hidden_size,
            @num_key_value_heads * @head_dim,
            bias: false
          )
          self.o_proj = MLX::NN::Linear.new(
            @num_attention_heads * @head_dim,
            @hidden_size,
            bias: false
          )

          self.q_norm = MLX::NN::RMSNorm.new(@head_dim, eps: args.rms_norm_eps)
          self.k_norm = MLX::NN::RMSNorm.new(@head_dim, eps: args.rms_norm_eps)
          self.gate_proj = MLX::NN::Linear.new(
            @hidden_size,
            @num_attention_heads * @head_dim,
            bias: false
          )

          if @is_local_attention
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

          queries = q_proj.call(x).reshape([b, l, @num_attention_heads, @head_dim]).transpose([0, 2, 1, 3])
          keys = k_proj.call(x).reshape([b, l, @num_key_value_heads, @head_dim]).transpose([0, 2, 1, 3])
          values = v_proj.call(x).reshape([b, l, @num_key_value_heads, @head_dim]).transpose([0, 2, 1, 3])

          queries = q_norm.call(queries)
          keys = k_norm.call(keys)

          if @is_local_attention && respond_to?(:rope)
            if cache
              queries = rope.call(queries, offset: cache.offset)
              keys = rope.call(keys, offset: cache.offset)
            else
              queries = rope.call(queries)
              keys = rope.call(keys)
            end
          end

          if cache
            keys, values = cache.update_and_fetch(keys, values)
          end

          output = mx.scaled_dot_product_attention(queries, keys, values, @scale, mask)
          output = output.transpose([0, 2, 1, 3]).reshape([b, l, @num_attention_heads * @head_dim])

          gate = mx.sigmoid(gate_proj.call(x))
          output = output * gate
          o_proj.call(output)
        end
      end

      class MLP < MLX::NN::Module
        def initialize(args, intermediate_size: nil)
          super()
          dim = args.hidden_size
          hidden_dim = intermediate_size || args.intermediate_size

          self.gate_proj = MLX::NN::Linear.new(dim, hidden_dim, bias: false)
          self.down_proj = MLX::NN::Linear.new(hidden_dim, dim, bias: false)
          self.up_proj = MLX::NN::Linear.new(dim, hidden_dim, bias: false)
        end

        def call(x)
          down_proj.call(Activations.swiglu(gate_proj.call(x), up_proj.call(x)))
        end
      end

      class MoERouter < MLX::NN::Module
        def initialize(args)
          super()
          self.gate = MLX::NN::Linear.new(args.hidden_size, args.num_experts, bias: false)
        end

        def call(x)
          gate.call(x)
        end
      end

      class AfmoeMoE < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          @num_experts = args.num_experts
          @num_experts_per_tok = args.num_experts_per_tok
          @route_norm = args.route_norm
          @route_scale = args.route_scale
          @score_func = args.score_func
          @n_group = args.n_group
          @topk_group = args.topk_group

          self.router = MoERouter.new(args)
          self.expert_bias = MLX::Core.zeros([args.num_experts])
          self.experts = SwitchLayers::SwitchGLU.new(
            args.hidden_size,
            args.moe_intermediate_size,
            args.num_experts
          )

          if args.num_shared_experts.to_i > 0
            shared_intermediate_size = args.moe_intermediate_size * args.num_shared_experts
            self.shared_experts = MLP.new(args, intermediate_size: shared_intermediate_size)
          end
        end

        def call(x)
          mx = MLX::Core

          gates = router.call(x)
          scores = if @score_func == "sigmoid"
            mx.sigmoid(gates.astype(mx.float32))
          else
            mx.softmax(gates.astype(mx.float32), -1)
          end

          selection_scores = scores + expert_bias

          if @n_group.to_i > 1
            experts_per_group = selection_scores.shape[-1] / @n_group
            selection_scores = mx.unflatten(selection_scores, -1, [@n_group, experts_per_group])
            group_scores = mx.topk(selection_scores, 2, -1)
            group_scores = mx.expand_dims(mx.sum(group_scores, -1), -1)

            drop_count = @n_group - @topk_group.to_i
            if drop_count > 0
              group_idx = mx.argpartition(group_scores, drop_count - 1, -2)
              take_ids = mx.array((0...drop_count).to_a, dtype: mx.int32)
              group_idx = mx.take(group_idx, take_ids, -2)
              selection_scores = mx.put_along_axis(
                selection_scores,
                mx.stop_gradient(group_idx),
                mx.array(0.0),
                -2
              )
            end

            selection_scores = mx.flatten(selection_scores, -2, -1)
          end

          k = [@num_experts_per_tok.to_i, selection_scores.shape[-1]].min
          inds = mx.argpartition(selection_scores * -1.0, k - 1, -1)
          take_ids = mx.array((0...k).to_a, dtype: mx.int32)
          inds = mx.take(inds, take_ids, -1)

          selected_scores = mx.take_along_axis(scores, inds, -1)
          if @route_norm && k > 1
            denominator = mx.expand_dims(mx.sum(selected_scores, -1), -1)
            selected_scores = selected_scores / denominator
          end
          selected_scores = selected_scores * @route_scale

          y = experts.call(x, inds)
          y = mx.sum(y * mx.expand_dims(selected_scores, -1), -2).astype(y.dtype)
          y = y + shared_experts.call(x) if @args.num_shared_experts.to_i > 0
          y
        end
      end

      class DecoderLayer < MLX::NN::Module
        attr_reader :use_sliding

        def initialize(args, layer_idx, use_sliding: false)
          super()
          @use_sliding = use_sliding
          self.self_attn = Attention.new(args, is_local_attention: @use_sliding)
          self.mlp = if layer_idx < args.num_dense_layers
            MLP.new(args)
          else
            AfmoeMoE.new(args)
          end
          self.input_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
          self.post_attention_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
          self.pre_mlp_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
          self.post_mlp_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
        end

        def call(x, mask: nil, cache: nil)
          r = self_attn.call(input_layernorm.call(x), mask: mask, cache: cache)
          r = post_attention_layernorm.call(r)
          h = x + r

          r = mlp.call(pre_mlp_layernorm.call(h))
          r = post_mlp_layernorm.call(r)
          h + r
        end
      end

      class AfmoeModel < MLX::NN::Module
        attr_reader :layer_types, :sliding_window

        def initialize(args)
          super()
          @hidden_size = args.hidden_size
          @layer_types = args.layer_types
          @sliding_window = args.sliding_window
          @mup_enabled = args.mup_enabled

          self.embed_tokens = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
          self.layers = @layer_types.each_with_index.map do |layer_type, idx|
            DecoderLayer.new(args, idx, use_sliding: layer_type == "sliding_attention")
          end
          self.norm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)

          self.fa_idx = @layer_types.index("full_attention") || 0
          self.swa_idx = @layer_types.index("sliding_attention")
        end

        def call(inputs, cache: nil)
          h = embed_tokens.call(inputs)
          h = h * Math.sqrt(@hidden_size) if @mup_enabled

          layer_cache = cache || [nil] * layers.length
          full_mask = _create_attention_mask(h, layer_cache[fa_idx])
          sliding_mask = if swa_idx.nil?
            nil
          else
            _create_attention_mask(h, layer_cache[swa_idx], window_size: @sliding_window)
          end

          layers.each_with_index do |layer, i|
            mask = layer.use_sliding ? sliding_mask : full_mask
            h = layer.call(h, mask: mask, cache: layer_cache[i])
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
          self.model = AfmoeModel.new(args)
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
          result = weights.reject { |key, _| key.to_s.include?("rotary_emb.inv_freq") }
          result = result.dup
          result.delete("lm_head.weight") if @args.tie_word_embeddings

          @args.num_hidden_layers.times do |layer_idx|
            next if layer_idx < @args.num_dense_layers.to_i

            prefix = "model.layers.#{layer_idx}"
            %w[up_proj down_proj gate_proj].each do |projection|
              %w[weight scales biases].each do |param|
                first_key = "#{prefix}.mlp.experts.0.#{projection}.#{param}"
                next unless result.key?(first_key)

                expert_keys = (0...@args.num_experts).map do |expert_idx|
                  "#{prefix}.mlp.experts.#{expert_idx}.#{projection}.#{param}"
                end
                next unless expert_keys.all? { |key| result.key?(key) }

                stacked = expert_keys.map { |key| result.delete(key) }
                result["#{prefix}.mlp.experts.#{projection}.#{param}"] = mx.stack(stacked)
              end
            end
          end

          result
        end

        def layers
          model.layers
        end

        def make_cache
          layers.map do |layer|
            if layer.use_sliding
              MlxLm::RotatingKVCache.new(max_size: model.sliding_window)
            else
              MlxLm::KVCache.new
            end
          end
        end

        def cast_predicate
          lambda { |key| !key.to_s.include?("expert_bias") }
        end

        def quant_predicate
          lambda do |path, _|
            if path.to_s.include?("router.gate")
              { group_size: 64, bits: 8 }
            else
              true
            end
          end
        end
      end

      Models.register("afmoe", Model, ModelArgs)
    end
  end
end
