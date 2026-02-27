require_relative "activations"
require_relative "cache"
require_relative "rope_utils"
require_relative "switch_layers"

module MlxLm
  module Models
    module Lfm2Moe
      class ModelArgs < BaseModelArgs
        field :model_type, default: "lfm2_moe"
        field :vocab_size
        field :hidden_size
        field :intermediate_size
        field :moe_intermediate_size
        field :num_hidden_layers
        field :num_experts
        field :num_experts_per_tok
        field :norm_topk_prob
        field :num_attention_heads
        field :num_key_value_heads, default: nil
        field :max_position_embeddings
        field :use_expert_bias
        field :num_dense_layers
        field :norm_eps
        field :conv_bias
        field :conv_L_cache
        field :rope_theta, default: 1_000_000.0
        field :rope_parameters, default: nil
        field :full_attn_idxs, default: nil
        field :layer_types, default: nil

        def initialize(**kwargs)
          super
          rope_theta_from_params = _rope_theta_from_parameters
          @rope_theta = rope_theta_from_params unless rope_theta_from_params.nil?
          @num_key_value_heads ||= @num_attention_heads
          @full_attn_idxs ||= _full_attn_idxs_from_layer_types
        end

        private

        def _rope_theta_from_parameters
          return nil unless @rope_parameters.is_a?(Hash)

          @rope_parameters["rope_theta"] || @rope_parameters[:rope_theta]
        end

        def _full_attn_idxs_from_layer_types
          return [] unless @layer_types.is_a?(Array)

          @layer_types.each_with_index.filter_map do |layer_type, i|
            i if layer_type.to_s == "full_attention"
          end
        end
      end

      class Attention < MLX::NN::Module
        def initialize(args)
          super()

          dim = args.hidden_size
          @n_heads = args.num_attention_heads
          @n_kv_heads = args.num_key_value_heads
          @head_dim = args.hidden_size / @n_heads
          @scale = @head_dim**(-0.5)

          self.q_layernorm = MLX::NN::RMSNorm.new(@head_dim, eps: args.norm_eps)
          self.k_layernorm = MLX::NN::RMSNorm.new(@head_dim, eps: args.norm_eps)

          self.q_proj = MLX::NN::Linear.new(dim, @n_heads * @head_dim, bias: false)
          self.k_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: false)
          self.v_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: false)
          self.out_proj = MLX::NN::Linear.new(@n_heads * @head_dim, dim, bias: false)

          self.rope = MlxLm::Models.initialize_rope(
            @head_dim,
            args.rope_theta,
            false,
            nil,
            max_position_embeddings: args.max_position_embeddings
          )
        end

        def call(x, mask: nil, cache: nil)
          mx = MLX::Core
          b, l, _d = x.shape

          queries = q_proj.call(x)
          keys = k_proj.call(x)
          values = v_proj.call(x)

          queries = q_layernorm.call(queries.reshape([b, l, @n_heads, @head_dim])).transpose([0, 2, 1, 3])
          keys = k_layernorm.call(keys.reshape([b, l, @n_kv_heads, @head_dim])).transpose([0, 2, 1, 3])
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
          out_proj.call(output)
        end
      end

      class ShortConv < MLX::NN::Module
        def initialize(args, layer_idx)
          super()
          _ = layer_idx
          @args = args
          @l_cache = args.conv_L_cache
          @hidden_size = args.hidden_size

          self.conv = MLX::NN::Conv1d.new(
            args.hidden_size,
            args.hidden_size,
            @l_cache,
            padding: 0,
            groups: args.hidden_size,
            bias: args.conv_bias
          )
          self.in_proj = MLX::NN::Linear.new(args.hidden_size, 3 * args.hidden_size, bias: args.conv_bias)
          self.out_proj = MLX::NN::Linear.new(args.hidden_size, args.hidden_size, bias: args.conv_bias)
        end

        def call(x, mask: nil, cache: nil)
          mx = MLX::Core

          projected = in_proj.call(x)
          b_gate, c_gate, x_gate = mx.split(projected, [@hidden_size, 2 * @hidden_size], -1)
          bx = b_gate * x_gate
          bx = mx.where(mask.reshape([mask.shape[0], mask.shape[1], 1]), bx, 0) unless mask.nil?

          if cache
            state = if cache[0].nil?
              mx.zeros([bx.shape[0], @l_cache - 1, @hidden_size], dtype: bx.dtype)
            else
              cache[0]
            end

            bx = mx.concatenate([state, bx], 1)
            n_keep = @l_cache - 1
            t = x_gate.shape[1]

            if cache.lengths
              ends = mx.clip(cache.lengths, 0, t)
              positions = mx.expand_dims(
                mx.expand_dims(ends, 1) + mx.arange(n_keep),
                -1
              )
              cache[0] = mx.take_along_axis(bx, positions, 1)
            else
              if n_keep > 0
                split_at = bx.shape[1] - n_keep
                cache[0] = mx.split(bx, [split_at], 1)[1]
              else
                cache[0] = mx.zeros([bx.shape[0], 0, bx.shape[2]], dtype: bx.dtype)
              end
            end

            cache.advance(t)
          else
            bx = mx.pad(
              bx,
              [
                [0, 0],
                [@l_cache - 1, 0],
                [0, 0],
              ]
            )
          end

          conv_out = conv.call(bx)
          out_proj.call(c_gate * conv_out)
        end
      end

      class MLP < MLX::NN::Module
        def initialize(config, intermediate_size: nil)
          super()
          @hidden_size = config.hidden_size
          @intermediate_size = intermediate_size || config.intermediate_size
          self.gate_proj = MLX::NN::Linear.new(@hidden_size, @intermediate_size, bias: false)
          self.up_proj = MLX::NN::Linear.new(@hidden_size, @intermediate_size, bias: false)
          self.down_proj = MLX::NN::Linear.new(@intermediate_size, @hidden_size, bias: false)
        end

        def call(x)
          down_proj.call(Activations.swiglu(gate_proj.call(x), up_proj.call(x)))
        end
      end

      class SparseMoeBlock < MLX::NN::Module
        def initialize(args)
          super()
          dim = args.hidden_size
          intermediate_size = args.moe_intermediate_size

          @num_experts = args.num_experts
          @top_k = args.num_experts_per_tok
          @norm_topk_prob = args.norm_topk_prob
          @use_expert_bias = args.use_expert_bias

          self.gate = MLX::NN::Linear.new(dim, @num_experts, bias: false)
          self.switch_mlp = SwitchLayers::SwitchGLU.new(dim, intermediate_size, @num_experts)
          self.expert_bias = MLX::Core.zeros([@num_experts]) if @use_expert_bias
        end

        def call(x)
          mx = MLX::Core

          gates = gate.call(x).astype(mx.float32)
          gates = mx.softmax(gates, -1)
          gates = gates + expert_bias if @use_expert_bias

          k = [[@top_k.to_i, 1].max, @num_experts].min
          inds = mx.argpartition(gates, -k, -1)
          take_ids = mx.array((@num_experts - k...@num_experts).to_a, dtype: mx.int32)
          inds = mx.take(inds, take_ids, -1)

          scores = mx.take_along_axis(gates, inds, -1)
          if @norm_topk_prob
            scores = scores / (mx.expand_dims(mx.sum(scores, -1), -1) + 1e-20)
          end
          scores = scores.astype(x.dtype)

          y = switch_mlp.call(x, inds)
          mx.sum(y * mx.expand_dims(scores, -1), -2)
        end
      end

      class DecoderLayer < MLX::NN::Module
        attr_reader :is_attention_layer

        def initialize(args, layer_idx)
          super()
          @is_attention_layer = args.full_attn_idxs.include?(layer_idx)

          if @is_attention_layer
            self.self_attn = Attention.new(args)
          else
            self.conv = ShortConv.new(args, layer_idx)
          end

          self.feed_forward = if layer_idx < args.num_dense_layers
            MLP.new(args, intermediate_size: args.intermediate_size)
          else
            SparseMoeBlock.new(args)
          end

          self.operator_norm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.norm_eps)
          self.ffn_norm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.norm_eps)
        end

        def call(x, mask: nil, cache: nil)
          r = if @is_attention_layer
            self_attn.call(operator_norm.call(x), mask: mask, cache: cache)
          else
            conv.call(operator_norm.call(x), mask: mask, cache: cache)
          end

          h = x + r
          h + feed_forward.call(ffn_norm.call(h))
        end
      end

      class Lfm2MoeModel < MLX::NN::Module
        def initialize(args)
          super()
          @args = args

          self.embed_tokens = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
          self.layers = Array.new(args.num_hidden_layers) { |i| DecoderLayer.new(args, i) }
          self.embedding_norm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.norm_eps)

          self.fa_idx = args.full_attn_idxs[0] || 0
          self.conv_idx = 0
          args.num_hidden_layers.times do |i|
            if args.full_attn_idxs.include?(i)
              self.conv_idx += 1
            else
              break
            end
          end
          self.conv_idx = [conv_idx, args.num_hidden_layers - 1].min
        end

        def call(inputs, cache: nil, input_embeddings: nil)
          h = input_embeddings || embed_tokens.call(inputs)
          layer_cache = cache || [nil] * layers.length

          attn_mask = _create_attention_mask(h, layer_cache[fa_idx])
          conv_mask = _create_ssm_mask(h, layer_cache[conv_idx])

          layers.each_with_index do |layer, i|
            mask = layer.is_attention_layer ? attn_mask : conv_mask
            h = layer.call(h, mask: mask, cache: layer_cache[i])
          end

          embedding_norm.call(h)
        end

        private

        def _create_attention_mask(h, cache = nil)
          n = h.shape[1]
          return cache.make_mask(n) if cache && cache.respond_to?(:make_mask)
          return nil if n == 1

          "causal"
        end

        def _create_ssm_mask(h, cache = nil)
          return cache.make_mask(h.shape[1]) if cache && cache.respond_to?(:make_mask)

          nil
        end
      end

      class Model < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.model_type = args.model_type
          self.model = Lfm2MoeModel.new(args)
        end

        def call(inputs, cache: nil, input_embeddings: nil)
          out = model.call(inputs, cache: cache, input_embeddings: input_embeddings)
          model.embed_tokens.as_linear(out)
        end

        def sanitize(weights)
          mx = MLX::Core
          sanitized = {}

          weights.each do |name, param|
            current = param
            if name.include?("conv.weight") && _transpose_conv_weight?(param)
              current = mx.swapaxes(param, 1, 2)
            end

            key = name
            {
              "w1.weight" => "gate_proj.weight",
              "w2.weight" => "down_proj.weight",
              "w3.weight" => "up_proj.weight",
            }.each do |old_name, new_name|
              key = key.gsub(old_name, new_name) if key.include?(old_name)
            end

            sanitized[key] = current
          end

          @args.num_hidden_layers.times do |layer_idx|
            prefix = "model.layers.#{layer_idx}"
            %w[gate_proj down_proj up_proj].each do |projection|
              first_key = "#{prefix}.feed_forward.experts.0.#{projection}.weight"
              next unless sanitized.key?(first_key)

              expert_keys = (0...@args.num_experts).map do |expert_idx|
                "#{prefix}.feed_forward.experts.#{expert_idx}.#{projection}.weight"
              end
              next unless expert_keys.all? { |k| sanitized.key?(k) }

              stacked = expert_keys.map { |k| sanitized.delete(k) }
              sanitized["#{prefix}.feed_forward.switch_mlp.#{projection}.weight"] = mx.stack(stacked)
            end
          end

          sanitized
        end

        def layers
          model.layers
        end

        def make_cache
          layers.map do |layer|
            if layer.is_attention_layer
              MlxLm::KVCache.new
            else
              MlxLm::ArraysCache.new(1)
            end
          end
        end

        def quant_predicate
          lambda do |path, _|
            if path.end_with?("feed_forward.gate")
              { group_size: 64, bits: 8 }
            else
              true
            end
          end
        end

        def cast_predicate
          lambda { |k| !k.include?("expert_bias") }
        end

        private

        def _transpose_conv_weight?(param)
          return false unless param.respond_to?(:shape)
          return false unless param.shape.is_a?(Array)
          return false unless param.shape.length >= 3

          param.shape[-1] > param.shape[1]
        end
      end

      Models.register("lfm2_moe", Model, ModelArgs)
    end
  end
end
