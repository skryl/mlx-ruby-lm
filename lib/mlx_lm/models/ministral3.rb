require_relative "activations"
require_relative "cache"
require_relative "pipeline"
require_relative "rope_utils"

module MlxLm
  module Models
    module Ministral3
      def self.llama4_attn_scale(size, offset, beta, max_position_embeddings)
        mx = MLX::Core
        positions = mx.arange(size) + offset
        scale = 1.0 + beta.to_f * mx.log(1.0 + mx.floor(positions / max_position_embeddings.to_f))
        scale.reshape([size, 1])
      end

      class ModelArgs < BaseModelArgs
        field :model_type, default: "ministral3"
        field :hidden_size
        field :num_hidden_layers
        field :intermediate_size
        field :num_attention_heads
        field :rms_norm_eps
        field :vocab_size
        field :head_dim, default: nil
        field :max_position_embeddings, default: nil
        field :num_key_value_heads, default: nil
        field :rope_parameters, default: nil
        field :tie_word_embeddings, default: true
        field :layer_types, default: nil
        field :sliding_window, default: nil

        def initialize(**kwargs)
          super
          @num_key_value_heads ||= @num_attention_heads
          @head_dim ||= @hidden_size / @num_attention_heads
          @rope_parameters = _stringify_keys(@rope_parameters || {})
          @rope_parameters["rope_theta"] = 10_000.0 unless @rope_parameters.key?("rope_theta")
          @layer_types ||= Array.new(@num_hidden_layers) { "full_attention" }
        end

        def rope_parameter(key, default = nil)
          return default unless @rope_parameters.is_a?(Hash)
          return @rope_parameters[key.to_s] if @rope_parameters.key?(key.to_s)
          return @rope_parameters[key.to_sym] if @rope_parameters.key?(key.to_sym)

          default
        end

        private

        def _stringify_keys(hash)
          hash.each_with_object({}) do |(key, value), out|
            out[key.to_s] = value
          end
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

          self.q_proj = MLX::NN::Linear.new(dim, @n_heads * @head_dim, bias: false)
          self.k_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: false)
          self.v_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: false)
          self.o_proj = MLX::NN::Linear.new(@n_heads * @head_dim, dim, bias: false)

          self.rope = MlxLm::Models.initialize_rope(
            @head_dim,
            args.rope_parameter("rope_theta", 10_000.0),
            false,
            args.rope_parameters,
            max_position_embeddings: args.max_position_embeddings
          )
        end

        def call(x, attn_scale:, mask: nil, cache: nil)
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

          queries = queries * attn_scale
          output = mx.scaled_dot_product_attention(queries, keys, values, @scale, mask)
          output = output.transpose([0, 2, 1, 3]).reshape([b, l, @n_heads * @head_dim])
          o_proj.call(output)
        end
      end

      class MLP < MLX::NN::Module
        def initialize(args)
          super()

          dim = args.hidden_size
          hidden_dim = args.intermediate_size
          self.gate_proj = MLX::NN::Linear.new(dim, hidden_dim, bias: false)
          self.down_proj = MLX::NN::Linear.new(hidden_dim, dim, bias: false)
          self.up_proj = MLX::NN::Linear.new(dim, hidden_dim, bias: false)
        end

        def call(x)
          down_proj.call(Activations.swiglu(gate_proj.call(x), up_proj.call(x)))
        end
      end

      class TransformerBlock < MLX::NN::Module
        attr_reader :use_sliding

        def initialize(args, use_sliding: false)
          super()
          @use_sliding = use_sliding
          self.self_attn = Attention.new(args)
          self.mlp = MLP.new(args)
          self.input_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
          self.post_attention_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
        end

        def call(x, attn_scale:, mask: nil, cache: nil)
          r = self_attn.call(input_layernorm.call(x), attn_scale: attn_scale, mask: mask, cache: cache)
          h = x + r
          r = mlp.call(post_attention_layernorm.call(h))
          h + r
        end
      end

      class LanguageModel < MLX::NN::Module
        include PipelineMixin
        attr_reader :sliding_window

        def initialize(args)
          super()
          @args = args
          @sliding_window = args.sliding_window
          self.embed_tokens = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
          self.layers = args.layer_types.map do |layer_type|
            TransformerBlock.new(args, use_sliding: layer_type == "sliding_attention")
          end
          self.norm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
        end

        def call(inputs, cache: nil, input_embeddings: nil)
          h = input_embeddings || embed_tokens.call(inputs)
          active_layers = pipeline_layers
          layer_cache = cache || Array.new(active_layers.length)

          first_cache = layer_cache.find { |entry| !entry.nil? }
          offset = first_cache ? first_cache.offset : 0

          fa_idx = nil
          swa_idx = nil
          active_layers.each_with_index do |layer, i|
            if layer.use_sliding
              swa_idx ||= i
            else
              fa_idx ||= i
            end
            break if fa_idx && swa_idx
          end

          fa_mask = fa_idx.nil? ? nil : _create_attention_mask(h, layer_cache[fa_idx])
          swa_mask = if swa_idx.nil?
            nil
          else
            _create_attention_mask(h, layer_cache[swa_idx], window_size: @sliding_window)
          end

          beta = @args.rope_parameter("llama_4_scaling_beta", 0.0).to_f
          max_pos = @args.rope_parameter(
            "original_max_position_embeddings",
            @args.max_position_embeddings || h.shape[1]
          ).to_i
          max_pos = 1 if max_pos <= 0

          attn_scale = MlxLm::Models::Ministral3.llama4_attn_scale(
            inputs.shape[1],
            offset,
            beta,
            max_pos
          ).astype(h.dtype)

          active_layers.each_with_index do |layer, idx|
            mask = layer.use_sliding ? swa_mask : fa_mask
            h = layer.call(h, attn_scale: attn_scale, mask: mask, cache: layer_cache[idx])
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
            offset = cache ? cache.offset : 0
            if cache && cache.instance_variable_defined?(:@max_size)
              max_size = cache.instance_variable_get(:@max_size)
              offset = [max_size - 1, offset].min if max_size && max_size > 0
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
          self.model = LanguageModel.new(args)
          unless args.tie_word_embeddings
            self.lm_head = MLX::NN::Linear.new(args.hidden_size, args.vocab_size, bias: false)
          end
        end

        def call(inputs, cache: nil, input_embeddings: nil)
          out = model.call(inputs, cache: cache, input_embeddings: input_embeddings)
          if @args.tie_word_embeddings
            model.embed_tokens.as_linear(out)
          else
            lm_head.call(out)
          end
        end

        def sanitize(weights)
          sanitized = weights.reject do |key, _|
            key_name = key.to_s
            key_name.include?("self_attn.rotary_emb.inv_freq") || key_name.include?("self_attn.rope.inv_freq")
          end
          sanitized.delete("lm_head.weight") if @args.tie_word_embeddings

          new_weights = {}
          sanitized.each do |key, value|
            key_name = key.to_s
            if key_name.include?("weight_scale_inv")
              wk = key_name.sub("_scale_inv", "")
              next unless sanitized.key?(wk)

              new_weights[wk] = sanitized[wk] * value
            elsif key_name.include?("activation_scale")
              next
            elsif !new_weights.key?(key)
              new_weights[key] = value
            end
          end
          new_weights
        end

        def layers
          model.pipeline_layers
        end

        def make_cache
          max_size = @args.sliding_window || @args.max_position_embeddings || 1
          layers.map do |layer|
            if layer.use_sliding
              MlxLm::RotatingKVCache.new(max_size: max_size)
            else
              MlxLm::KVCache.new
            end
          end
        end
      end

      Models.register("ministral3", Model, ModelArgs)
    end
  end
end
