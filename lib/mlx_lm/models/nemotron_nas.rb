require_relative "cache"
require_relative "rope_utils"

module MlxLm
  module Models
    module NemotronNas
      module_function

      def find_multiple(n, k)
        remainder = n % k
        remainder.zero? ? n : (n + k - remainder)
      end

      def ffn_mult_to_intermediate_size(ffn_mult, hidden_size)
        intermediate_size = (2 * ffn_mult.to_f * hidden_size / 3).to_i
        find_multiple(intermediate_size, 256)
      end

      class AttentionConfig
        attr_reader :no_op, :replace_with_linear, :sparsify, :n_heads_in_group, :window_length,
          :num_sink_tokens, :use_prefill_window_in_sink_attention, :unshifted_sink

        def initialize(
          no_op: false,
          replace_with_linear: false,
          sparsify: nil,
          n_heads_in_group: nil,
          window_length: nil,
          num_sink_tokens: nil,
          use_prefill_window_in_sink_attention: false,
          unshifted_sink: false
        )
          @no_op = no_op
          @replace_with_linear = replace_with_linear
          @sparsify = sparsify
          @n_heads_in_group = n_heads_in_group
          @window_length = window_length
          @num_sink_tokens = num_sink_tokens
          @use_prefill_window_in_sink_attention = use_prefill_window_in_sink_attention
          @unshifted_sink = unshifted_sink

          if @no_op || @replace_with_linear
            @n_heads_in_group = nil
            @window_length = nil
            @num_sink_tokens = nil
          else
            raise ArgumentError, "n_heads_in_group must be specified for active attention blocks" if @n_heads_in_group.nil?
            raise ArgumentError, "n_heads_in_group must be positive, got #{@n_heads_in_group}" if @n_heads_in_group.to_i <= 0
          end
        end

        def self.from_dict(data)
          hash = _symbolize_keys(data || {})
          new(**hash)
        end

        def self._symbolize_keys(hash)
          hash.each_with_object({}) { |(k, v), out| out[k.to_sym] = v }
        end
        private_class_method :_symbolize_keys
      end

      class FFNConfig
        attr_reader :no_op, :replace_with_linear, :sparsify, :ffn_mult

        def initialize(
          no_op: false,
          replace_with_linear: false,
          sparsify: nil,
          ffn_mult: nil
        )
          @no_op = no_op
          @replace_with_linear = replace_with_linear
          @sparsify = sparsify
          @ffn_mult = ffn_mult

          if @no_op || @replace_with_linear
            @ffn_mult = nil
          else
            raise ArgumentError, "ffn_mult must be specified for active FFN blocks" if @ffn_mult.nil?
            @ffn_mult = @ffn_mult.to_f.round(6)
          end
        end

        def self.from_dict(data)
          hash = _symbolize_keys(data || {})
          new(**hash)
        end

        def self._symbolize_keys(hash)
          hash.each_with_object({}) { |(k, v), out| out[k.to_sym] = v }
        end
        private_class_method :_symbolize_keys
      end

      class BlockConfig
        attr_reader :attention, :ffn

        def initialize(attention:, ffn:)
          @attention = attention
          @ffn = ffn
        end

        def self.from_dict(data)
          hash = data || {}
          attention_data = hash["attention"] || hash[:attention] || {}
          ffn_data = hash["ffn"] || hash[:ffn] || {}
          new(
            attention: AttentionConfig.from_dict(attention_data),
            ffn: FFNConfig.from_dict(ffn_data)
          )
        end
      end

      class ModelArgs < BaseModelArgs
        field :model_type, default: "nemotron-nas"
        field :hidden_size, default: 8192
        field :num_hidden_layers, default: 80
        field :num_attention_heads, default: 64
        field :rms_norm_eps, default: 1e-5
        field :vocab_size, default: 128_256
        field :block_configs, default: []
        field :hidden_act, default: "silu"
        field :attention_bias, default: false
        field :mlp_bias, default: false
        field :rope_theta, default: 500_000.0
        field :rope_scaling, default: nil
        field :max_position_embeddings, default: 131_072
        field :tie_word_embeddings, default: false

        def initialize(**kwargs)
          super
          @block_configs = Array(@block_configs).map do |config|
            config.is_a?(BlockConfig) ? config : BlockConfig.from_dict(config)
          end

          if @block_configs.length != @num_hidden_layers
            raise ArgumentError,
              "Number of block_configs (#{@block_configs.length}) must match num_hidden_layers (#{@num_hidden_layers})"
          end

          validate_rope_scaling!
          validate_block_configs!
        end

        private

        def validate_rope_scaling!
          return unless @rope_scaling

          factor = rope_scaling_value(:factor)
          raise ArgumentError, "rope_scaling must contain 'factor'" if factor.nil?

          rope_type = rope_scaling_value(:rope_type) || rope_scaling_value(:type)
          raise ArgumentError, "rope_scaling must contain 'rope_type'" if rope_type.nil?

          normalized = @rope_scaling.dup
          normalized["rope_type"] = rope_type
          normalized[:rope_type] = rope_type
          @rope_scaling = normalized
        end

        def rope_scaling_value(key)
          return nil unless @rope_scaling
          return @rope_scaling[key] if @rope_scaling.key?(key)

          @rope_scaling[key.to_s]
        end

        def validate_block_configs!
          @block_configs.each_with_index do |block_config, i|
            attention = block_config.attention
            next if attention.no_op || attention.replace_with_linear

            heads_in_group = attention.n_heads_in_group.to_i
            if (@num_attention_heads % heads_in_group) != 0
              raise ArgumentError,
                "Layer #{i}: num_attention_heads (#{@num_attention_heads}) must be divisible by n_heads_in_group (#{attention.n_heads_in_group})"
            end
          end
        end
      end

      class Attention < MLX::NN::Module
        def initialize(args, attention_config)
          super()

          dim = args.hidden_size
          @n_heads = args.num_attention_heads
          @n_kv_heads = @n_heads / attention_config.n_heads_in_group
          @head_dim = args.hidden_size / @n_heads
          raise ArgumentError, "hidden_size (#{dim}) must be divisible by num_attention_heads (#{@n_heads})" if (@head_dim * @n_heads) != dim

          @scale = @head_dim**(-0.5)
          self.q_proj = MLX::NN::Linear.new(dim, @n_heads * @head_dim, bias: args.attention_bias)
          self.k_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: args.attention_bias)
          self.v_proj = MLX::NN::Linear.new(dim, @n_kv_heads * @head_dim, bias: args.attention_bias)
          self.o_proj = MLX::NN::Linear.new(@n_heads * @head_dim, dim, bias: args.attention_bias)
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

      class MLP < MLX::NN::Module
        def initialize(args, ffn_config)
          super()
          hidden_dim = NemotronNas.ffn_mult_to_intermediate_size(ffn_config.ffn_mult, args.hidden_size)
          @act_fn = args.hidden_act

          supported = %w[silu relu gelu gelu_new gelu_fast]
          unless supported.include?(@act_fn)
            raise ArgumentError, "Unknown activation function: #{@act_fn}"
          end

          self.gate_proj = MLX::NN::Linear.new(args.hidden_size, hidden_dim, bias: args.mlp_bias)
          self.down_proj = MLX::NN::Linear.new(hidden_dim, args.hidden_size, bias: args.mlp_bias)
          self.up_proj = MLX::NN::Linear.new(args.hidden_size, hidden_dim, bias: args.mlp_bias)
        end

        def call(x)
          gate = _activate(gate_proj.call(x))
          down_proj.call(gate * up_proj.call(x))
        end

        private

        def _activate(x)
          case @act_fn
          when "silu"
            MLX::NN.silu(x)
          when "relu"
            MLX::NN.relu(x)
          when "gelu"
            MLX::NN.gelu(x)
          when "gelu_new", "gelu_fast"
            MLX::NN.gelu_approx(x)
          else
            x
          end
        end
      end

      class LinearSubblockReplacement < MLX::NN::Module
        def initialize(hidden_size, bias)
          super()
          self.linear = MLX::NN::Linear.new(hidden_size, hidden_size, bias: bias)
        end

        def call(x, mask: nil, cache: nil)
          _ = mask
          _ = cache
          linear.call(x)
        end
      end

      class TransformerBlock < MLX::NN::Module
        def initialize(args, layer_idx)
          super()
          block_config = args.block_configs[layer_idx]
          @attention_config = block_config.attention
          @ffn_config = block_config.ffn

          self.input_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps) unless @attention_config.no_op
          self.self_attn = if @attention_config.no_op
            nil
          elsif @attention_config.replace_with_linear
            LinearSubblockReplacement.new(args.hidden_size, args.attention_bias)
          else
            Attention.new(args, @attention_config)
          end

          self.post_attention_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps) unless @ffn_config.no_op
          self.mlp = if @ffn_config.no_op
            nil
          elsif @ffn_config.replace_with_linear
            LinearSubblockReplacement.new(args.hidden_size, args.mlp_bias)
          else
            MLP.new(args, @ffn_config)
          end
        end

        def call(x, mask: nil, cache: nil)
          if self_attn
            residual = x
            h = input_layernorm.call(x)
            x = residual + self_attn.call(h, mask: mask, cache: cache)
          end

          if mlp
            residual = x
            h = post_attention_layernorm.call(x)
            x = residual + mlp.call(h)
          end

          x
        end
      end

      class NemotronNASModel < MLX::NN::Module
        attr_reader :num_attn_layers

        def initialize(args)
          super()
          self.embed_tokens = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
          self.layers = Array.new(args.num_hidden_layers) { |layer_idx| TransformerBlock.new(args, layer_idx) }
          self.norm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
          @num_attn_layers = layers.count { |layer| !layer.self_attn.nil? }
        end

        def call(inputs, cache: nil)
          h = embed_tokens.call(inputs)
          layer_cache = cache || [nil] * @num_attn_layers
          mask = _create_attention_mask(h, layer_cache[0])

          cache_idx = 0
          layers.each do |layer|
            layer_state = if layer.self_attn
              state = layer_cache[cache_idx]
              cache_idx += 1
              state
            end
            h = layer.call(h, mask: mask, cache: layer_state)
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
          self.model = NemotronNASModel.new(args)
          self.lm_head = MLX::NN::Linear.new(args.hidden_size, args.vocab_size, bias: false) unless args.tie_word_embeddings
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

        def make_cache
          layers.filter_map do |layer|
            MlxLm::KVCache.new if layer.self_attn
          end
        end
      end

      Models.register("nemotron-nas", Model, ModelArgs)
    end
  end
end
