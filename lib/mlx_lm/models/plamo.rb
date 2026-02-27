module MlxLm
  module Models
    module Plamo
      class ModelArgs < BaseModelArgs
        field :model_type, default: "plamo"
        field :hidden_size
        field :num_hidden_layers
        field :intermediate_size
        field :num_attention_heads
        field :rms_norm_eps
        field :vocab_size
        field :n_shared_head, default: 8
        field :rope_theta, default: 10_000.0
        field :rope_traditional, default: false
      end

      class Attention < MLX::NN::Module
        def initialize(config)
          super()
          @config = config
          @hidden_size = config.hidden_size
          @q_num_heads = config.num_attention_heads
          @head_dim = @hidden_size / @q_num_heads
          @qk_dim = @head_dim
          @v_dim = @head_dim
          @k_num_heads = (@q_num_heads.to_f / config.n_shared_head).ceil
          @v_num_heads = @k_num_heads
          @scale = @head_dim**(-0.5)

          self.q_proj = MLX::NN::Linear.new(@hidden_size, @q_num_heads * @qk_dim, bias: false)
          self.k_proj = MLX::NN::Linear.new(@hidden_size, @k_num_heads * @qk_dim, bias: false)
          self.v_proj = MLX::NN::Linear.new(@hidden_size, @v_num_heads * @v_dim, bias: false)
          self.o_proj = MLX::NN::Linear.new(@q_num_heads * @v_dim, @hidden_size, bias: false)
          self.rotary_emb = MLX::NN::RoPE.new(
            @head_dim,
            traditional: config.rope_traditional,
            base: config.rope_theta,
            scale: 1.0
          )
        end

        def call(hidden_states, attention_mask: nil, cache: nil)
          mx = MLX::Core
          bsz, q_len, _d = hidden_states.shape

          queries = q_proj.call(hidden_states)
          keys = k_proj.call(hidden_states)
          values = v_proj.call(hidden_states)

          queries = queries.reshape([bsz, q_len, @q_num_heads, @qk_dim]).transpose([0, 2, 1, 3])
          keys = keys.reshape([bsz, q_len, @k_num_heads, @qk_dim]).transpose([0, 2, 1, 3])
          values = values.reshape([bsz, q_len, @v_num_heads, @v_dim]).transpose([0, 2, 1, 3])

          if cache
            queries = rotary_emb.call(queries, offset: cache.offset)
            keys = rotary_emb.call(keys, offset: cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
          else
            queries = rotary_emb.call(queries)
            keys = rotary_emb.call(keys)
          end

          keys = mx.tile(keys, [1, @config.n_shared_head, 1, 1])
          values = mx.tile(values, [1, @config.n_shared_head, 1, 1])

          output = mx.scaled_dot_product_attention(
            queries,
            keys,
            values,
            @scale,
            attention_mask
          )
          output = output.transpose([0, 2, 1, 3]).reshape([bsz, q_len, @q_num_heads * @v_dim])
          o_proj.call(output)
        end
      end

      class MLP < MLX::NN::Module
        def initialize(config)
          super()
          self.gate_proj = MLX::NN::Linear.new(config.hidden_size, config.intermediate_size, bias: false)
          self.up_proj = MLX::NN::Linear.new(config.hidden_size, config.intermediate_size, bias: false)
          self.down_proj = MLX::NN::Linear.new(config.intermediate_size, config.hidden_size, bias: false)
        end

        def call(x)
          down_proj.call(Activations.swiglu(gate_proj.call(x), up_proj.call(x)))
        end
      end

      class DecoderLayer < MLX::NN::Module
        def initialize(config)
          super()
          self.self_attn = Attention.new(config)
          self.mlp = MLP.new(config)
          self.norm = MLX::NN::RMSNorm.new(config.hidden_size, eps: config.rms_norm_eps)
        end

        def call(hidden_states, attention_mask: nil, cache: nil)
          residual = hidden_states
          hidden_states = norm.call(hidden_states)

          hidden_states_sa = self_attn.call(
            hidden_states,
            attention_mask: attention_mask,
            cache: cache
          )
          hidden_states_mlp = mlp.call(hidden_states)

          residual + hidden_states_sa + hidden_states_mlp
        end
      end

      class PlamoModel < MLX::NN::Module
        def initialize(config)
          super()
          self.embed_tokens = MLX::NN::Embedding.new(config.vocab_size, config.hidden_size)
          self.layers = Array.new(config.num_hidden_layers) { DecoderLayer.new(config) }
          self.norm = MLX::NN::RMSNorm.new(config.hidden_size, eps: config.rms_norm_eps)
        end

        def call(inputs, cache: nil)
          h = embed_tokens.call(inputs)
          layer_cache = cache || [nil] * layers.length
          mask = _create_attention_mask(h, layer_cache[0])

          layers.each_with_index do |layer, i|
            h = layer.call(h, attention_mask: mask, cache: layer_cache[i])
          end

          norm.call(h)
        end

        private

        def _create_attention_mask(h, cache)
          return cache.make_mask(h.shape[1]) if cache && cache.respond_to?(:make_mask)
          return nil if h.shape[1] == 1

          "causal"
        end
      end

      class Model < MLX::NN::Module
        def initialize(args)
          super()
          self.model_type = args.model_type
          self.model = PlamoModel.new(args)
          self.lm_head = MLX::NN::Linear.new(args.hidden_size, args.vocab_size, bias: false)
        end

        def call(inputs, cache: nil)
          out = model.call(inputs, cache: cache)
          lm_head.call(out)
        end

        def sanitize(weights)
          weights.reject { |k, _| k.include?("self_attn.rotary_emb.inv_freq") }
        end

        def layers
          model.layers
        end
      end

      Models.register("plamo", Model, ModelArgs)
    end
  end
end
