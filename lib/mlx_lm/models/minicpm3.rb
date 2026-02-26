require_relative "activations"
require_relative "rope_utils"

module MlxLm
  module Models
    module MiniCPM3
      class ModelArgs < BaseModelArgs
        field :model_type, default: "minicpm3"
        field :hidden_size
        field :dim_model_base
        field :num_hidden_layers
        field :intermediate_size
        field :num_attention_heads
        field :rms_norm_eps
        field :vocab_size
        field :num_key_value_heads
        field :q_lora_rank
        field :qk_nope_head_dim
        field :qk_rope_head_dim
        field :kv_lora_rank
        field :scale_depth
        field :scale_emb
        field :max_position_embeddings
        field :attention_bias, default: false
        field :rope_theta, default: 1_000_000.0
        field :rope_traditional, default: false
        field :rope_scaling, default: nil
        field :tie_word_embeddings, default: false

        def initialize(**kwargs)
          super
          @num_key_value_heads ||= @num_attention_heads
          @rope_scaling ||= {}
        end
      end

      class Attention < MLX::NN::Module
        def initialize(args)
          super()
          @qk_rope_head_dim = args.qk_rope_head_dim
          @qk_nope_head_dim = args.qk_nope_head_dim
          @kv_lora_rank = args.kv_lora_rank
          @num_heads = args.num_attention_heads
          @hidden_size = args.hidden_size
          @v_head_dim = @hidden_size / @num_heads
          @q_head_dim = @qk_nope_head_dim + @qk_rope_head_dim
          @kv_head_dim = @qk_nope_head_dim + @v_head_dim
          @softmax_scale = @q_head_dim**(-0.5)

          self.q_a_proj = MLX::NN::Linear.new(
            @hidden_size,
            args.q_lora_rank,
            bias: args.attention_bias
          )
          self.q_a_layernorm = MLX::NN::RMSNorm.new(args.q_lora_rank, eps: args.rms_norm_eps)
          self.q_b_proj = MLX::NN::Linear.new(
            args.q_lora_rank,
            @num_heads * @q_head_dim,
            bias: false
          )

          self.kv_a_proj_with_mqa = MLX::NN::Linear.new(
            @hidden_size,
            @kv_lora_rank + @qk_rope_head_dim,
            bias: args.attention_bias
          )
          self.kv_a_layernorm = MLX::NN::RMSNorm.new(@kv_lora_rank, eps: args.rms_norm_eps)
          self.kv_b_proj = MLX::NN::Linear.new(
            @kv_lora_rank,
            @num_heads * @kv_head_dim,
            bias: false
          )

          self.o_proj = MLX::NN::Linear.new(
            @num_heads * @v_head_dim,
            @hidden_size,
            bias: args.attention_bias
          )

          self.rope = SuScaledRoPE.new(
            @qk_rope_head_dim,
            base: args.rope_theta,
            max_position_embeddings: args.max_position_embeddings,
            original_max_position_embeddings: scaling_value(args.rope_scaling, "original_max_position_embeddings", 4096),
            short_factor: scaling_value(args.rope_scaling, "short_factor", 1.0),
            long_factor: scaling_value(args.rope_scaling, "long_factor", 1.0)
          )
        end

        def call(x, mask: nil, cache: nil)
          mx = MLX::Core
          b, l, _ = x.shape

          q = q_b_proj.call(q_a_layernorm.call(q_a_proj.call(x)))
          q = q.reshape([b, l, @num_heads, @q_head_dim]).transpose([0, 2, 1, 3])
          q_nope, q_pe = mx.split(q, [@qk_nope_head_dim], -1)

          compressed_kv = kv_a_proj_with_mqa.call(x)
          compressed_kv, k_pe = mx.split(compressed_kv, [@kv_lora_rank], -1)
          k_pe = k_pe.reshape([b, l, 1, @qk_rope_head_dim]).transpose([0, 2, 1, 3])

          kv = kv_b_proj.call(kv_a_layernorm.call(compressed_kv))
          kv = kv.reshape([b, l, @num_heads, @kv_head_dim]).transpose([0, 2, 1, 3])
          k_nope, values = mx.split(kv, [@qk_nope_head_dim], -1)

          if cache
            q_pe = rope.call(q_pe, offset: cache.offset)
            k_pe = rope.call(k_pe, offset: cache.offset)
          else
            q_pe = rope.call(q_pe)
            k_pe = rope.call(k_pe)
          end

          k_pe_broadcasted = mx.broadcast_to(k_pe, [b, @num_heads, l, @qk_rope_head_dim])
          queries = mx.concatenate([q_nope, q_pe], -1)
          keys = mx.concatenate([k_nope, k_pe_broadcasted], -1)

          if cache
            keys, values = cache.update_and_fetch(keys, values)
          end

          output = mx.scaled_dot_product_attention(queries, keys, values, @softmax_scale, mask)
          output = output.transpose([0, 2, 1, 3]).reshape([b, l, @num_heads * @v_head_dim])
          o_proj.call(output)
        end

        private

        def scaling_value(config, key, default)
          return default if config.nil?
          return config[key] if config.key?(key)

          config.fetch(key.to_sym, default)
        end
      end

      class MLP < MLX::NN::Module
        def initialize(args)
          super()
          self.gate_proj = MLX::NN::Linear.new(args.hidden_size, args.intermediate_size, bias: false)
          self.up_proj = MLX::NN::Linear.new(args.hidden_size, args.intermediate_size, bias: false)
          self.down_proj = MLX::NN::Linear.new(args.intermediate_size, args.hidden_size, bias: false)
        end

        def call(x)
          down_proj.call(Activations.swiglu(gate_proj.call(x), up_proj.call(x)))
        end
      end

      class DecoderLayer < MLX::NN::Module
        def initialize(args)
          super()
          self.self_attn = Attention.new(args)
          self.mlp = MLP.new(args)
          self.input_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
          self.post_attention_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
          @residual_scale = args.scale_depth / Math.sqrt(args.num_hidden_layers)
        end

        def call(x, mask: nil, cache: nil)
          r = self_attn.call(input_layernorm.call(x), mask: mask, cache: cache)
          h = x + r * @residual_scale
          r = mlp.call(post_attention_layernorm.call(h))
          h + r * @residual_scale
        end
      end

      class MiniCPM3Model < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.embed_tokens = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
          self.layers = Array.new(args.num_hidden_layers) { DecoderLayer.new(args) }
          self.norm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
        end

        def call(inputs, mask: nil, cache: nil)
          h = embed_tokens.call(inputs) * @args.scale_emb
          layer_cache = cache || [nil] * layers.length
          local_mask = mask || _create_attention_mask(h, layer_cache[0])

          layers.each_with_index do |layer, i|
            h = layer.call(h, mask: local_mask, cache: layer_cache[i])
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
          self.model_type = args.model_type
          self.model = MiniCPM3Model.new(args)
          self.lm_head = MLX::NN::Linear.new(args.hidden_size, args.vocab_size, bias: false) unless args.tie_word_embeddings
        end

        def call(inputs, mask: nil, cache: nil)
          out = model.call(inputs, mask: mask, cache: cache)
          if @args.tie_word_embeddings
            model.embed_tokens.as_linear(out)
          else
            lm_head.call(out / (@args.hidden_size.to_f / @args.dim_model_base))
          end
        end

        def sanitize(weights)
          result = weights.reject { |k, _| k.include?("self_attn.rotary_emb.inv_freq") }

          if @args.tie_word_embeddings
            result.delete("lm_head.weight")
          elsif !result.key?("lm_head.weight") && result.key?("model.embed_tokens.weight")
            result["lm_head.weight"] = result["model.embed_tokens.weight"]
          end

          result
        end

        def layers
          model.layers
        end
      end

      Models.register("minicpm3", Model, ModelArgs)
    end
  end
end
