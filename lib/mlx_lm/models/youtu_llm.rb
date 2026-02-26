module MlxLm
  module Models
    module YoutuLLM
      class ModelArgs < BaseModelArgs
        field :model_type, default: "youtu_llm"
        field :vocab_size, default: 128_256
        field :hidden_size, default: 2048
        field :intermediate_size, default: 6144
        field :num_hidden_layers, default: 32
        field :num_attention_heads, default: 16
        field :num_key_value_heads, default: 16
        field :kv_lora_rank, default: 512
        field :q_lora_rank, default: 1536
        field :qk_rope_head_dim, default: 64
        field :v_head_dim, default: 128
        field :qk_nope_head_dim, default: 128
        field :max_position_embeddings, default: 131_072
        field :rms_norm_eps, default: 1e-6
        field :rope_theta, default: 1_600_000.0
        field :rope_traditional, default: true
        field :rope_scaling, default: nil
        field :attention_bias, default: false
        field :mlp_bias, default: false
        field :tie_word_embeddings, default: true
      end

      class YoutuLLMAttention < MLX::NN::Module
        def initialize(config)
          super()
          @hidden_size = config.hidden_size
          @num_heads = config.num_attention_heads
          @q_lora_rank = config.q_lora_rank
          @qk_rope_head_dim = config.qk_rope_head_dim
          @kv_lora_rank = config.kv_lora_rank
          @v_head_dim = config.v_head_dim
          @qk_nope_head_dim = config.qk_nope_head_dim
          @q_head_dim = @qk_nope_head_dim + @qk_rope_head_dim
          @kv_head_dim = @qk_nope_head_dim + @v_head_dim
          @scale = @q_head_dim**(-0.5)

          if @q_lora_rank.nil?
            self.q_proj = MLX::NN::Linear.new(
              @hidden_size,
              @num_heads * @q_head_dim,
              bias: false
            )
          else
            self.q_a_proj = MLX::NN::Linear.new(
              @hidden_size,
              @q_lora_rank,
              bias: config.attention_bias
            )
            self.q_a_layernorm = MLX::NN::RMSNorm.new(@q_lora_rank, eps: config.rms_norm_eps)
            self.q_b_proj = MLX::NN::Linear.new(@q_lora_rank, @num_heads * @q_head_dim, bias: false)
          end

          self.kv_a_proj_with_mqa = MLX::NN::Linear.new(
            @hidden_size,
            @kv_lora_rank + @qk_rope_head_dim,
            bias: config.attention_bias
          )
          self.kv_a_layernorm = MLX::NN::RMSNorm.new(@kv_lora_rank, eps: config.rms_norm_eps)
          self.kv_b_proj = MLX::NN::Linear.new(
            @kv_lora_rank,
            @num_heads * (@q_head_dim - @qk_rope_head_dim + @v_head_dim),
            bias: false
          )

          self.o_proj = MLX::NN::Linear.new(
            @num_heads * @v_head_dim,
            @hidden_size,
            bias: config.attention_bias
          )

          self.rope = MlxLm::Models.initialize_rope(
            @qk_rope_head_dim,
            config.rope_theta,
            config.rope_traditional,
            config.rope_scaling,
            max_position_embeddings: config.max_position_embeddings
          )
        end

        def call(x, mask: nil, cache: nil)
          mx = MLX::Core
          b, l, _d = x.shape

          q = if @q_lora_rank.nil?
            q_proj.call(x)
          else
            q_b_proj.call(q_a_layernorm.call(q_a_proj.call(x)))
          end

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
            k_pe = mx.repeat(k_pe, @num_heads, 1)
            keys, values = cache.update_and_fetch(mx.concatenate([k_nope, k_pe], -1), values)
          else
            q_pe = rope.call(q_pe)
            k_pe = rope.call(k_pe)
            k_pe = mx.repeat(k_pe, @num_heads, 1)
            keys = mx.concatenate([k_nope, k_pe], -1)
          end

          queries = mx.concatenate([q_nope, q_pe], -1)
          output = mx.scaled_dot_product_attention(queries, keys, values, @scale, mask)
          output = output.transpose([0, 2, 1, 3]).reshape([b, l, @num_heads * @v_head_dim])
          o_proj.call(output)
        end
      end

      class YoutuLLMMLP < MLX::NN::Module
        def initialize(config)
          super()
          self.gate_proj = MLX::NN::Linear.new(
            config.hidden_size,
            config.intermediate_size,
            bias: config.mlp_bias
          )
          self.up_proj = MLX::NN::Linear.new(
            config.hidden_size,
            config.intermediate_size,
            bias: config.mlp_bias
          )
          self.down_proj = MLX::NN::Linear.new(
            config.intermediate_size,
            config.hidden_size,
            bias: config.mlp_bias
          )
        end

        def call(x)
          down_proj.call(Activations.swiglu(gate_proj.call(x), up_proj.call(x)))
        end
      end

      class YoutuLLMDecoderLayer < MLX::NN::Module
        def initialize(config)
          super()
          self.self_attn = YoutuLLMAttention.new(config)
          self.mlp = YoutuLLMMLP.new(config)
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

      class YoutuLLMModel < MLX::NN::Module
        def initialize(config)
          super()
          self.embed_tokens = MLX::NN::Embedding.new(config.vocab_size, config.hidden_size)
          self.layers = Array.new(config.num_hidden_layers) { YoutuLLMDecoderLayer.new(config) }
          self.norm = MLX::NN::RMSNorm.new(config.hidden_size, eps: config.rms_norm_eps)
        end

        def call(inputs, cache: nil)
          h = embed_tokens.call(inputs)
          layer_cache = cache || [nil] * layers.length
          mask = _create_attention_mask(h, layer_cache[0])

          layers.each_with_index do |layer, i|
            h = layer.call(h, mask: mask, cache: layer_cache[i])
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
        def initialize(config)
          super()
          @config = config
          self.model_type = config.model_type
          self.model = YoutuLLMModel.new(config)
          unless config.tie_word_embeddings
            self.lm_head = MLX::NN::Linear.new(config.hidden_size, config.vocab_size, bias: false)
          end
        end

        def call(inputs, cache: nil)
          out = model.call(inputs, cache: cache)
          if @config.tie_word_embeddings
            model.embed_tokens.as_linear(out)
          else
            lm_head.call(out)
          end
        end

        def sanitize(weights)
          result = weights.dup
          result.delete("lm_head.weight") if @config.tie_word_embeddings
          result
        end

        def layers
          model.layers
        end
      end

      Models.register("youtu_llm", Model, ModelArgs)
    end
  end
end
