module MlxLm
  module Models
    module Llama4Text
      class ModelArgs < BaseModelArgs
        field :model_type, default: "llama4_text"
        field :hidden_size
        field :num_attention_heads
        field :num_hidden_layers
        field :vocab_size
        field :intermediate_size, default: nil
        field :intermediate_size_mlp, default: nil
        field :num_key_value_heads, default: nil
        field :rms_norm_eps, default: 1e-5
        field :rope_theta, default: 10_000.0
        field :head_dim, default: nil
        field :tie_word_embeddings, default: true
        field :no_rope_layers, default: nil
        field :use_qk_norm, default: false

        def initialize(**kwargs)
          super
          @num_key_value_heads ||= @num_attention_heads
          @head_dim ||= @hidden_size / @num_attention_heads
          @intermediate_size_mlp ||= @intermediate_size

          if @no_rope_layers.nil?
            @no_rope_layers = Array.new(@num_hidden_layers, 1)
          elsif @no_rope_layers.length != @num_hidden_layers
            raise ArgumentError, "`no_rope_layers` length mismatch"
          end
        end
      end

      class Attention < MLX::NN::Module
        def initialize(args, use_rope)
          super()
          @n_heads = args.num_attention_heads
          @n_kv_heads = args.num_key_value_heads
          @head_dim = args.head_dim
          @scale = @head_dim**(-0.5)
          @use_rope = !!use_rope
          @use_qk_norm = !!args.use_qk_norm
          @rms_norm_eps = args.rms_norm_eps

          self.q_proj = MLX::NN::Linear.new(args.hidden_size, @n_heads * @head_dim, bias: false)
          self.k_proj = MLX::NN::Linear.new(args.hidden_size, @n_kv_heads * @head_dim, bias: false)
          self.v_proj = MLX::NN::Linear.new(args.hidden_size, @n_kv_heads * @head_dim, bias: false)
          self.o_proj = MLX::NN::Linear.new(@n_heads * @head_dim, args.hidden_size, bias: false)

          if @use_rope
            self.rope = MLX::NN::RoPE.new(@head_dim, traditional: true, base: args.rope_theta)
          end
        end

        def call(x, mask: nil, cache: nil)
          mx = MLX::Core
          b, l, _d = x.shape

          queries = q_proj.call(x)
          keys = k_proj.call(x)
          values = v_proj.call(x)

          queries = queries.reshape([b, l, @n_heads, @head_dim])
          keys = keys.reshape([b, l, @n_kv_heads, @head_dim])

          if @use_qk_norm
            queries = mx.rms_norm(queries, nil, @rms_norm_eps)
            keys = mx.rms_norm(keys, nil, @rms_norm_eps)
          end

          queries = queries.transpose([0, 2, 1, 3])
          keys = keys.transpose([0, 2, 1, 3])
          values = values.reshape([b, l, @n_kv_heads, @head_dim]).transpose([0, 2, 1, 3])

          if @use_rope
            if cache
              queries = rope.call(queries, offset: cache.offset)
              keys = rope.call(keys, offset: cache.offset)
            else
              queries = rope.call(queries)
              keys = rope.call(keys)
            end
          end

          keys, values = cache.update_and_fetch(keys, values) if cache

          output = mx.scaled_dot_product_attention(queries, keys, values, @scale, mask)
          output = output.transpose([0, 2, 1, 3]).reshape([b, l, @n_heads * @head_dim])
          o_proj.call(output)
        end
      end

      class MLP < MLX::NN::Module
        def initialize(dim, intermediate_size)
          super()
          self.gate_proj = MLX::NN::Linear.new(dim, intermediate_size, bias: false)
          self.up_proj = MLX::NN::Linear.new(dim, intermediate_size, bias: false)
          self.down_proj = MLX::NN::Linear.new(intermediate_size, dim, bias: false)
        end

        def call(x)
          down_proj.call(Activations.swiglu(gate_proj.call(x), up_proj.call(x)))
        end
      end

      class TransformerBlock < MLX::NN::Module
        def initialize(args, use_rope)
          super()
          self.self_attn = Attention.new(args, use_rope)
          self.feed_forward = MLP.new(args.hidden_size, args.intermediate_size_mlp)
          self.input_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
          self.post_attention_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
        end

        def call(x, mask: nil, cache: nil)
          r = self_attn.call(input_layernorm.call(x), mask: mask, cache: cache)
          h = x + r
          r = feed_forward.call(post_attention_layernorm.call(h))
          h + r
        end
      end

      class LanguageModel < MLX::NN::Module
        def initialize(args)
          super()
          self.embed_tokens = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
          self.layers = Array.new(args.num_hidden_layers) do |i|
            TransformerBlock.new(args, args.no_rope_layers[i])
          end
          self.norm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
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
        def initialize(args)
          super()
          @args = args
          self.model_type = args.model_type
          self.model = LanguageModel.new(args)
          self.output = nil
          unless args.tie_word_embeddings
            self.output = MLX::NN::Linear.new(args.hidden_size, args.vocab_size, bias: false)
          end
        end

        def call(inputs, cache: nil)
          h = model.call(inputs, cache: cache)
          if @args.tie_word_embeddings
            model.embed_tokens.as_linear(h)
          else
            output.call(h)
          end
        end

        def sanitize(weights)
          sanitized = weights.reject do |k, _|
            k.include?("self_attn.rotary_emb.inv_freq") || k.include?("self_attn.rope.inv_freq")
          end
          if @args.tie_word_embeddings
            sanitized.delete("output.weight")
            sanitized.delete("lm_head.weight")
          end
          sanitized
        end

        def layers
          model.layers
        end
      end

      Models.register("llama4_text", Model, ModelArgs)
    end
  end
end
