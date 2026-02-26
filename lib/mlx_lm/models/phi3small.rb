module MlxLm
  module Models
    module Phi3small
      class ModelArgs < BaseModelArgs
        field :model_type, default: "phi3small"
        field :hidden_size
        field :dense_attention_every_n_layers
        field :ff_intermediate_size
        field :gegelu_limit
        field :num_hidden_layers
        field :num_attention_heads
        field :layer_norm_epsilon
        field :vocab_size
        field :num_key_value_heads
        field :mup_attn_multiplier, default: 1.0
        field :mup_use_scaling, default: true
        field :mup_embedding_multiplier, default: 10.0
        field :mup_width_multiplier, default: 8.0
        field :rope_embedding_base, default: 1_000_000.0
        field :rope_position_scale, default: 1.0
        field :blocksparse_block_size, default: 64
        field :blocksparse_num_local_blocks, default: 16
        field :blocksparse_vert_stride, default: 8

        def initialize(**kwargs)
          super
          @num_key_value_heads ||= @num_attention_heads
        end
      end

      class Attention < MLX::NN::Module
        def initialize(args, layer_idx)
          super()

          dim = args.hidden_size
          @n_heads = args.num_attention_heads
          @n_kv_heads = args.num_key_value_heads
          @n_q_per_kv = @n_heads / @n_kv_heads
          @head_dim = dim / @n_heads

          self.query_key_value = MLX::NN::Linear.new(
            dim,
            (@n_heads + 2 * @n_kv_heads) * @head_dim
          )
          self.dense = MLX::NN::Linear.new(dim, dim)

          norm_factor = if args.mup_use_scaling
            @head_dim / args.mup_attn_multiplier.to_f
          else
            Math.sqrt(@head_dim)
          end
          @scale = 1.0 / norm_factor

          self.rope = MLX::NN::RoPE.new(
            @head_dim,
            traditional: false,
            base: args.rope_embedding_base,
            scale: args.rope_position_scale
          )

          @block_sparse = (layer_idx % args.dense_attention_every_n_layers).zero?
        end

        def call(x, mask: nil, cache: nil)
          mx = MLX::Core
          b, l, _d = x.shape

          qkv = query_key_value.call(x)
          q_size = @n_heads * @head_dim
          k_size = @n_kv_heads * @head_dim

          queries = mx.split(qkv, [q_size, q_size + k_size], -1)[0]
          keys = mx.split(qkv, [q_size, q_size + k_size], -1)[1]
          values = mx.split(qkv, [q_size + k_size], -1)[1]

          queries = queries.reshape([b, l, @n_heads, @head_dim]).transpose([0, 2, 1, 3])
          keys = keys.reshape([b, l, @n_kv_heads, @head_dim]).transpose([0, 2, 1, 3])
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
          dense.call(output)
        end
      end

      class MLP < MLX::NN::Module
        def initialize(args)
          super()
          dim = args.hidden_size
          @hidden_dim = args.ff_intermediate_size
          self.up_proj = MLX::NN::Linear.new(dim, 2 * @hidden_dim)
          self.down_proj = MLX::NN::Linear.new(@hidden_dim, dim)
        end

        def call(x)
          mx = MLX::Core
          x = up_proj.call(x)
          a_gelu, a_linear = mx.split(x, [@hidden_dim], -1)
          out_gelu = a_gelu * mx.sigmoid(1.702 * a_gelu)
          down_proj.call(out_gelu * (a_linear + 1.0))
        end
      end

      class TransformerBlock < MLX::NN::Module
        def initialize(args, layer_idx)
          super()
          self.self_attn = Attention.new(args, layer_idx)
          self.mlp = MLP.new(args)
          self.input_layernorm = MLX::NN::LayerNorm.new(args.hidden_size, eps: args.layer_norm_epsilon)
          self.post_attention_layernorm = MLX::NN::LayerNorm.new(args.hidden_size, eps: args.layer_norm_epsilon)
        end

        def call(x, mask: nil, cache: nil)
          r = self_attn.call(input_layernorm.call(x), mask: mask, cache: cache)
          h = x + r
          r = mlp.call(post_attention_layernorm.call(h))
          h + r
        end
      end

      class Phi3Model < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.embed_tokens = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
          self.layers = Array.new(args.num_hidden_layers) { |layer_idx| TransformerBlock.new(args, layer_idx) }
          self.final_layernorm = MLX::NN::LayerNorm.new(args.hidden_size, eps: args.layer_norm_epsilon)
        end

        def call(inputs, cache: nil)
          h = embed_tokens.call(inputs)
          h = h * @args.mup_embedding_multiplier if @args.mup_embedding_multiplier

          layer_cache = cache || [nil] * layers.length
          mask = _create_attention_mask(h, layer_cache[0])

          layers.each_with_index do |layer, layer_idx|
            h = layer.call(h, mask: mask, cache: layer_cache[layer_idx])
          end

          final_layernorm.call(h)
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
          self.model = Phi3Model.new(args)
        end

        def call(inputs, cache: nil)
          out = model.call(inputs, cache: cache)
          out = model.embed_tokens.as_linear(out)
          out = out / @args.mup_width_multiplier if @args.mup_width_multiplier
          out
        end

        def sanitize(weights)
          weights.reject do |key, _|
            key_name = key.to_s
            key_name.include?("self_attn.rotary_emb.inv_freq") ||
              key_name.include?("rotary_emb.inv_freq") ||
              key_name.include?("position_embeddings.inv_freq")
          end
        end

        def layers
          model.layers
        end
      end

      Models.register("phi3small", Model, ModelArgs)
    end
  end
end
