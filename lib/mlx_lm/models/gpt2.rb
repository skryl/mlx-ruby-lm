module MlxLm
  module Models
    module GPT2
      class ModelArgs < BaseModelArgs
        field :model_type, default: "gpt2"
        field :n_ctx
        field :n_embd
        field :n_head
        field :n_layer
        field :n_positions
        field :layer_norm_epsilon
        field :vocab_size
        field :num_key_value_heads, default: nil

        def initialize(**kwargs)
          super
          @num_key_value_heads ||= @n_head
        end
      end

      class Attention < MLX::NN::Module
        def initialize(args)
          super()
          unless (args.n_embd % args.n_head).zero?
            raise ArgumentError, "n_embd must be divisible by n_head"
          end

          @n_embd = args.n_embd
          @n_head = args.n_head
          @head_dim = @n_embd / @n_head
          @scale = @head_dim**(-0.5)

          self.c_attn = MLX::NN::Linear.new(@n_embd, 3 * @n_embd, bias: true)
          self.c_proj = MLX::NN::Linear.new(@n_embd, @n_embd, bias: true)
        end

        def call(x, mask: nil, cache: nil)
          mx = MLX::Core
          b, l, _d = x.shape

          qkv = c_attn.call(x)
          queries, keys, values = mx.split(qkv, 3, 2)

          queries = queries.reshape([b, l, @n_head, @head_dim]).transpose([0, 2, 1, 3])
          keys = keys.reshape([b, l, @n_head, @head_dim]).transpose([0, 2, 1, 3])
          values = values.reshape([b, l, @n_head, @head_dim]).transpose([0, 2, 1, 3])

          if cache
            keys, values = cache.update_and_fetch(keys, values)
          end

          output = mx.scaled_dot_product_attention(queries, keys, values, @scale, mask)
          output = output.transpose([0, 2, 1, 3]).reshape([b, l, @n_embd])
          c_proj.call(output)
        end
      end

      class MLP < MLX::NN::Module
        def initialize(args)
          super()
          self.c_fc = MLX::NN::Linear.new(args.n_embd, 4 * args.n_embd, bias: true)
          self.c_proj = MLX::NN::Linear.new(4 * args.n_embd, args.n_embd, bias: true)
        end

        def call(x)
          c_proj.call(MLX::NN.gelu_approx(c_fc.call(x)))
        end
      end

      class TransformerBlock < MLX::NN::Module
        def initialize(args)
          super()
          self.attn = Attention.new(args)
          self.mlp = MLP.new(args)
          self.ln_1 = MLX::NN::LayerNorm.new(args.n_embd, eps: args.layer_norm_epsilon)
          self.ln_2 = MLX::NN::LayerNorm.new(args.n_embd, eps: args.layer_norm_epsilon)
        end

        def call(x, mask: nil, cache: nil)
          r = attn.call(ln_1.call(x), mask: mask, cache: cache)
          h = x + r
          r = mlp.call(ln_2.call(h))
          h + r
        end
      end

      class GPT2Model < MLX::NN::Module
        def initialize(args)
          super()
          self.wte = MLX::NN::Embedding.new(args.vocab_size, args.n_embd)
          self.wpe = MLX::NN::Embedding.new(args.n_positions, args.n_embd)
          self.h = Array.new(args.n_layer) { TransformerBlock.new(args) }
          self.ln_f = MLX::NN::LayerNorm.new(args.n_embd, eps: args.layer_norm_epsilon)
        end

        def call(inputs, cache: nil)
          mx = MLX::Core
          _b, l = inputs.shape

          hidden_states = wte.call(inputs)
          layer_cache = cache || [nil] * h.length
          offset = layer_cache[0] ? layer_cache[0].offset : 0
          position_ids = mx.add(mx.arange(0, l, 1, mx.int32), offset)
          hidden_states = hidden_states + wpe.call(position_ids)

          mask = _create_attention_mask(hidden_states, layer_cache[0])
          h.each_with_index do |layer, i|
            hidden_states = layer.call(hidden_states, mask: mask, cache: layer_cache[i])
          end
          ln_f.call(hidden_states)
        end

        private

        def _create_attention_mask(h, cache)
          return cache.make_mask(h.shape[1]) if cache && cache.respond_to?(:make_mask)
          return nil if h.shape[1] == 1

          "causal"
        end
      end

      class Model < MLX::NN::Module
        attr_reader :args

        def initialize(args)
          super()
          @args = args
          self.model_type = args.model_type
          self.model = GPT2Model.new(args)
        end

        def call(inputs, cache: nil)
          out = model.call(inputs, cache: cache)
          model.wte.as_linear(out)
        end

        def sanitize(weights)
          result = {}
          weights.each do |k, v|
            next if k.match?(/\Ah\.\d+\.attn\.bias\z/)

            value = if k.match?(/\Ah\.\d+\.(attn\.c_attn|attn\.c_proj|mlp\.c_fc|mlp\.c_proj)\.weight\z/)
              v.transpose([1, 0])
            else
              v
            end

            if k.start_with?("model.")
              result[k] = value
            else
              result["model.#{k}"] = value
            end
          end
          result
        end

        def layers
          model.h
        end
      end

      Models.register("gpt2", Model, ModelArgs)
    end
  end
end
