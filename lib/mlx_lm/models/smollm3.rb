module MlxLm
  module Models
    module SmolLM3
      class ModelArgs < Llama::ModelArgs
        field :model_type, default: "smollm3"
        field :no_rope_layer_interval, default: 4
        field :no_rope_layers, default: nil

        def initialize(**kwargs)
          super

          if @no_rope_layers.nil?
            @no_rope_layers = Array.new(@num_hidden_layers) do |i|
              ((i + 1) % @no_rope_layer_interval).zero? ? 0 : 1
            end
          elsif @no_rope_layers.length != @num_hidden_layers
            raise ArgumentError, "`no_rope_layers` length mismatch"
          end
        end
      end

      class NoPE < MLX::NN::Module
        def call(x, offset: 0)
          x
        end
      end

      class Model < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.model_type = args.model_type
          self.model = Llama::LlamaModel.new(args)
          unless args.tie_word_embeddings
            self.lm_head = MLX::NN::Linear.new(args.hidden_size, args.vocab_size, bias: false)
          end

          args.no_rope_layers.each_with_index do |use_rope, idx|
            next if use_rope && use_rope != 0

            model.layers[idx].self_attn.rope = NoPE.new
          end
        end

        def call(inputs, cache: nil, input_embeddings: nil)
          out = if input_embeddings.nil?
            model.call(inputs, cache: cache)
          else
            _call_with_input_embeddings(input_embeddings, cache)
          end

          if @args.tie_word_embeddings
            model.embed_tokens.as_linear(out)
          else
            lm_head.call(out)
          end
        end

        def layers
          model.layers
        end

        def sanitize(weights)
          result = weights.reject { |k, _| k.include?("self_attn.rotary_emb.inv_freq") }
          result.delete("lm_head.weight") if @args.tie_word_embeddings
          result
        end

        private

        def _call_with_input_embeddings(input_embeddings, cache)
          h = input_embeddings
          layer_cache = cache || [nil] * model.layers.length

          mask = nil
          mask = "causal" if h.shape[1] > 1

          model.layers.each_with_index do |layer, i|
            h = layer.call(h, mask: mask, cache: layer_cache[i])
          end

          model.norm.call(h)
        end
      end

      Models.register("smollm3", Model, ModelArgs)
    end
  end
end
