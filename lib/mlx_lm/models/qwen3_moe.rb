require_relative "qwen3"
require_relative "switch_layers"

module MlxLm
  module Models
    module Qwen3Moe
      class ModelArgs < Qwen3::ModelArgs
        field :model_type, default: "qwen3_moe"
        field :num_experts, default: 128
        field :num_experts_per_tok, default: 8
        field :decoder_sparse_step, default: 1
        field :mlp_only_layers, default: []
        field :moe_intermediate_size, default: 1408
        field :norm_topk_prob, default: false

        def initialize(**kwargs)
          super
          @mlp_only_layers ||= []
        end
      end

      class SparseMoeBlock < MLX::NN::Module
        def initialize(args)
          super()
          @top_k = [args.num_experts_per_tok.to_i, 1].max
          @num_experts = args.num_experts
          @norm_topk_prob = args.norm_topk_prob

          dim = args.hidden_size
          hidden_dim = args.moe_intermediate_size

          self.gate = MLX::NN::Linear.new(dim, @num_experts, bias: false)
          self.switch_mlp = SwitchLayers::SwitchGLU.new(dim, hidden_dim, @num_experts)
        end

        def call(x)
          mx = MLX::Core

          gates = gate.call(x)
          gates = mx.softmax(gates.astype(mx.float32), -1).astype(gates.dtype)

          k = [@top_k, @num_experts].min
          inds = mx.stop_gradient(mx.argpartition(gates * -1.0, k - 1, -1))
          take_ids = mx.array((0...k).to_a, dtype: mx.int32)
          inds = mx.take(inds, take_ids, -1)
          scores = mx.take_along_axis(gates, inds, -1)

          if @norm_topk_prob
            denom = mx.expand_dims(mx.sum(scores, -1), -1)
            scores = scores / denom
          end

          y = switch_mlp.call(x, inds)
          mx.sum(y * mx.expand_dims(scores, -1), -2)
        end
      end

      class DecoderLayer < MLX::NN::Module
        def initialize(args, layer_idx)
          super()
          self.self_attn = Qwen3::Attention.new(args)
          self.input_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
          self.post_attention_layernorm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)

          if _use_sparse_moe_layer?(args, layer_idx)
            self.mlp = SparseMoeBlock.new(args)
          else
            self.mlp = Qwen3::MLP.new(args.hidden_size, args.intermediate_size)
          end
        end

        def call(x, mask: nil, cache: nil)
          r = self_attn.call(input_layernorm.call(x), mask: mask, cache: cache)
          h = x + r
          r = mlp.call(post_attention_layernorm.call(h))
          h + r
        end

        private

        def _use_sparse_moe_layer?(args, layer_idx)
          sparse_step = [args.decoder_sparse_step.to_i, 1].max
          mlp_only_layers = args.mlp_only_layers || []

          !mlp_only_layers.include?(layer_idx) &&
            args.num_experts.to_i > 0 &&
            ((layer_idx + 1) % sparse_step).zero?
        end
      end

      class Qwen3MoeModel < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.embed_tokens = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
          self.layers = Array.new(args.num_hidden_layers) { |layer_idx| DecoderLayer.new(args, layer_idx) }
          self.norm = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.rms_norm_eps)
        end

        def call(inputs, cache: nil, input_embeddings: nil)
          h = input_embeddings || embed_tokens.call(inputs)
          layer_cache = cache || [nil] * layers.length

          mask = nil
          mask = "causal" if h.shape[1] > 1

          layers.each_with_index do |layer, layer_idx|
            h = layer.call(h, mask: mask, cache: layer_cache[layer_idx])
          end

          norm.call(h)
        end
      end

      class Model < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.model_type = args.model_type
          self.model = Qwen3MoeModel.new(args)
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
          mx = MLX::Core

          result = weights.dup
          result.delete("lm_head.weight") if @args.tie_word_embeddings
          return result unless result.key?("model.layers.0.mlp.experts.0.up_proj.weight")

          @args.num_hidden_layers.times do |layer_idx|
            prefix = "model.layers.#{layer_idx}.mlp"
            %w[up_proj down_proj gate_proj].each do |projection|
              expert_keys = (0...@args.num_experts).map do |expert_idx|
                "#{prefix}.experts.#{expert_idx}.#{projection}.weight"
              end
              next unless expert_keys.all? { |key| result.key?(key) }

              stacked = expert_keys.map { |key| result.delete(key) }
              result["#{prefix}.switch_mlp.#{projection}.weight"] = mx.stack(stacked)
            end
          end

          result
        end

        def layers
          model.layers
        end
      end

      Models.register("qwen3_moe", Model, ModelArgs)
    end
  end
end
