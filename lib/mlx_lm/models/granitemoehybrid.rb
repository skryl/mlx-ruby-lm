require_relative "falcon_h1"

module MlxLm
  module Models
    module GraniteMoeHybrid
      class ModelArgs < FalconH1::ModelArgs
        field :model_type, default: "granitemoehybrid"
        field :embedding_multiplier, default: 1.0
        field :attention_multiplier, default: 1.0
        field :logits_scaling, default: 1.0
        field :residual_multiplier, default: 1.0
        field :num_local_experts, default: nil
        field :num_experts_per_tok, default: nil
        field :shared_intermediate_size, default: nil
        field :mamba_n_heads, default: nil
        field :mamba_d_head, default: nil
        field :mamba_proj_bias, default: false
        field :mamba_d_state, default: nil
        field :mamba_n_groups, default: nil
        field :mamba_conv_bias, default: false
        field :layer_types, default: nil
        field :position_embedding_type, default: "rope"
        field :time_step_limit, default: [0.001, 100.0]
        field :mlp_bias, default: false

        def initialize(**kwargs)
          super
          @num_hidden_layers ||= Array(@layer_types).length
          @num_attention_heads ||= @mamba_n_heads
          @num_key_value_heads ||= @num_attention_heads
          @head_dim ||= @mamba_d_head
          @mamba_d_conv ||= 4
          @layer_types ||= _default_layer_types
          @block_types ||= _to_block_types
        end

        def to_falcon_h1_dict
          hidden_size = @hidden_size
          attention_heads = @num_attention_heads
          inferred_head_dim = if !@head_dim.nil?
            @head_dim
          elsif !@mamba_d_head.nil?
            @mamba_d_head
          elsif !hidden_size.nil? && attention_heads.to_i > 0
            hidden_size / attention_heads
          else
            64
          end

          {
            "model_type" => @model_type,
            "attention_bias" => @attention_bias,
            "head_dim" => inferred_head_dim,
            "hidden_size" => hidden_size,
            "intermediate_size" => @intermediate_size || @shared_intermediate_size || hidden_size.to_i * 2,
            "max_position_embeddings" => @max_position_embeddings,
            "mamba_d_conv" => @mamba_d_conv,
            "num_attention_heads" => attention_heads,
            "num_hidden_layers" => @num_hidden_layers,
            "num_key_value_heads" => @num_key_value_heads,
            "rms_norm_eps" => @rms_norm_eps,
            "rope_theta" => @rope_theta,
            "vocab_size" => @vocab_size,
            "tie_word_embeddings" => @tie_word_embeddings,
            "attention_window_size" => @attention_window_size,
            "block_types" => @block_types,
          }
        end

        private

        def _default_layer_types
          count = @num_hidden_layers.to_i
          return nil if count <= 0

          Array.new(count) { |idx| idx.even? ? "mamba" : "attention" }
        end

        def _to_block_types
          return @block_types if @block_types.is_a?(Array) && !@block_types.empty?
          return nil unless @layer_types.is_a?(Array) && !@layer_types.empty?

          @layer_types.map { |layer_type| layer_type.to_s == "mamba" ? "recurrent" : "attention" }
        end
      end

      class Model < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.model_type = args.model_type
          self.wrapped_model = FalconH1::Model.new(
            FalconH1::ModelArgs.from_dict(args.to_falcon_h1_dict)
          )
        end

        def call(inputs, cache: nil)
          wrapped_model.call(inputs, cache: cache)
        end

        def sanitize(weights)
          normalized = weights.dup
          _rewrite_block_sparse_moe!(normalized)
          _rewrite_shared_mlp!(normalized)
          normalized.delete("lm_head.weight") if @args.tie_word_embeddings

          remapped = {}
          normalized.each do |key, value|
            remapped[_remap_weight_key(key)] = value
          end
          wrapped_model.sanitize(remapped)
        end

        def layers
          wrapped_model.layers
        end

        def make_cache
          return nil unless wrapped_model.respond_to?(:make_cache)

          wrapped_model.make_cache
        end

        private

        def _rewrite_block_sparse_moe!(weights)
          mx = MLX::Core

          @args.num_hidden_layers.to_i.times do |layer_idx|
            prefix = "model.layers.#{layer_idx}.block_sparse_moe"
            input_key = "#{prefix}.input_linear.weight"
            output_key = "#{prefix}.output_linear.weight"
            next unless weights.key?(input_key) && weights.key?(output_key)

            input_linear = weights.delete(input_key)
            output_linear = weights.delete(output_key)
            mid = input_linear.shape[1] / 2
            gate_proj, up_proj = mx.split(input_linear, [mid], 1)

            weights["#{prefix}.switch_mlp.gate_proj.weight"] = gate_proj
            weights["#{prefix}.switch_mlp.up_proj.weight"] = up_proj
            weights["#{prefix}.switch_mlp.down_proj.weight"] = output_linear
          end
        end

        def _rewrite_shared_mlp!(weights)
          mx = MLX::Core

          @args.num_hidden_layers.to_i.times do |layer_idx|
            prefix = "model.layers.#{layer_idx}.shared_mlp"
            input_key = "#{prefix}.input_linear.weight"
            output_key = "#{prefix}.output_linear.weight"
            next unless weights.key?(input_key) && weights.key?(output_key)

            input_linear = weights.delete(input_key)
            mid = input_linear.shape[0] / 2
            gate_proj, up_proj = mx.split(input_linear, [mid], 0)

            weights["model.layers.#{layer_idx}.mlp.gate_proj.weight"] = gate_proj
            weights["model.layers.#{layer_idx}.mlp.up_proj.weight"] = up_proj
            weights["model.layers.#{layer_idx}.mlp.down_proj.weight"] = weights.delete(output_key)
          end
        end

        def _remap_weight_key(key)
          mapped = key.dup
          mapped = mapped.gsub(".block_sparse_moe.", ".feed_forward.")
          mapped = mapped.gsub(".shared_mlp.", ".feed_forward.")
          mapped = mapped.gsub(".post_attention_layernorm.", ".pre_ff_layernorm.")
          mapped = mapped.gsub("model.norm.", "model.final_layernorm.")
          mapped
        end
      end

      Models.register("granitemoehybrid", Model, ModelArgs)
    end
  end
end
