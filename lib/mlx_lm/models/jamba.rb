require_relative "falcon_h1"

module MlxLm
  module Models
    module Jamba
      class ModelArgs < FalconH1::ModelArgs
        field :model_type, default: "jamba"
        field :attn_layer_offset, default: 1
        field :attn_layer_period, default: 2
        field :expert_layer_offset, default: 1
        field :expert_layer_period, default: 2
        field :mamba_d_state, default: nil
        field :mamba_expand, default: nil
        field :num_experts, default: 1
        field :num_experts_per_tok, default: 1
        field :mamba_dt_rank, default: "auto"
        field :mamba_proj_bias, default: false
        field :mamba_conv_bias, default: true
        field :layers_block_type, default: nil

        def initialize(**kwargs)
          super
          @mamba_d_conv ||= 4
          @num_key_value_heads ||= @num_attention_heads
          @layers_block_type ||= _default_layers_block_type
          @num_hidden_layers ||= Array(@layers_block_type).length
          @block_types ||= _to_block_types
        end

        def to_falcon_h1_dict
          hidden_size = @hidden_size
          attention_heads = @num_attention_heads
          inferred_head_dim = if !@head_dim.nil?
            @head_dim
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
            "intermediate_size" => @intermediate_size,
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

        def _default_layers_block_type
          count = @num_hidden_layers.to_i
          return nil if count <= 0

          period = @attn_layer_period.to_i
          offset = @attn_layer_offset.to_i
          period = 1 if period <= 0

          Array.new(count) do |idx|
            (idx % period == offset) ? "attention" : "mamba"
          end
        end

        def _to_block_types
          return @block_types if @block_types.is_a?(Array) && !@block_types.empty?
          return nil unless @layers_block_type.is_a?(Array) && !@layers_block_type.empty?

          @layers_block_type.map { |layer_type| layer_type.to_s == "mamba" ? "recurrent" : "attention" }
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
          _stack_experts!(normalized)

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

        def _stack_experts!(weights)
          mx = MLX::Core

          @args.num_hidden_layers.to_i.times do |layer_idx|
            prefix = "model.layers.#{layer_idx}.feed_forward"
            %w[gate_proj up_proj down_proj].each do |projection|
              %w[weight bias scales biases].each do |param|
                pattern = /\A#{Regexp.escape(prefix)}\.experts\.(\d+)\.#{projection}\.#{param}\z/
                matches = weights.keys.filter_map do |key|
                  match = pattern.match(key)
                  next nil unless match

                  [match[1].to_i, key]
                end
                next if matches.empty?

                stacked = matches.sort_by(&:first).map do |(_, key)|
                  weights.delete(key)
                end
                weights["#{prefix}.switch_mlp.#{projection}.#{param}"] = mx.stack(stacked)
              end
            end
          end
        end

        def _remap_weight_key(key)
          mapped = key.dup
          mapped = mapped.gsub("model.norm.", "model.final_layernorm.")
          mapped = mapped.gsub(".mixer.", ".mamba.")
          mapped = mapped.gsub(".feed_forward.router.", ".feed_forward.gate.")
          mapped
        end
      end

      Models.register("jamba", Model, ModelArgs)
    end
  end
end
