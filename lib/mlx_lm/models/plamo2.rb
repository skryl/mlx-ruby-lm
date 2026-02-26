require_relative "falcon_h1"

module MlxLm
  module Models
    module Plamo2
      class ModelArgs < FalconH1::ModelArgs
        field :model_type, default: "plamo2"
        field :rope_theta, default: 10_000.0
        field :tie_word_embeddings, default: true
        field :hidden_size_per_head, default: nil
        field :full_attention_idx, default: nil
        field :mamba_d_state, default: nil
        field :mamba_num_heads, default: nil
        field :mamba_step, default: 2
        field :mamba_chunk_size, default: nil
        field :mamba_enabled, default: true

        def initialize(**kwargs)
          super
          @head_dim = @hidden_size_per_head if kwargs.key?(:hidden_size_per_head) && !kwargs.key?(:head_dim) && !@hidden_size_per_head.nil?
          @num_attention_heads ||= @mamba_num_heads
          @num_key_value_heads ||= @num_attention_heads
          @mamba_d_conv ||= 4
          @attention_window_size ||= @max_position_embeddings
          @block_types ||= _to_block_types
        end

        def to_falcon_h1_dict
          hidden_size = @hidden_size
          attention_heads = @num_attention_heads
          inferred_head_dim = if !@head_dim.nil?
            @head_dim
          elsif !@hidden_size_per_head.nil?
            @hidden_size_per_head
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

        def _to_block_types
          return @block_types if @block_types.is_a?(Array) && !@block_types.empty?

          count = @num_hidden_layers.to_i
          return nil if count <= 0

          if @full_attention_idx.is_a?(Array) && !@full_attention_idx.empty?
            full_attention = @full_attention_idx.map(&:to_i)
            return Array.new(count) { |i| full_attention.include?(i) ? "attention" : "recurrent" }
          end

          return Array.new(count, "attention") unless @mamba_enabled

          step = @mamba_step.to_i
          step = 2 if step <= 1
          midpoint = step / 2

          if count <= midpoint
            return Array.new(count) { |i| i == count - 1 ? "attention" : "recurrent" }
          end

          Array.new(count) { |i| (i % step) == midpoint ? "attention" : "recurrent" }
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
          normalized = weights.is_a?(Hash) ? weights.dup : weights.to_h
          _split_gate_up_proj!(normalized)

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

        def _split_gate_up_proj!(weights)
          mx = MLX::Core
          pattern = /\A(model\.layers(?:\.layers)?\.\d+\.mlp)\.gate_up_proj\.(weight|bias|scales|biases)\z/

          weights.keys.each do |key|
            match = pattern.match(key)
            next unless match

            prefix = match[1]
            param = match[2]
            gate_up = weights.delete(key)
            mid = gate_up.shape[0] / 2
            next if mid <= 0

            gate_proj, up_proj = mx.split(gate_up, [mid], 0)
            weights["#{prefix}.gate_proj.#{param}"] = gate_proj
            weights["#{prefix}.up_proj.#{param}"] = up_proj
          end
        end

        def _remap_weight_key(key)
          mapped = key.dup
          mapped = mapped.gsub("model.layers.layers.", "model.layers.")
          mapped = mapped.gsub("model.norm.", "model.final_layernorm.")

          mapped = mapped.gsub(/\.layers\.(\d+)\.pre_mixer_norm\./) { ".layers.#{$1}.input_layernorm." }
          mapped = mapped.gsub(/\.layers\.(\d+)\.pre_mlp_norm\./) { ".layers.#{$1}.pre_ff_layernorm." }

          mapped = mapped.gsub(".mixer.conv1d.", ".mamba.conv1d.")
          mapped = mapped.gsub(".mixer.in_proj.", ".mamba.in_proj.")
          mapped = mapped.gsub(".mixer.out_proj.", ".mamba.out_proj.")
          mapped = mapped.gsub(".mixer.qkv_proj.", ".self_attn.q_proj.")
          mapped = mapped.gsub(".mixer.q_proj.", ".self_attn.q_proj.")
          mapped = mapped.gsub(".mixer.k_proj.", ".self_attn.k_proj.")
          mapped = mapped.gsub(".mixer.v_proj.", ".self_attn.v_proj.")
          mapped = mapped.gsub(".mixer.o_proj.", ".self_attn.o_proj.")
          mapped = mapped.gsub(".mlp.gate_up_proj.", ".feed_forward.gate_proj.")
          mapped = mapped.gsub(".mlp.gate_proj.", ".feed_forward.gate_proj.")
          mapped = mapped.gsub(".mlp.up_proj.", ".feed_forward.up_proj.")
          mapped = mapped.gsub(".mlp.down_proj.", ".feed_forward.down_proj.")
          mapped
        end
      end

      Models.register("plamo2", Model, ModelArgs)
    end
  end
end
