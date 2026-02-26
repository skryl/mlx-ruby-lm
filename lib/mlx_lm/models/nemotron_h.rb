require_relative "falcon_h1"

module MlxLm
  module Models
    module NemotronH
      class ModelArgs < FalconH1::ModelArgs
        field :model_type, default: "nemotron_h"
        field :tie_word_embeddings, default: false
        field :mamba_num_heads, default: nil
        field :mamba_head_dim, default: nil
        field :mamba_proj_bias, default: nil
        field :ssm_state_size, default: nil
        field :conv_kernel, default: nil
        field :n_groups, default: nil
        field :mlp_bias, default: nil
        field :layer_norm_epsilon, default: nil
        field :use_bias, default: nil
        field :use_conv_bias, default: nil
        field :hybrid_override_pattern, default: nil
        field :moe_intermediate_size, default: nil
        field :moe_shared_expert_intermediate_size, default: nil
        field :n_group, default: nil
        field :n_routed_experts, default: nil
        field :n_shared_experts, default: nil
        field :topk_group, default: nil
        field :num_experts_per_tok, default: nil
        field :norm_topk_prob, default: nil
        field :routed_scaling_factor, default: nil
        field :time_step_limit, default: nil
        field :time_step_min, default: nil
        field :time_step_max, default: nil

        def initialize(**kwargs)
          super

          @mamba_d_conv = @conv_kernel if kwargs.key?(:conv_kernel) && !kwargs.key?(:mamba_d_conv) && !@conv_kernel.nil?
          @rms_norm_eps = @layer_norm_epsilon if kwargs.key?(:layer_norm_epsilon) && !kwargs.key?(:rms_norm_eps) && !@layer_norm_epsilon.nil?
          @num_attention_heads ||= @mamba_num_heads
          @head_dim ||= @mamba_head_dim

          pattern = _hybrid_pattern_array
          @hybrid_override_pattern = pattern unless pattern.nil?
          @hybrid_override_pattern ||= _default_hybrid_pattern

          if @num_hidden_layers.nil? && @hybrid_override_pattern.is_a?(Array) && !@hybrid_override_pattern.empty?
            @num_hidden_layers = @hybrid_override_pattern.length
          end

          @num_key_value_heads ||= @num_attention_heads
          @mamba_d_conv ||= 4
          @block_types ||= _to_block_types(@hybrid_override_pattern)
        end

        def to_falcon_h1_dict
          hidden_size = @hidden_size
          attention_heads = @num_attention_heads
          inferred_head_dim = if !@head_dim.nil?
            @head_dim
          elsif !@mamba_head_dim.nil?
            @mamba_head_dim
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
            "intermediate_size" => @intermediate_size || @moe_shared_expert_intermediate_size || hidden_size.to_i * 2,
            "max_position_embeddings" => @max_position_embeddings,
            "mamba_d_conv" => @mamba_d_conv,
            "num_attention_heads" => attention_heads,
            "num_hidden_layers" => @num_hidden_layers,
            "num_key_value_heads" => @num_key_value_heads,
            "rms_norm_eps" => @rms_norm_eps || @layer_norm_epsilon || 1e-5,
            "rope_theta" => @rope_theta,
            "vocab_size" => @vocab_size,
            "tie_word_embeddings" => @tie_word_embeddings,
            "attention_window_size" => @attention_window_size,
            "block_types" => @block_types,
          }
        end

        private

        def _hybrid_pattern_array
          return nil if @hybrid_override_pattern.nil?
          return @hybrid_override_pattern if @hybrid_override_pattern.is_a?(Array)
          return @hybrid_override_pattern.chars if @hybrid_override_pattern.is_a?(String)

          nil
        end

        def _default_hybrid_pattern
          count = @num_hidden_layers.to_i
          return nil if count <= 0

          Array.new(count) { |idx| idx.even? ? "*" : "M" }
        end

        def _to_block_types(pattern)
          return @block_types if @block_types.is_a?(Array) && !@block_types.empty?
          return nil unless pattern.is_a?(Array) && !pattern.empty?

          pattern.map do |block_type|
            case block_type.to_s
            when "*"
              "attention"
            else
              "recurrent"
            end
          end
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
          grouped = Hash.new { |h, k| h[k] = [] }
          pattern = /\A(backbone\.layers\.\d+\.mixer|model\.layers(?:\.layers)?\.\d+\.mixer)\.experts\.(\d+)\.(up_proj|down_proj)\.(weight|bias|scales|biases)\z/

          weights.keys.each do |key|
            match = pattern.match(key)
            next unless match

            prefix = match[1]
            expert_idx = match[2].to_i
            projection = match[3]
            param = match[4]
            grouped[[prefix, projection, param]] << [expert_idx, key]
          end

          grouped.each do |(prefix, projection, param), entries|
            next if entries.empty?

            stacked = entries.sort_by(&:first).map { |_, key| weights.delete(key) }
            target = projection == "up_proj" ? "fc1" : "fc2"
            weights["#{prefix}.switch_mlp.#{target}.#{param}"] = mx.stack(stacked)
          end
        end

        def _remap_weight_key(key)
          mapped = key.dup
          mapped = mapped.gsub("backbone.embeddings.", "model.embed_tokens.")
          mapped = mapped.gsub("backbone.norm_f.", "model.final_layernorm.")
          mapped = mapped.gsub("backbone.layers.", "model.layers.")
          mapped = mapped.gsub("model.layers.layers.", "model.layers.")

          mapped = mapped.gsub(/\.layers\.(\d+)\.norm\./) { ".layers.#{$1}.input_layernorm." }

          mapped = mapped.gsub(".mixer.conv1d.", ".mamba.conv1d.")
          mapped = mapped.gsub(".mixer.in_proj.", ".mamba.in_proj.")
          mapped = mapped.gsub(".mixer.out_proj.", ".mamba.out_proj.")
          mapped = mapped.gsub(".mixer.q_proj.", ".self_attn.q_proj.")
          mapped = mapped.gsub(".mixer.k_proj.", ".self_attn.k_proj.")
          mapped = mapped.gsub(".mixer.v_proj.", ".self_attn.v_proj.")
          mapped = mapped.gsub(".mixer.o_proj.", ".self_attn.o_proj.")
          mapped = mapped.gsub(".mixer.gate.", ".feed_forward.router.")
          mapped = mapped.gsub(".mixer.switch_mlp.fc1.", ".feed_forward.switch_mlp.up_proj.")
          mapped = mapped.gsub(".mixer.switch_mlp.fc2.", ".feed_forward.switch_mlp.down_proj.")
          mapped = mapped.gsub(".mixer.shared_experts.up_proj.", ".feed_forward.up_proj.")
          mapped = mapped.gsub(".mixer.shared_experts.down_proj.", ".feed_forward.down_proj.")
          mapped = mapped.gsub(".mixer.up_proj.", ".feed_forward.up_proj.")
          mapped = mapped.gsub(".mixer.down_proj.", ".feed_forward.down_proj.")
          mapped
        end
      end

      Models.register("nemotron_h", Model, ModelArgs)
    end
  end
end
