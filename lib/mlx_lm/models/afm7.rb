require_relative "afmoe"

module MlxLm
  module Models
    module Afm7
      class ModelArgs < Afmoe::ModelArgs
        field :model_type, default: "afm7"
        field :hidden_dim, default: nil
        field :num_layers, default: nil
        field :num_kv_reuse_layers, default: 0
        field :num_heads, default: nil
        field :num_kv_heads, default: nil
        field :hidden_dim_scale_factor, default: nil

        def initialize(**kwargs)
          afm7_style = _afm7_style_kwargs?(kwargs)
          super

          @hidden_size = @hidden_dim if kwargs.key?(:hidden_dim) && !@hidden_dim.nil?
          @num_hidden_layers = @num_layers if kwargs.key?(:num_layers) && !@num_layers.nil?
          @num_attention_heads = @num_heads if kwargs.key?(:num_heads) && !@num_heads.nil?
          @num_key_value_heads = @num_kv_heads if kwargs.key?(:num_kv_heads) && !@num_kv_heads.nil?

          if kwargs.key?(:hidden_dim_scale_factor) && !@hidden_dim_scale_factor.nil? && !@hidden_size.nil?
            @intermediate_size = (@hidden_size * @hidden_dim_scale_factor.to_f).to_i
          end

          if !@hidden_size.nil? && !@num_attention_heads.nil? && @num_attention_heads.to_i > 0
            @head_dim = @hidden_size / @num_attention_heads
          end

          if kwargs.key?(:num_kv_reuse_layers) && !@num_hidden_layers.nil?
            @num_dense_layers = [@num_hidden_layers.to_i - @num_kv_reuse_layers.to_i, 0].max
          elsif afm7_style && !@num_hidden_layers.nil?
            @num_dense_layers = @num_hidden_layers
          end

          if afm7_style
            @num_experts = 1 unless kwargs.key?(:num_experts)
            @num_experts_per_tok = 1 unless kwargs.key?(:num_experts_per_tok)
            @num_shared_experts = 0 unless kwargs.key?(:num_shared_experts)
            @mup_enabled = false unless kwargs.key?(:mup_enabled)
            @layer_types = Array.new(@num_hidden_layers) { "full_attention" } unless kwargs.key?(:layer_types)
          end

          @num_key_value_heads ||= @num_attention_heads
          @layer_types ||= Array.new(@num_hidden_layers) { "full_attention" } unless @num_hidden_layers.nil?
        end

        def to_afmoe_dict
          {
            "model_type" => @model_type,
            "layer_types" => @layer_types,
            "vocab_size" => @vocab_size,
            "hidden_size" => @hidden_size,
            "intermediate_size" => @intermediate_size,
            "moe_intermediate_size" => @moe_intermediate_size,
            "num_hidden_layers" => @num_hidden_layers,
            "num_attention_heads" => @num_attention_heads,
            "num_key_value_heads" => @num_key_value_heads,
            "head_dim" => @head_dim,
            "max_position_embeddings" => @max_position_embeddings,
            "rms_norm_eps" => @rms_norm_eps,
            "rope_theta" => @rope_theta,
            "rope_scaling" => @rope_scaling,
            "tie_word_embeddings" => @tie_word_embeddings,
            "num_experts" => @num_experts,
            "num_experts_per_tok" => @num_experts_per_tok,
            "num_shared_experts" => @num_shared_experts,
            "num_dense_layers" => @num_dense_layers,
            "route_norm" => @route_norm,
            "route_scale" => @route_scale,
            "score_func" => @score_func,
            "n_group" => @n_group,
            "topk_group" => @topk_group,
            "sliding_window" => @sliding_window,
            "mup_enabled" => @mup_enabled,
          }
        end

        private

        def _afm7_style_kwargs?(kwargs)
          kwargs.key?(:hidden_dim) ||
            kwargs.key?(:num_layers) ||
            kwargs.key?(:num_heads) ||
            kwargs.key?(:num_kv_heads) ||
            kwargs.key?(:num_kv_reuse_layers) ||
            kwargs.key?(:hidden_dim_scale_factor)
        end
      end

      class Model < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.model_type = args.model_type
          self.wrapped_model = Afmoe::Model.new(Afmoe::ModelArgs.from_dict(args.to_afmoe_dict))
        end

        def call(inputs, cache: nil)
          wrapped_model.call(inputs, cache: cache)
        end

        def sanitize(weights)
          wrapped_model.sanitize(weights)
        end

        def layers
          wrapped_model.layers
        end

        def make_cache
          return nil unless wrapped_model.respond_to?(:make_cache)

          wrapped_model.make_cache
        end

        def cast_predicate
          wrapped_model.cast_predicate
        end

        def quant_predicate
          wrapped_model.quant_predicate
        end
      end

      Models.register("afm7", Model, ModelArgs)
    end
  end
end
