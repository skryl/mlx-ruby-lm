require_relative "glm4_moe_lite"

module MlxLm
  module Models
    module LongcatFlash
      class ModelArgs < Glm4MoeLite::ModelArgs
        field :model_type, default: "longcat_flash"
        field :hidden_dim, default: nil
        field :ffn_hidden_size, default: nil
        field :num_layers, default: nil
        field :num_heads, default: nil
        field :num_kv_heads, default: nil
        field :num_experts, default: nil
        field :num_local_experts, default: nil
        field :num_shared_experts, default: nil
        field :top_k, default: nil
        field :score_function, default: nil

        def self.from_dict(params)
          normalized = params.each_with_object({}) do |(key, value), out|
            out[key.to_s] = value
          end

          {
            "hidden_dim" => "hidden_size",
            "ffn_hidden_size" => "intermediate_size",
            "num_layers" => "num_hidden_layers",
            "num_heads" => "num_attention_heads",
            "num_kv_heads" => "num_key_value_heads",
            "num_local_experts" => "n_routed_experts",
            "num_experts" => "n_routed_experts",
            "num_shared_experts" => "n_shared_experts",
            "top_k" => "num_experts_per_tok",
            "score_function" => "scoring_func",
          }.each do |source_key, target_key|
            next unless normalized.key?(source_key)

            normalized[target_key] = normalized[source_key] unless normalized.key?(target_key)
          end

          normalized["model_type"] ||= "longcat_flash"
          super(normalized)
        end

        def initialize(**kwargs)
          super
          @hidden_size = @hidden_dim if kwargs.key?(:hidden_dim) && !kwargs.key?(:hidden_size) && !@hidden_dim.nil?
          @intermediate_size = @ffn_hidden_size if kwargs.key?(:ffn_hidden_size) && !kwargs.key?(:intermediate_size) && !@ffn_hidden_size.nil?
          @num_hidden_layers = @num_layers if kwargs.key?(:num_layers) && !kwargs.key?(:num_hidden_layers) && !@num_layers.nil?
          @num_attention_heads = @num_heads if kwargs.key?(:num_heads) && !kwargs.key?(:num_attention_heads) && !@num_heads.nil?
          @num_key_value_heads = @num_kv_heads if kwargs.key?(:num_kv_heads) && !kwargs.key?(:num_key_value_heads) && !@num_kv_heads.nil?
          @n_routed_experts = @num_local_experts if kwargs.key?(:num_local_experts) && !kwargs.key?(:n_routed_experts) && !@num_local_experts.nil?
          @n_routed_experts = @num_experts if kwargs.key?(:num_experts) && !kwargs.key?(:n_routed_experts) && !kwargs.key?(:num_local_experts) && !@num_experts.nil?
          @n_shared_experts = @num_shared_experts if kwargs.key?(:num_shared_experts) && !kwargs.key?(:n_shared_experts) && !@num_shared_experts.nil?
          @num_experts_per_tok = @top_k if kwargs.key?(:top_k) && !kwargs.key?(:num_experts_per_tok) && !@top_k.nil?
          @scoring_func = @score_function if kwargs.key?(:score_function) && !kwargs.key?(:scoring_func) && !@score_function.nil?
          @num_key_value_heads ||= @num_attention_heads
        end

        def to_glm4_moe_lite_dict
          {
            "model_type" => @model_type,
            "vocab_size" => @vocab_size,
            "hidden_size" => @hidden_size,
            "intermediate_size" => @intermediate_size,
            "moe_intermediate_size" => @moe_intermediate_size,
            "num_hidden_layers" => @num_hidden_layers,
            "num_attention_heads" => @num_attention_heads,
            "num_key_value_heads" => @num_key_value_heads,
            "n_shared_experts" => @n_shared_experts,
            "n_routed_experts" => @n_routed_experts,
            "routed_scaling_factor" => @routed_scaling_factor,
            "kv_lora_rank" => @kv_lora_rank,
            "q_lora_rank" => @q_lora_rank,
            "qk_rope_head_dim" => @qk_rope_head_dim,
            "qk_nope_head_dim" => @qk_nope_head_dim,
            "v_head_dim" => @v_head_dim,
            "topk_method" => @topk_method,
            "scoring_func" => @scoring_func,
            "norm_topk_prob" => @norm_topk_prob,
            "n_group" => @n_group,
            "topk_group" => @topk_group,
            "num_experts_per_tok" => @num_experts_per_tok,
            "moe_layer_freq" => @moe_layer_freq,
            "first_k_dense_replace" => @first_k_dense_replace,
            "max_position_embeddings" => @max_position_embeddings,
            "rms_norm_eps" => @rms_norm_eps,
            "rope_theta" => @rope_theta,
            "rope_scaling" => @rope_scaling,
            "attention_bias" => @attention_bias,
            "attention_dropout" => @attention_dropout,
            "partial_rotary_factor" => @partial_rotary_factor,
            "tie_word_embeddings" => @tie_word_embeddings,
            "num_nextn_predict_layers" => @num_nextn_predict_layers,
            "quantization" => @quantization,
          }
        end
      end

      class Model < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.model_type = args.model_type
          self.wrapped_model = Glm4MoeLite::Model.new(
            Glm4MoeLite::ModelArgs.from_dict(args.to_glm4_moe_lite_dict)
          )
        end

        def call(inputs, cache: nil)
          wrapped_model.call(inputs, cache: cache)
        end

        def sanitize(weights)
          remapped = {}
          flat_weights = weights.is_a?(Hash) ? weights : weights.to_h
          flat_weights.each do |key, value|
            remapped[_remap_weight_key(key)] = value
          end
          wrapped_model.sanitize(remapped)
        end

        def layers
          wrapped_model.layers
        end

        def make_cache
          return wrapped_model.make_cache if wrapped_model.respond_to?(:make_cache)

          nil
        end

        def cast_predicate
          return wrapped_model.cast_predicate if wrapped_model.respond_to?(:cast_predicate)

          lambda { |_key| true }
        end

        private

        def _remap_weight_key(key)
          mapped = key.dup
          mapped = mapped.gsub(".attention.", ".self_attn.")
          mapped = mapped.gsub(".block_sparse_moe.", ".mlp.")
          mapped = mapped.gsub(".mlp.router.", ".mlp.gate.")
          mapped
        end
      end

      Models.register("longcat_flash", Model, ModelArgs)
    end
  end
end
