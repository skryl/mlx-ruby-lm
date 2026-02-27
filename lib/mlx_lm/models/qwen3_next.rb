require_relative "kimi_linear"

module MlxLm
  module Models
    module Qwen3Next
      class ModelArgs < KimiLinear::ModelArgs
        field :model_type, default: "qwen3_next"
        field :linear_num_value_heads, default: nil
        field :linear_num_key_heads, default: nil
        field :linear_key_head_dim, default: nil
        field :linear_value_head_dim, default: nil
        field :linear_conv_kernel_dim, default: nil
        field :decoder_sparse_step, default: nil
        field :shared_expert_intermediate_size, default: nil
        field :mlp_only_layers, default: []
        field :full_attention_interval, default: 4
        field :head_dim, default: nil
        field :attention_bias, default: false
        field :num_shared_experts, default: 1
        field :norm_topk_prob, default: false
        field :first_k_dense_replace, default: 0

        def self.from_dict(params)
          normalized = params.each_with_object({}) do |(key, value), out|
            out[key.to_s] = value
          end

          {
            "shared_expert_intermediate_size" => "moe_shared_expert_intermediate_size",
          }.each do |source_key, target_key|
            next unless normalized.key?(source_key)

            normalized[target_key] = normalized[source_key] unless normalized.key?(target_key)
          end

          if normalized.key?("attention_bias")
            normalized["use_bias"] = normalized["attention_bias"] unless normalized.key?("use_bias")
            normalized["use_qkv_bias"] = normalized["attention_bias"] unless normalized.key?("use_qkv_bias")
          end

          if normalized.key?("linear_num_key_heads") && !normalized.key?("num_key_value_heads")
            normalized["num_key_value_heads"] = normalized["linear_num_key_heads"]
          end

          if normalized.key?("mlp_only_layers") && !normalized.key?("first_k_dense_replace")
            normalized["first_k_dense_replace"] = _dense_prefix_length(normalized["mlp_only_layers"])
          end

          normalized["num_shared_experts"] = 1 unless normalized.key?("num_shared_experts")
          normalized["norm_topk_prob"] = false unless normalized.key?("norm_topk_prob")
          normalized["first_k_dense_replace"] = 0 unless normalized.key?("first_k_dense_replace")
          normalized["model_type"] ||= "qwen3_next"
          super(normalized)
        end

        def initialize(**kwargs)
          super
          @moe_shared_expert_intermediate_size = @shared_expert_intermediate_size if kwargs.key?(:shared_expert_intermediate_size) && !kwargs.key?(:moe_shared_expert_intermediate_size) && !@shared_expert_intermediate_size.nil?

          if kwargs.key?(:attention_bias) && !@attention_bias.nil?
            @use_bias = @attention_bias unless kwargs.key?(:use_bias)
            @use_qkv_bias = @attention_bias unless kwargs.key?(:use_qkv_bias)
          end

          if kwargs.key?(:mlp_only_layers) && !kwargs.key?(:first_k_dense_replace)
            @first_k_dense_replace = self.class._dense_prefix_length(@mlp_only_layers)
          end

          @num_shared_experts = 1 if @num_shared_experts.nil?
          @norm_topk_prob = false if @norm_topk_prob.nil?
          @first_k_dense_replace = 0 if @first_k_dense_replace.nil?
          @num_key_value_heads ||= @num_attention_heads
        end

        def to_kimi_linear_dict
          dict = to_bailing_moe_linear_dict
          dict["model_type"] = @model_type
          dict["num_shared_experts"] = @num_shared_experts || 1
          dict["norm_topk_prob"] = @norm_topk_prob.nil? ? false : @norm_topk_prob
          dict["first_k_dense_replace"] = @first_k_dense_replace || 0
          dict["use_bias"] = @use_bias
          dict["use_qkv_bias"] = @use_qkv_bias
          dict["moe_shared_expert_intermediate_size"] = @moe_shared_expert_intermediate_size unless @moe_shared_expert_intermediate_size.nil?
          dict
        end

        def self._dense_prefix_length(mlp_only_layers)
          layers = Array(mlp_only_layers).map(&:to_i)
          count = 0
          count += 1 while layers.include?(count)
          count
        end
      end

      class Model < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.model_type = args.model_type
          self.wrapped_model = KimiLinear::Model.new(
            KimiLinear::ModelArgs.from_dict(args.to_kimi_linear_dict)
          )
        end

        def call(inputs, cache: nil)
          wrapped_model.call(inputs, cache: cache)
        end

        def sanitize(weights)
          remapped = {}
          flat_weights = weights.is_a?(Hash) ? weights : weights.to_h
          flat_weights.each do |key, value|
            mapped = key.to_s.gsub(".mlp.shared_expert.", ".mlp.shared_experts.")
            next if mapped.include?(".mtp.")

            remapped[mapped] = value
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

        def quant_predicate
          return wrapped_model.quant_predicate if wrapped_model.respond_to?(:quant_predicate)

          lambda { |_key, _value| true }
        end
      end

      Models.register("qwen3_next", Model, ModelArgs)
    end
  end
end
