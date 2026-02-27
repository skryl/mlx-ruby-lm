require_relative "bailing_moe_linear"

module MlxLm
  module Models
    module KimiLinear
      class ModelArgs < BailingMoeLinear::ModelArgs
        field :model_type, default: "kimi_linear"
        field :hidden_dim, default: nil
        field :ffn_hidden_size, default: nil
        field :num_layers, default: nil
        field :num_heads, default: nil
        field :num_kv_heads, default: nil
        field :num_local_experts, default: nil
        field :n_routed_experts, default: nil
        field :n_shared_experts, default: nil
        field :top_k, default: nil
        field :score_func, default: nil

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
            "num_local_experts" => "num_experts",
            "n_routed_experts" => "num_experts",
            "n_shared_experts" => "num_shared_experts",
            "top_k" => "num_experts_per_tok",
            "score_func" => "score_function",
          }.each do |source_key, target_key|
            next unless normalized.key?(source_key)

            normalized[target_key] = normalized[source_key] unless normalized.key?(target_key)
          end

          normalized["model_type"] ||= "kimi_linear"
          super(normalized)
        end

        def initialize(**kwargs)
          super
          @hidden_size = @hidden_dim if kwargs.key?(:hidden_dim) && !kwargs.key?(:hidden_size) && !@hidden_dim.nil?
          @intermediate_size = @ffn_hidden_size if kwargs.key?(:ffn_hidden_size) && !kwargs.key?(:intermediate_size) && !@ffn_hidden_size.nil?
          @num_hidden_layers = @num_layers if kwargs.key?(:num_layers) && !kwargs.key?(:num_hidden_layers) && !@num_layers.nil?
          @num_attention_heads = @num_heads if kwargs.key?(:num_heads) && !kwargs.key?(:num_attention_heads) && !@num_heads.nil?
          @num_key_value_heads = @num_kv_heads if kwargs.key?(:num_kv_heads) && !kwargs.key?(:num_key_value_heads) && !@num_kv_heads.nil?
          @num_experts = @num_local_experts if kwargs.key?(:num_local_experts) && !kwargs.key?(:num_experts) && !@num_local_experts.nil?
          @num_experts = @n_routed_experts if kwargs.key?(:n_routed_experts) && !kwargs.key?(:num_experts) && !kwargs.key?(:num_local_experts) && !@n_routed_experts.nil?
          @num_shared_experts = @n_shared_experts if kwargs.key?(:n_shared_experts) && !kwargs.key?(:num_shared_experts) && !@n_shared_experts.nil?
          @num_experts_per_tok = @top_k if kwargs.key?(:top_k) && !kwargs.key?(:num_experts_per_tok) && !@top_k.nil?
          @score_function = @score_func if kwargs.key?(:score_func) && !kwargs.key?(:score_function) && !@score_func.nil?
          @num_key_value_heads ||= @num_attention_heads
        end

        def to_bailing_moe_linear_dict
          to_bailing_moe_dict
        end
      end

      class Model < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.model_type = args.model_type
          self.wrapped_model = BailingMoeLinear::Model.new(
            BailingMoeLinear::ModelArgs.from_dict(args.to_bailing_moe_linear_dict)
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

        def quant_predicate
          return wrapped_model.quant_predicate if wrapped_model.respond_to?(:quant_predicate)

          lambda { |_key, _value| true }
        end

        private

        def _remap_weight_key(key)
          mapped = key.dup
          mapped = mapped.gsub(".mlp.router.", ".mlp.gate.")
          mapped = mapped.gsub("model.embed_tokens.", "model.word_embeddings.")
          mapped = mapped.gsub("model.tok_embeddings.", "model.word_embeddings.")
          mapped
        end
      end

      Models.register("kimi_linear", Model, ModelArgs)
    end
  end
end
