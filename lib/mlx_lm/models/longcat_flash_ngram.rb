require_relative "longcat_flash"

module MlxLm
  module Models
    module LongcatFlashNgram
      class ModelArgs < LongcatFlash::ModelArgs
        field :model_type, default: "longcat_flash_ngram"
        field :attention_method, default: nil
        field :zero_expert_type, default: "identity"
        field :moe_topk, default: nil
        field :expert_ffn_hidden_size, default: nil
        field :zero_expert_num, default: nil
        field :num_layers, default: nil
        field :ngram_vocab_size_ratio, default: 78
        field :emb_neighbor_num, default: 4
        field :emb_split_num, default: 4
        field :mla_scale_q_lora, default: nil
        field :mla_scale_kv_lora, default: nil
        field :router_bias, default: false

        def self.from_dict(params)
          normalized = params.each_with_object({}) do |(key, value), out|
            out[key.to_s] = value
          end

          {
            "num_layers" => "num_hidden_layers",
            "moe_topk" => "num_experts_per_tok",
            "expert_ffn_hidden_size" => "moe_intermediate_size",
          }.each do |source_key, target_key|
            next unless normalized.key?(source_key)

            normalized[target_key] = normalized[source_key] unless normalized.key?(target_key)
          end

          if normalized.key?("n_routed_experts") && normalized.key?("zero_expert_num") && !normalized.key?("num_local_experts")
            normalized["num_local_experts"] = normalized["n_routed_experts"].to_i + normalized["zero_expert_num"].to_i
          end

          if normalized.key?("num_attention_heads") && !normalized.key?("num_key_value_heads") && !normalized.key?("num_kv_heads")
            normalized["num_key_value_heads"] = normalized["num_attention_heads"]
          end

          normalized["model_type"] ||= "longcat_flash_ngram"
          super(normalized)
        end

        def initialize(**kwargs)
          super
          @num_hidden_layers = @num_layers if kwargs.key?(:num_layers) && !kwargs.key?(:num_hidden_layers) && !@num_layers.nil?
          @num_experts_per_tok = @moe_topk if kwargs.key?(:moe_topk) && !kwargs.key?(:num_experts_per_tok) && !@moe_topk.nil?
          @moe_intermediate_size = @expert_ffn_hidden_size if kwargs.key?(:expert_ffn_hidden_size) && !kwargs.key?(:moe_intermediate_size) && !@expert_ffn_hidden_size.nil?

          if kwargs.key?(:zero_expert_num) && !@zero_expert_num.nil? && !kwargs.key?(:num_local_experts) && !kwargs.key?(:n_routed_experts) && !@n_routed_experts.nil?
            @n_routed_experts = @n_routed_experts.to_i + @zero_expert_num.to_i
          end

          if kwargs.key?(:num_attention_heads) && !kwargs.key?(:num_key_value_heads) && !kwargs.key?(:num_kv_heads)
            @num_key_value_heads = @num_attention_heads
          end

          @num_key_value_heads ||= @num_attention_heads
        end

        def to_longcat_flash_dict
          routed_experts = @n_routed_experts
          if !@zero_expert_num.nil? && !routed_experts.nil?
            routed_experts = routed_experts.to_i + @zero_expert_num.to_i
          end

          dict = to_glm4_moe_lite_dict
          dict["model_type"] = @model_type
          dict["n_routed_experts"] = routed_experts unless routed_experts.nil?
          dict
        end
      end

      class Model < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.model_type = args.model_type
          self.wrapped_model = LongcatFlash::Model.new(
            LongcatFlash::ModelArgs.from_dict(args.to_longcat_flash_dict)
          )
        end

        def call(inputs, cache: nil)
          wrapped_model.call(inputs, cache: cache)
        end

        def sanitize(weights)
          remapped = {}
          flat_weights = weights.is_a?(Hash) ? weights : weights.to_h
          flat_weights.each do |key, value|
            remapped[_to_longcat_flash_key(key)] = value
          end

          sanitized = wrapped_model.sanitize(remapped)
          restored = {}
          sanitized.each do |key, value|
            restored[_from_longcat_flash_key(key)] = value
          end
          restored
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

        def _to_longcat_flash_key(key)
          key.to_s.gsub("model.ngram_embeddings.word_embeddings.", "model.embed_tokens.")
        end

        def _from_longcat_flash_key(key)
          key.to_s.gsub("model.embed_tokens.", "model.ngram_embeddings.word_embeddings.")
        end
      end

      Models.register("longcat_flash_ngram", Model, ModelArgs)
    end
  end
end
