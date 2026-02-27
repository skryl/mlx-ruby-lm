require_relative "olmo2"

module MlxLm
  module Models
    module OLMoE
      class ModelArgs < OLMo2::ModelArgs
        field :model_type, default: "olmoe"
        field :num_experts
        field :num_experts_per_tok
        field :norm_topk_prob, default: false
      end

      class Model < OLMo2::Model
        def sanitize(weights)
          result = super(weights)
          rewrite_expert_weights(result)
        end

        private

        def rewrite_expert_weights(weights)
          return weights unless weights.key?("model.layers.0.mlp.experts.0.up_proj.weight")

          mx = MLX::Core

          layers.length.times do |layer_idx|
            prefix = "model.layers.#{layer_idx}.mlp"
            %w[up_proj down_proj gate_proj].each do |projection|
              %w[weight scales biases].each do |param|
                first_key = "#{prefix}.experts.0.#{projection}.#{param}"
                next unless weights.key?(first_key)

                expert_count = @args.num_experts || infer_expert_count(weights, prefix, projection, param)
                next unless expert_count && expert_count.positive?

                expert_keys = (0...expert_count).map do |expert_idx|
                  "#{prefix}.experts.#{expert_idx}.#{projection}.#{param}"
                end
                next unless expert_keys.all? { |key| weights.key?(key) }

                weights["#{prefix}.switch_mlp.#{projection}.#{param}"] = mx.stack(expert_keys.map { |key| weights.delete(key) })
              end
            end
          end

          weights
        end

        def infer_expert_count(weights, prefix, projection, param)
          pattern = /\A#{Regexp.escape(prefix)}\.experts\.(\d+)\.#{projection}\.#{param}\z/
          indices = weights.keys.filter_map do |key|
            match = pattern.match(key)
            match[1].to_i if match
          end
          return 0 if indices.empty?

          indices.max + 1
        end
      end

      Models.register("olmoe", Model, ModelArgs)
    end
  end
end
