require_relative "deepseek_v32"

module MlxLm
  module Models
    module GlmMoeDsa
      class ModelArgs < DeepseekV32::ModelArgs
        field :model_type, default: "glm_moe_dsa"
        field :rope_parameters, default: nil

        def initialize(**kwargs)
          super
          return unless @rope_parameters.respond_to?(:[])

          @rope_scaling = @rope_parameters
          rope_theta = @rope_parameters["rope_theta"] || @rope_parameters[:rope_theta]
          @rope_theta = rope_theta unless rope_theta.nil?
        end
      end

      class Model < DeepseekV32::Model
      end

      Models.register("glm_moe_dsa", Model, ModelArgs)
    end
  end
end
