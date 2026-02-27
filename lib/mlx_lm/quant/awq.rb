module MlxLm
  module Quant
    module Awq
      class ScaleConfig
        attr_accessor :prev, :layers, :block, :kwargs, :use_config

        def initialize(prev:, layers:, block: nil, kwargs: [], use_config: nil)
          @prev = prev
          @layers = layers
          @block = block
          @kwargs = kwargs
          @use_config = use_config
        end

        def use_for?(layer_block)
          return true if use_config.nil?

          use_config.call(layer_block)
        end
      end

      class AWQConfig
        attr_accessor :embed, :lm_head, :no_clip, :scale_configs, :lm_key

        def initialize(embed:, lm_head:, no_clip:, scale_configs:, lm_key: nil)
          @embed = embed
          @lm_head = lm_head
          @no_clip = no_clip
          @scale_configs = scale_configs
          @lm_key = lm_key
        end
      end

      class Catcher < MLX::NN::Module
        attr_reader :wrapped_module

        def initialize(wrapped_module)
          super()
          @wrapped_module = wrapped_module
        end

        def call(x, *args, **kwargs)
          append_to_feature(:@input_feat, x)
          append_to_feature(:@indices, args.first) if switch_linear_like? && !args.empty?
          wrapped_module.call(x, *args, **kwargs)
        end

        private

        def append_to_feature(ivar_name, value)
          existing = wrapped_module.instance_variable_get(ivar_name)
          if existing.nil?
            wrapped_module.instance_variable_set(ivar_name, value)
          else
            wrapped_module.instance_variable_set(
              ivar_name,
              MLX::Core.concatenate([existing, value], 0)
            )
          end
        end

        def switch_linear_like?
          wrapped_module.class.name.to_s.include?("SwitchLinear")
        end
      end
    end
  end
end
