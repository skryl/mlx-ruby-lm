module MlxLm
  module Quant
    module Gptq
      class Catcher < MLX::NN::Module
        attr_reader :wrapped_module, :hessian

        def initialize(wrapped_module)
          super()
          @wrapped_module = wrapped_module
          @hessian = MLX::Core.array(0.0)
        end

        def call(x, *args, **kwargs)
          mx = MLX::Core
          xf = mx.flatten(x, 0, -2)
          @hessian = @hessian + mx.matmul(mx.transpose(xf), xf)
          wrapped_module.call(x, *args, **kwargs)
        end
      end
    end
  end
end
