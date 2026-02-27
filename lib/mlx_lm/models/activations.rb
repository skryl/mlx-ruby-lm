module MlxLm
  module Models
    module Activations
      module_function

      def swiglu(gate, x)
        MLX::NN.silu(gate) * x
      end

      def xielu(x, alpha_p, alpha_n, beta, eps)
        mx = MLX::Core
        alpha_p = MLX::NN.softplus(alpha_p)
        alpha_n = beta + MLX::NN.softplus(alpha_n)

        mx.where(
          mx.greater(x, 0.0),
          alpha_p * mx.square(x) + beta * x,
          (mx.expm1(mx.minimum(x, eps)) - x) * alpha_n + beta * x
        )
      end

      class XieLU < MLX::NN::Module
        def initialize(
          alpha_p_init: 0.8,
          alpha_n_init: 0.8,
          beta: 0.5,
          eps: -1e-6
        )
          super()
          mx = MLX::Core
          alpha_p_tensor = mx.array(alpha_p_init)
          alpha_n_tensor = mx.array(alpha_n_init - beta)

          self.alpha_p = mx.log(mx.exp(alpha_p_tensor) - 1.0)
          self.alpha_n = mx.log(mx.exp(alpha_n_tensor) - 1.0)
          self.beta = mx.array(beta)
          self.eps = mx.array(eps)
        end

        def call(x)
          Activations.xielu(x, alpha_p, alpha_n, beta, eps)
        end
      end
    end
  end
end
