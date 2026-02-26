require_relative "activations"
require_relative "cache"

module MlxLm
  module Models
    module Mamba
      class ModelArgs < BaseModelArgs
        field :model_type, default: "mamba"
        field :vocab_size
        field :hidden_size, default: nil
        field :intermediate_size, default: nil
        field :state_size, default: nil
        field :num_hidden_layers, default: nil
        field :conv_kernel, default: nil
        field :use_bias, default: nil
        field :use_conv_bias, default: nil
        field :time_step_rank, default: "auto"
        field :tie_word_embeddings, default: true
        field :use_bcdt_rms, default: false
        field :mixer_rms_eps, default: 1e-6

        field :d_model, default: nil
        field :d_inner, default: nil
        field :d_state, default: nil
        field :n_layer, default: nil
        field :n_layers, default: nil
        field :d_conv, default: nil
        field :bias, default: nil
        field :conv_bias, default: nil

        def initialize(**kwargs)
          super

          @hidden_size ||= @d_model
          @intermediate_size ||= @d_inner
          @state_size ||= @d_state
          @num_hidden_layers ||= @n_layer
          @num_hidden_layers ||= @n_layers
          @conv_kernel ||= @d_conv
          @use_bias = @bias if @use_bias.nil?
          @use_conv_bias = @conv_bias if @use_conv_bias.nil?

          @time_step_rank = (@hidden_size.to_f / 16.0).ceil if @time_step_rank == "auto"
          @use_bcdt_rms = true if @model_type == "falcon_mamba"

          @hidden_size ||= 768
          @intermediate_size ||= 1536
          @state_size ||= 16
          @num_hidden_layers ||= 24
          @conv_kernel ||= 4
          @use_bias = true if @use_bias.nil?
          @use_conv_bias = true if @use_conv_bias.nil?
        end
      end

      class MambaBlock < MLX::NN::Module
        def initialize(args)
          super()

          @hidden_size = args.hidden_size
          @ssm_state_size = args.state_size
          @conv_kernel_size = args.conv_kernel
          @intermediate_size = args.intermediate_size
          @time_step_rank = args.time_step_rank.to_i
          @use_conv_bias = args.use_conv_bias
          @use_bcdt_rms = args.use_bcdt_rms
          @mixer_rms_eps = args.mixer_rms_eps

          self.in_proj = MLX::NN::Linear.new(
            @hidden_size,
            @intermediate_size * 2,
            bias: args.use_bias
          )

          self.conv1d = MLX::NN::Conv1d.new(
            @intermediate_size,
            @intermediate_size,
            @conv_kernel_size,
            groups: @intermediate_size,
            bias: @use_conv_bias,
            padding: 0
          )

          self.x_proj = MLX::NN::Linear.new(
            @intermediate_size,
            @time_step_rank + 2 * @ssm_state_size,
            bias: false
          )
          self.dt_proj = MLX::NN::Linear.new(@time_step_rank, @intermediate_size, bias: true)

          mx = MLX::Core
          a = mx.repeat(
            mx.arange(1.0, @ssm_state_size + 1.0, 1.0).reshape([1, @ssm_state_size]),
            @intermediate_size,
            0
          )
          self.a_log = mx.log(a)
          self.d = mx.ones([@intermediate_size])

          self.out_proj = MLX::NN::Linear.new(
            @intermediate_size,
            @hidden_size,
            bias: args.use_bias
          )
        end

        def call(x, cache)
          if cache.nil?
            conv_cache = nil
            state_cache = nil
          else
            conv_cache = cache[0]
            state_cache = cache[1]
          end

          output, new_conv_cache, new_state_cache = _process_sequence(x, conv_cache, state_cache)

          if cache.is_a?(MlxLm::ArraysCache)
            cache[0] = new_conv_cache
            cache[1] = new_state_cache
          end

          output
        end

        def ssm_step(x, a, state = nil)
          mx = MLX::Core

          delta_bc = x_proj.call(x)
          delta, b, c = mx.split(
            delta_bc,
            [@time_step_rank, @time_step_rank + @ssm_state_size],
            -1
          )

          if @use_bcdt_rms
            delta = _rms_norm(delta, eps: @mixer_rms_eps)
            b = _rms_norm(b, eps: @mixer_rms_eps)
            c = _rms_norm(c, eps: @mixer_rms_eps)
          end

          delta = MLX::NN.softplus(dt_proj.call(delta))
          new_state = mx.expand_dims(delta * x, -1) * mx.expand_dims(b, 1)

          unless state.nil?
            new_state = new_state + state * mx.exp(mx.expand_dims(delta, -1) * a)
          end

          y = mx.squeeze(mx.matmul(new_state, mx.expand_dims(c, -1)), 2)
          y = y + d * x

          [y, new_state]
        end

        private

        def _process_sequence(x, conv_cache, state_cache)
          mx = MLX::Core

          xz = in_proj.call(x)
          x_part, z = mx.split(xz, 2, -1)

          if conv_cache.nil?
            x_full = mx.pad(
              x_part,
              [
                [0, 0],
                [@conv_kernel_size - 1, 0],
                [0, 0],
              ]
            )
          else
            x_full = mx.concatenate([conv_cache, x_part], 1)
          end

          conv_out = conv1d.call(x_full)

          n_keep = @conv_kernel_size - 1
          new_conv_cache = if n_keep > 0
            split_at = x_full.shape[1] - n_keep
            mx.split(x_full, [split_at], 1)[1]
          else
            mx.zeros([x_full.shape[0], 0, x_full.shape[2]], x_full.dtype)
          end

          x_part = MLX::NN.silu(conv_out)
          a = mx.multiply(-1.0, mx.exp(a_log))

          current_state = state_cache
          ys = []
          x_part.shape[1].times do |t|
            x_t = _slice_step(x_part, t)
            y_t, current_state = ssm_step(x_t, a, current_state)
            ys << y_t
          end

          y = mx.stack(ys, 1)
          out = out_proj.call(Activations.swiglu(z, y))

          [out, new_conv_cache, current_state]
        end

        def _slice_step(array, idx)
          mx = MLX::Core
          tail = idx.zero? ? array : mx.split(array, [idx], 1)[1]
          mx.squeeze(mx.split(tail, [1], 1)[0], 1)
        end

        def _rms_norm(x, eps:)
          mx = MLX::Core
          variance = mx.mean(mx.square(x), -1, true)
          x * mx.rsqrt(variance + eps)
        end
      end

      class ResidualBlock < MLX::NN::Module
        def initialize(args)
          super()
          self.mixer = MambaBlock.new(args)
          self.norm = MLX::NN::RMSNorm.new(args.hidden_size)
        end

        def call(x, cache)
          mixer.call(norm.call(x), cache) + x
        end
      end

      class MambaModel < MLX::NN::Module
        def initialize(args)
          super()
          self.embeddings = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
          self.layers = Array.new(args.num_hidden_layers) { ResidualBlock.new(args) }
          self.norm_f = MLX::NN::RMSNorm.new(args.hidden_size)
        end

        def call(x, cache)
          hidden = embeddings.call(x)
          layer_cache = cache || [nil] * layers.length

          layers.each_with_index do |layer, i|
            hidden = layer.call(hidden, layer_cache[i])
          end

          norm_f.call(hidden)
        end
      end

      class Model < MLX::NN::Module
        def initialize(args)
          super()
          self.args = args
          self.model_type = args.model_type
          self.backbone = MambaModel.new(args)
          self.lm_head = MLX::NN::Linear.new(args.hidden_size, args.vocab_size, bias: false) unless args.tie_word_embeddings
        end

        def call(inputs, cache: nil)
          hidden = backbone.call(inputs, cache)

          if args.tie_word_embeddings
            backbone.embeddings.as_linear(hidden)
          else
            lm_head.call(hidden)
          end
        end

        def make_cache
          Array.new(layers.length) { MlxLm::ArraysCache.new(2) }
        end

        def layers
          backbone.layers
        end

        def sanitize(weights)
          sanitized = {}
          weights.each do |name, param|
            current = param
            if name.include?("conv1d.weight") && _transpose_conv_weight?(param)
              current = MLX::Core.swapaxes(param, 1, 2)
            end
            sanitized[name] = current
          end
          sanitized
        end

        private

        def _transpose_conv_weight?(param)
          return false unless param.respond_to?(:shape)
          return false unless param.shape.is_a?(Array)
          return false unless param.shape.length >= 3

          param.shape[-1] != 1
        end
      end

      Models.register("mamba", Model, ModelArgs)
    end
  end
end
