require_relative "activations"
require_relative "cache"
require_relative "ssm"

module MlxLm
  module Models
    module Mamba2
      class ModelArgs < BaseModelArgs
        field :model_type, default: "mamba2"
        field :num_heads
        field :head_dim
        field :vocab_size
        field :hidden_size
        field :intermediate_size, default: nil
        field :state_size
        field :num_hidden_layers
        field :layer_norm_epsilon, default: 1e-6
        field :conv_kernel
        field :n_groups
        field :use_bias, default: true
        field :use_conv_bias, default: true
        field :tie_word_embeddings, default: true
        field :time_step_limit, default: [0.001, 100.0]
        field :time_step_rank, default: "auto"
        field :ssm_state_size, default: nil
        field :max_position_embeddings, default: 2056

        def initialize(**kwargs)
          super

          @time_step_rank = (@hidden_size.to_f / 16.0).ceil if @time_step_rank == "auto"
          @ssm_state_size ||= @state_size
          @intermediate_size ||= @num_heads * @head_dim
        end
      end

      class MambaRMSNormGated < MLX::NN::Module
        def initialize(hidden_size, eps: 1e-6)
          super()
          @eps = eps
          self.weight = MLX::Core.ones([hidden_size])
        end

        def call(hidden_states, gate = nil)
          hidden_states = Activations.swiglu(gate, hidden_states) unless gate.nil?
          MLX::Core.rms_norm(hidden_states, weight, @eps)
        end
      end

      class Mamba2Block < MLX::NN::Module
        def initialize(args, layer_idx)
          super()

          _ = layer_idx
          @num_heads = args.num_heads
          @hidden_size = args.hidden_size
          @ssm_state_size = args.ssm_state_size
          @conv_kernel_size = args.conv_kernel
          @intermediate_size = args.num_heads * args.head_dim
          @n_groups = args.n_groups
          @head_dim = args.head_dim
          @time_step_limit = args.time_step_limit
          @heads_per_group = @num_heads / @n_groups

          @conv_dim = @intermediate_size + 2 * @n_groups * @ssm_state_size

          self.conv1d = MLX::NN::Conv1d.new(
            @conv_dim,
            @conv_dim,
            args.conv_kernel,
            padding: 0,
            groups: @conv_dim,
            bias: args.use_conv_bias
          )

          projection_size = @intermediate_size + @conv_dim + @num_heads
          self.in_proj = MLX::NN::Linear.new(@hidden_size, projection_size, bias: args.use_bias)

          mx = MLX::Core
          self.dt_bias = mx.ones([@num_heads])
          self.a_log = mx.log(mx.arange(1, @num_heads + 1, 1, mx.float32))
          self.d = mx.ones([@num_heads])

          self.norm = MambaRMSNormGated.new(@intermediate_size, eps: args.layer_norm_epsilon)
          self.out_proj = MLX::NN::Linear.new(@intermediate_size, @hidden_size, bias: args.use_bias)
        end

        def call(hidden_states, mask, cache = nil)
          mx = MLX::Core

          projected = in_proj.call(hidden_states)
          gate, conv_input, dt = mx.split(
            projected,
            [@intermediate_size, @intermediate_size + @conv_dim],
            -1
          )

          conv_output = _conv(conv_input, cache, mask)
          ssm_hidden, b, c = mx.split(
            conv_output,
            [@intermediate_size, @intermediate_size + @n_groups * @ssm_state_size],
            -1
          )

          y = _ssm(ssm_hidden, b, c, dt, cache, mask: mask)
          cache.advance(y.shape[1]) if cache

          out_proj.call(norm.call(y, gate))
        end

        private

        def _conv(conv_input, cache, mask)
          mx = MLX::Core

          conv_input = mx.where(mx.expand_dims(mask, -1), conv_input, 0) unless mask.nil?

          if cache
            conv_state = if cache[0].nil?
              mx.zeros(
                [conv_input.shape[0], @conv_kernel_size - 1, @conv_dim],
                conv_input.dtype
              )
            else
              cache[0]
            end

            padded_input = mx.concatenate([conv_state, conv_input], 1)
            n_keep = @conv_kernel_size - 1

            if cache.lengths
              t = padded_input.shape[1]
              ends = mx.clip(cache.lengths, 0, t - n_keep)
              positions = mx.expand_dims(
                mx.expand_dims(ends, 1) + mx.arange(n_keep),
                -1
              )
              cache[0] = mx.take_along_axis(padded_input, positions, 1)
            else
              if n_keep > 0
                split_at = padded_input.shape[1] - n_keep
                cache[0] = mx.split(padded_input, [split_at], 1)[1]
              else
                cache[0] = mx.zeros([padded_input.shape[0], 0, padded_input.shape[2]], padded_input.dtype)
              end
            end
          else
            padded_input = mx.pad(
              conv_input,
              [
                [0, 0],
                [@conv_kernel_size - 1, 0],
                [0, 0],
              ]
            )
          end

          MLX::NN.silu(conv1d.call(padded_input))
        end

        def _ssm(hidden_states, b, c, dt, cache, mask:)
          batch_size, seq_len, = hidden_states.shape
          hidden_states = hidden_states.reshape(
            [batch_size, seq_len, @num_heads, @head_dim]
          )
          b = b.reshape([batch_size, seq_len, @n_groups, @ssm_state_size])
          c = c.reshape([batch_size, seq_len, @n_groups, @ssm_state_size])

          if cache
            state = cache[1]
            lengths = cache.lengths
          else
            state = nil
            lengths = nil
          end

          y, state = SSM.ssm_update(
            hidden_states,
            a_log,
            b,
            c,
            d,
            dt,
            dt_bias,
            state: state,
            time_step_limit: @time_step_limit,
            mask: mask,
            lengths: lengths
          )

          cache[1] = state if cache
          y.reshape([batch_size, seq_len, @intermediate_size])
        end
      end

      class ResidualBlock < MLX::NN::Module
        def initialize(args, layer_idx)
          super()
          self.mixer = Mamba2Block.new(args, layer_idx)
          self.norm = MLX::NN::RMSNorm.new(args.hidden_size)
        end

        def call(x, mask, cache = nil)
          mixer.call(norm.call(x), mask, cache) + x
        end
      end

      class Mamba2Model < MLX::NN::Module
        def initialize(args)
          super()
          @args = args
          self.embeddings = MLX::NN::Embedding.new(args.vocab_size, args.hidden_size)
          self.layers = Array.new(args.num_hidden_layers) { |i| ResidualBlock.new(args, i) }
          self.norm_f = MLX::NN::RMSNorm.new(args.hidden_size, eps: args.layer_norm_epsilon)
        end

        def call(x, cache = nil)
          hidden = embeddings.call(x)
          layer_cache = cache || [nil] * layers.length

          mask = _create_ssm_mask(hidden, layer_cache[0])
          layers.each_with_index do |layer, i|
            hidden = layer.call(hidden, mask, layer_cache[i])
          end

          norm_f.call(hidden)
        end

        private

        def _create_ssm_mask(hidden, cache)
          return cache.make_mask(hidden.shape[1]) if cache && cache.respond_to?(:make_mask)

          nil
        end
      end

      class Model < MLX::NN::Module
        def initialize(args)
          super()
          self.args = args
          self.model_type = args.model_type
          self.backbone = Mamba2Model.new(args)
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

        def make_cache(batch_size: 1)
          _ = batch_size
          Array.new(args.num_hidden_layers) { MlxLm::ArraysCache.new(2) }
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

      Models.register("mamba2", Model, ModelArgs)
    end
  end
end
