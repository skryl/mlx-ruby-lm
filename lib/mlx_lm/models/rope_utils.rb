module MlxLm
  module Models
    class SuScaledRoPE < MLX::NN::Module
      def initialize(
        dims,
        base: 10_000.0,
        max_position_embeddings: 131_072,
        original_max_position_embeddings: 4096,
        short_factor: 1.0,
        long_factor: 1.0,
        short_mscale: nil,
        long_mscale: nil
      )
        super()
        mx = MLX::Core
        @dim = dims
        @original_max_position_embeddings = original_max_position_embeddings

        freqs = mx.power(
          base.to_f,
          mx.divide(mx.arange(0, dims, 2, mx.float32), dims.to_f)
        )
        self._freqs = mx.multiply(mx.array(long_factor, dtype: mx.float32), freqs)

        factor = max_position_embeddings.to_f / original_max_position_embeddings
        self._scale = long_mscale || if factor <= 1.0
          1.0
        else
          Math.sqrt(1 + Math.log(factor) / Math.log(original_max_position_embeddings))
        end
      end

      def call(x, offset: 0)
        mx = MLX::Core
        x = scale_rotary_part(x, _scale)
        mx.rope(x, @dim, false, nil, 1.0, offset, _freqs)
      end

      private

      def scale_rotary_part(x, scale)
        return x if scale == 1.0

        mx = MLX::Core
        rotary, rest = mx.split(x, [@dim], -1)
        mx.concatenate([mx.multiply(rotary, scale), rest], -1)
      end
    end

    class Llama3RoPE < MLX::NN::Module
      def initialize(
        dims:,
        max_position_embeddings: 2048,
        traditional: false,
        base: 10_000,
        scaling_config: nil
      )
        super()
        mx = MLX::Core

        @dims = dims
        @max_position_embeddings = max_position_embeddings
        @traditional = traditional

        factor = config_value(scaling_config, "factor")
        low_freq_factor = config_value(scaling_config, "low_freq_factor", 1.0)
        high_freq_factor = config_value(scaling_config, "high_freq_factor", 4.0)
        old_context_len = config_value(
          scaling_config,
          "original_max_position_embeddings",
          8192
        )

        low_freq_wavelen = old_context_len.to_f / low_freq_factor
        high_freq_wavelen = old_context_len.to_f / high_freq_factor

        freqs = mx.power(
          base.to_f,
          mx.divide(mx.arange(0, dims, 2), dims.to_f)
        )
        wavelens = mx.multiply(2.0 * Math::PI, freqs)

        freqs = mx.where(
          mx.greater(wavelens, low_freq_wavelen),
          mx.multiply(freqs, factor),
          freqs
        )

        is_medium_freq = mx.logical_and(
          mx.greater(wavelens, high_freq_wavelen),
          mx.less(wavelens, low_freq_wavelen)
        )

        smooth_factors = mx.divide(
          mx.subtract(mx.divide(old_context_len.to_f, wavelens), low_freq_factor),
          (high_freq_factor - low_freq_factor).to_f
        )

        smooth_freqs = mx.divide(
          freqs,
          mx.add(
            mx.divide(mx.subtract(1.0, smooth_factors), factor.to_f),
            smooth_factors
          )
        )

        self._freqs = mx.where(is_medium_freq, smooth_freqs, freqs)
      end

      def extra_repr
        "#{@dims}, traditional=#{@traditional}, max_position_embeddings=#{@max_position_embeddings}"
      end

      def call(x, offset: 0)
        MLX::Core.rope(x, @dims, @traditional, nil, 1.0, offset, _freqs)
      end

      private

      def config_value(config, key, default = nil)
        return default if config.nil?
        return config[key] if config.key?(key)

        config.fetch(key.to_sym, default)
      end
    end

    class YarnRoPE < MLX::NN::Module
      def initialize(
        dims,
        traditional: false,
        max_position_embeddings: 2048,
        base: 10_000,
        scaling_factor: 1.0,
        original_max_position_embeddings: 4096,
        beta_fast: 32,
        beta_slow: 1,
        mscale: 1,
        mscale_all_dim: 0
      )
        super()
        mx = MLX::Core

        self.mscale = yarn_get_mscale(scaling_factor, mscale) /
                      yarn_get_mscale(scaling_factor, mscale_all_dim)

        freq_extra = mx.power(
          base.to_f,
          mx.divide(mx.arange(0, dims, 2, mx.float32), dims.to_f)
        )
        freq_inter = mx.multiply(scaling_factor.to_f, freq_extra)

        low, high = yarn_find_correction_range(
          dims,
          base,
          original_max_position_embeddings,
          beta_fast,
          beta_slow
        )

        freq_mask = mx.subtract(1.0, yarn_linear_ramp_mask(low, high, dims / 2))
        self._freqs = mx.divide(
          mx.multiply(freq_inter, freq_extra),
          mx.add(
            mx.multiply(freq_inter, freq_mask),
            mx.multiply(freq_extra, mx.subtract(1.0, freq_mask))
          )
        )

        @dims = dims
        @traditional = traditional
      end

      def call(x, offset: 0)
        mx = MLX::Core
        x = scale_rotary_part(x, mscale) unless mscale == 1.0

        mx.rope(x, @dims, @traditional, nil, 1.0, offset, _freqs)
      end

      private

      def scale_rotary_part(x, scale)
        mx = MLX::Core
        rotary, rest = mx.split(x, [@dims], -1)
        mx.concatenate([mx.multiply(rotary, scale), rest], -1)
      end

      def yarn_find_correction_dim(dims, base, original_max_position_embeddings, num_rotations)
        dims * Math.log(original_max_position_embeddings.to_f / (num_rotations * 2 * Math::PI)) /
          (2 * Math.log(base))
      end

      def yarn_find_correction_range(dims, base, original_max_position_embeddings, beta_fast, beta_slow)
        low = yarn_find_correction_dim(dims, base, original_max_position_embeddings, beta_fast).floor
        high = yarn_find_correction_dim(dims, base, original_max_position_embeddings, beta_slow).ceil
        [
          [low, 0].max,
          [high, dims - 1].min,
        ]
      end

      def yarn_get_mscale(scale = 1, mscale = 1)
        return 1.0 if scale <= 1

        0.1 * mscale * Math.log(scale) + 1.0
      end

      def yarn_linear_ramp_mask(min_val, max_val, dim)
        mx = MLX::Core

        max_val += 0.001 if min_val == max_val

        linear = mx.divide(
          mx.subtract(mx.arange(0, dim, 1, mx.float32), min_val),
          max_val - min_val
        )
        mx.clip(linear, 0.0, 1.0)
      end
    end

    module_function

    def initialize_rope(
      dims,
      base,
      traditional,
      scaling_config = nil,
      max_position_embeddings: nil
    )
      rope_type = if scaling_config
        rope_config_value(scaling_config, "type") ||
          rope_config_value(scaling_config, "rope_type", "default")
      else
        "default"
      end

      case rope_type
      when "default", "linear"
        scale = rope_type == "linear" ? 1.0 / rope_config_value(scaling_config, "factor") : 1.0
        MLX::NN::RoPE.new(dims, traditional: traditional, base: base, scale: scale)
      when "llama3"
        Llama3RoPE.new(
          dims: dims,
          max_position_embeddings: max_position_embeddings,
          traditional: traditional,
          base: base,
          scaling_config: scaling_config
        )
      when "yarn", "deepseek_yarn", "telechat3-yarn"
        rope_kwargs = {}
        %w[
          original_max_position_embeddings
          beta_fast
          beta_slow
          mscale
          mscale_all_dim
        ].each do |key|
          value = rope_config_value(scaling_config, key)
          rope_kwargs[key.to_sym] = value unless value.nil?
        end

        YarnRoPE.new(
          dims,
          max_position_embeddings: max_position_embeddings,
          traditional: traditional,
          scaling_factor: rope_config_value(scaling_config, "factor"),
          base: base,
          **rope_kwargs
        )
      when "longrope"
        SuScaledRoPE.new(
          dims,
          base: base,
          max_position_embeddings: max_position_embeddings,
          original_max_position_embeddings: rope_config_value(
            scaling_config,
            "original_max_position_embeddings"
          ),
          short_factor: rope_config_value(scaling_config, "short_factor"),
          long_factor: rope_config_value(scaling_config, "long_factor")
        )
      when "mrope"
        mrope_section = rope_config_value(scaling_config, "mrope_section", [])
        unless mrope_section.length == 3
          raise ArgumentError,
            "MRoPE currently only supports 3 sections, got #{mrope_section.length}."
        end

        MLX::NN::RoPE.new(dims, traditional: traditional, base: base)
      else
        raise ArgumentError, "Unsupported RoPE type #{rope_type}"
      end
    end

    def rope_config_value(config, key, default = nil)
      return default if config.nil?
      return config[key] if config.key?(key)

      config.fetch(key.to_sym, default)
    end
    private_class_method :rope_config_value

    module RoPEUtils
      SuScaledRoPE = MlxLm::Models::SuScaledRoPE
      Llama3RoPE = MlxLm::Models::Llama3RoPE
      YarnRoPE = MlxLm::Models::YarnRoPE

      module_function

      def initialize_rope(*args, **kwargs)
        MlxLm::Models.initialize_rope(*args, **kwargs)
      end
    end
  end
end
