module MlxLm
  module SampleUtils
    module_function

    # Build a sampler callable (proc) from the given parameters.
    # Returns a proc that takes logprobs (mx.array) and returns a token (mx.array).
    def make_sampler(
      temp: 0.0,
      top_p: 0.0,
      min_p: 0.0,
      min_tokens_to_keep: 1,
      top_k: 0
    )
      mx = MLX::Core

      if temp == 0
        return ->(x) { mx.argmax(x, -1) }
      end

      sampling_methods = []
      if top_p > 0 && top_p < 1.0
        sampling_methods << ->(x) { apply_top_p(x, top_p) }
      end
      if min_p != 0.0
        sampling_methods << ->(x) { apply_min_p(x, min_p, min_tokens_to_keep) }
      end
      if top_k > 0
        sampling_methods << ->(x) { apply_top_k(x, top_k) }
      end

      ->(logprobs) {
        sampling_methods.each { |method| logprobs = method.call(logprobs) }
        categorical_sampling(logprobs, temp)
      }
    end

    def make_logits_processors(repetition_penalty: nil, repetition_context_size: 20)
      processors = []
      if repetition_penalty && repetition_penalty != 0.0
        processors << make_repetition_penalty(repetition_penalty, repetition_context_size)
      end
      processors
    end

    def apply_top_k(logprobs, top_k)
      mx = MLX::Core
      vocab_size = logprobs.shape[-1]
      raise ArgumentError, "top_k must be in (0, #{vocab_size}]" unless top_k.is_a?(Integer) && top_k > 0 && top_k < vocab_size

      neg_logprobs = mx.negative(logprobs)
      mask_idx = mx.argpartition(neg_logprobs, top_k - 1, -1)
      # Get indices after top_k (the ones to mask)
      rest = mx.split(mask_idx, [top_k], -1)[1]
      neg_inf = mx.array([-Float::INFINITY]).astype(logprobs.dtype)
      mx.put_along_axis(logprobs, rest, neg_inf, -1)
    end

    def apply_min_p(logprobs, min_p, min_tokens_to_keep = 1)
      mx = MLX::Core
      raise ArgumentError, "min_p must be in [0, 1]" unless min_p >= 0 && min_p <= 1.0

      # Sort indices in decreasing order
      neg_logprobs = mx.negative(logprobs)
      sorted_indices = mx.argsort(neg_logprobs, -1)
      sorted_logprobs = mx.take_along_axis(logprobs, sorted_indices, -1)

      # Top probability
      top_logprobs = mx.split(sorted_logprobs, [1], -1)[0]

      # Calculate the min_p threshold
      scaled_min_p = top_logprobs + Math.log(min_p)

      # Mask tokens below threshold
      tokens_to_remove = mx.less(sorted_logprobs, scaled_min_p)

      neg_inf = mx.array(-Float::INFINITY).astype(sorted_logprobs.dtype)
      selected_logprobs = mx.where(tokens_to_remove, neg_inf, sorted_logprobs)

      # Restore the top min_tokens_to_keep tokens regardless
      if min_tokens_to_keep > 0
        top_sorted = mx.split(sorted_logprobs, [min_tokens_to_keep], -1)[0]
        rest_selected = mx.split(selected_logprobs, [min_tokens_to_keep], -1)[1]
        selected_logprobs = mx.concatenate([top_sorted, rest_selected], -1)
      end

      # Create inverse mapping to restore original order
      inverse_indices = mx.put_along_axis(
        mx.zeros_like(sorted_indices),
        sorted_indices,
        mx.arange(sorted_indices.shape[-1]).astype(sorted_indices.dtype),
        -1
      )

      mx.take_along_axis(selected_logprobs, inverse_indices, -1)
    end

    def apply_top_p(logprobs, top_p)
      mx = MLX::Core
      probs = mx.exp(logprobs)
      # sort in ascending order
      sorted_indices = mx.argsort(logprobs, -1)
      sorted_probs = mx.take_along_axis(probs, sorted_indices, -1)

      cumulative_probs = mx.cumsum(sorted_probs, -1)

      # Rearrange cumulative probs back to original order
      inverse_indices = mx.put_along_axis(
        mx.zeros_like(sorted_indices),
        sorted_indices,
        mx.arange(sorted_indices.shape[-1]).astype(sorted_indices.dtype),
        -1
      )
      cumulative_probs = mx.take_along_axis(cumulative_probs, inverse_indices, -1)

      # select tokens with cumulative probs above threshold
      threshold = mx.array(1.0 - top_p).astype(cumulative_probs.dtype)
      mask = mx.greater(cumulative_probs, threshold)
      neg_inf = mx.array(-Float::INFINITY).astype(logprobs.dtype)
      mx.where(mask, logprobs, neg_inf)
    end

    def categorical_sampling(logits, temp)
      mx = MLX::Core
      mx.categorical(logits * (1.0 / temp))
    end

    def make_repetition_penalty(penalty, context_size = 20)
      mx = MLX::Core
      raise ArgumentError, "penalty must be a non-negative float" unless penalty.is_a?(Numeric) && penalty >= 0

      ->(tokens, logits) {
        if tokens && tokens.size > 0
          recent = if tokens.is_a?(::Array)
            tokens.last(context_size)
          elsif tokens.respond_to?(:tolist)
            tokens.tolist.last(context_size)
          else
            []
          end
          if recent.length > 0
            token_indices = mx.array(recent).astype(mx.int32)
            n_tokens = recent.length
            idx_2d = token_indices.reshape([1, n_tokens])
            selected_logits = mx.take_along_axis(logits, idx_2d, -1)
            zero = mx.array(0.0).astype(selected_logits.dtype)
            is_negative = mx.less(selected_logits, zero)
            selected_logits = mx.where(
              is_negative,
              selected_logits * penalty,
              selected_logits / penalty
            )
            logits = mx.put_along_axis(logits, idx_2d, selected_logits, -1)
          end
        end
        logits
      }
    end
  end
end
