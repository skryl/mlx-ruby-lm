module MlxLm
  module Perplexity
    module_function

    # Compute perplexity of a model on a token sequence.
    # Returns exp(average negative log-likelihood per token).
    def compute(model, tokens, batch_size: nil)
      ll = log_likelihood(model, tokens, batch_size: batch_size)
      num_tokens = tokens.size - 1
      avg_nll = -ll / num_tokens.to_f
      Math.exp(avg_nll)
    end

    # Compute total log-likelihood of a token sequence.
    # Sum of log P(token_i | token_0..token_{i-1}) for i in 1..n.
    def log_likelihood(model, tokens, batch_size: nil)
      mx = MLX::Core

      token_arr = tokens.is_a?(MLX::Core::Array) ? tokens : mx.array(tokens, dtype: mx.int32)
      total_tokens = token_arr.size

      # Process all at once for small sequences
      input = token_arr.reshape([1, total_tokens])
      logits = model.call(input)
      mx.eval(logits)

      # Compute log probabilities
      # logits shape: [1, total_tokens, vocab_size]
      # We want P(token[i+1] | token[0..i])
      vocab_size = logits.shape[-1]
      logits_2d = logits.reshape([total_tokens, vocab_size])

      # Log softmax
      log_probs = logits_2d - mx.logsumexp(logits_2d, -1, true)

      # Gather log probs for actual next tokens
      # For position i, the model predicts token i+1
      total_ll = 0.0
      (0...(total_tokens - 1)).each do |i|
        target_token = token_arr[i + 1].item
        lp = log_probs[i][target_token].item
        total_ll += lp
      end

      total_ll
    end
  end
end
