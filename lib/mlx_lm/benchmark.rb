module MlxLm
  module Benchmark
    module_function

    # Measure generation performance (tokens/sec).
    def measure_generation(model, prompt_tokens: 32, gen_tokens: 64, vocab_size: 32000)
      mx = MLX::Core

      # Create random prompt tokens (generate float then cast to int)
      prompt = mx.random_uniform([prompt_tokens], 0.0, (vocab_size - 1).to_f, mx.float32)
      prompt = prompt.astype(mx.int32)
      mx.eval(prompt)

      # Create cache
      cache = Cache.make_prompt_cache(model)

      # Measure prompt processing
      prompt_input = prompt.reshape([1, prompt_tokens])
      prompt_start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      logits = model.call(prompt_input, cache: cache)
      mx.eval(logits)
      mx.eval(*cache.map(&:state).flatten.compact)
      prompt_elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - prompt_start
      prompt_tps = prompt_tokens.to_f / [prompt_elapsed, 1e-9].max

      # Get first generated token
      last_logits = logits.reshape([prompt_tokens, logits.shape[-1]])
      # Take last position
      last_pos = mx.split(last_logits, [prompt_tokens - 1], 0)[1]
      y = mx.argmax(last_pos, -1)
      mx.eval(y)

      # Measure generation
      gen_start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      gen_tokens.times do
        y_input = y.reshape([1, 1])
        logits = model.call(y_input, cache: cache)
        mx.eval(logits)
        mx.eval(*cache.map(&:state).flatten.compact)
        y = mx.argmax(logits.reshape([1, logits.shape[-1]]), -1)
        mx.eval(y)
      end
      gen_elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - gen_start
      gen_tps = gen_tokens.to_f / [gen_elapsed, 1e-9].max

      {
        prompt_tokens: prompt_tokens,
        prompt_time: prompt_elapsed,
        prompt_tps: prompt_tps,
        generation_tokens: gen_tokens,
        generation_time: gen_elapsed,
        generation_tps: gen_tps,
      }
    end

    # Get model statistics (parameter count, etc.)
    def model_stats(model)
      params = MLX::Utils.tree_flatten(model.parameters)
      total = 0
      params.each { |_, v| total += v.size }

      {
        total_params: total,
        num_layers: model.respond_to?(:layers) ? model.layers.length : 0,
      }
    end
  end
end
