module MlxLm
  # Response object yielded during streaming generation
  GenerationResponse = Struct.new(
    :text,
    :token,
    :logprobs,
    :prompt_tokens,
    :prompt_tps,
    :generation_tokens,
    :generation_tps,
    :peak_memory,
    :finish_reason,
    keyword_init: true
  )

  module Generate
    module_function

    # A generator producing token ids based on the given prompt from the model.
    # Yields [token_id, logprobs] for each generated token.
    def generate_step(
      prompt,
      model,
      max_tokens: 256,
      sampler: nil,
      logits_processors: nil,
      max_kv_size: nil,
      prompt_cache: nil,
      prefill_step_size: 2048
    )
      mx = MLX::Core

      raise ArgumentError, "prompt must not be empty" if prompt.size == 0

      tokens = nil

      # Create the KV cache for generation
      prompt_cache ||= Cache.make_prompt_cache(model, max_kv_size: max_kv_size)

      sampler ||= ->(x) { mx.argmax(x, -1) }

      model_call = ->(input_tokens_2d) {
        model.call(input_tokens_2d, cache: prompt_cache)
      }

      step = ->(input_tokens_1d) {
        seq_len = input_tokens_1d.size
        input_2d = input_tokens_1d.reshape([1, seq_len])
        logits = model_call.call(input_2d)

        # Take the last token's logits
        last_dim = logits.shape[1]
        if last_dim > 1
          logits = mx.split(logits, [last_dim - 1], 1)[1]
        end
        vocab_size = logits.shape[-1]
        logits = logits.reshape([1, vocab_size])

        if logits_processors && input_tokens_1d.size > 0
          tokens = if tokens.nil?
            input_tokens_1d
          else
            mx.concatenate([tokens, input_tokens_1d], 0)
          end
          logits_processors.each { |processor| logits = processor.call(tokens, logits) }
        end

        logprobs = logits - mx.logsumexp(logits, -1, true)
        sampled = sampler.call(logprobs)
        [sampled, logprobs.reshape([vocab_size])]
      }

      # Prompt prefilling - process prompt in chunks
      prompt_arr = prompt.is_a?(::Array) ? mx.array(prompt, dtype: mx.uint32) : prompt
      total_prompt_tokens = prompt_arr.size

      # Process prompt chunks (all but last token)
      while total_prompt_tokens > 1
        remaining = total_prompt_tokens - 1
        n_to_process = [prefill_step_size, remaining].min
        chunk = mx.split(prompt_arr, [n_to_process], 0)[0]
        chunk_len = chunk.size
        model_call.call(chunk.reshape([1, chunk_len]))
        mx.eval(*prompt_cache.map(&:state).flatten.compact)
        prompt_arr = mx.split(prompt_arr, [n_to_process], 0)[1]
        total_prompt_tokens -= n_to_process
      end

      # Process last token and get first generated token
      y, logprobs = step.call(prompt_arr)
      mx.eval(y, logprobs)

      Enumerator.new do |yielder|
        n = 0
        loop do
          break if n == max_tokens

          y_1d = y.ndim > 1 ? y.reshape([y.size]) : y
          next_y, next_logprobs = step.call(y_1d)
          mx.eval(next_y, next_logprobs)

          yielder.yield [y.item, logprobs]
          y, logprobs = next_y, next_logprobs
          n += 1
        end
      end
    end

    # Stream text generation from the model.
    # Yields GenerationResponse objects with text segments.
    def stream_generate(model, tokenizer, prompt, max_tokens: 256, **kwargs)
      tokenizer = TokenizerWrapper.new(tokenizer) unless tokenizer.is_a?(TokenizerWrapper)

      unless prompt.is_a?(MLX::Core::Array)
        if prompt.is_a?(String)
          prompt = tokenizer.encode(prompt)
        end
        prompt = MLX::Core.array(prompt, dtype: MLX::Core.uint32)
      end

      detokenizer = tokenizer.detokenizer

      token_generator = generate_step(prompt, model, max_tokens: max_tokens, **kwargs)

      tic = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      prompt_tps = 0.0

      Enumerator.new do |yielder|
        n = 0
        last_token = nil
        token_generator.each do |token, logprobs|
          if n == 0
            prompt_time = Process.clock_gettime(Process::CLOCK_MONOTONIC) - tic
            prompt_tps = prompt.size.to_f / [prompt_time, 1e-9].max
            tic = Process.clock_gettime(Process::CLOCK_MONOTONIC)
          end

          last_token = token

          if tokenizer.eos_token_ids.include?(token)
            detokenizer.finalize
            elapsed = [Process.clock_gettime(Process::CLOCK_MONOTONIC) - tic, 1e-9].max
            yielder.yield GenerationResponse.new(
              text: detokenizer.last_segment,
              token: token,
              logprobs: logprobs,
              prompt_tokens: prompt.size,
              prompt_tps: prompt_tps,
              generation_tokens: n + 1,
              generation_tps: (n + 1).to_f / elapsed,
              peak_memory: 0.0,
              finish_reason: "stop"
            )
            break
          end

          detokenizer.add_token(token)
          elapsed = [Process.clock_gettime(Process::CLOCK_MONOTONIC) - tic, 1e-9].max

          yielder.yield GenerationResponse.new(
            text: detokenizer.last_segment,
            token: token,
            logprobs: logprobs,
            prompt_tokens: prompt.size,
            prompt_tps: prompt_tps,
            generation_tokens: n + 1,
            generation_tps: (n + 1).to_f / elapsed,
            peak_memory: 0.0,
            finish_reason: ((n + 1) == max_tokens ? "length" : nil)
          )

          n += 1
          break if (n + 1) == max_tokens
        end
      end
    end

    # Non-streaming generation, returns complete text.
    def generate(model, tokenizer, prompt, verbose: false, **kwargs)
      text = ""
      response = nil
      stream_generate(model, tokenizer, prompt, **kwargs).each do |resp|
        text += resp.text
        response = resp
        if verbose
          print resp.text
          $stdout.flush
        end
      end

      if verbose
        puts
        puts "=" * 10
        if text.empty?
          puts "No text generated for this prompt"
          return text
        end
        puts "Prompt: #{response.prompt_tokens} tokens, #{'%.3f' % response.prompt_tps} tokens-per-sec"
        puts "Generation: #{response.generation_tokens} tokens, #{'%.3f' % response.generation_tps} tokens-per-sec"
      end
      text
    end
  end
end
