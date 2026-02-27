require "optparse"

module MlxLm
  module CLI
    COMMANDS = %w[generate chat server].freeze

    module_function

    def parse_args(argv)
      command = argv.shift
      unless COMMANDS.include?(command)
        raise ArgumentError, "Unknown command '#{command}'. Valid commands: #{COMMANDS.join(', ')}"
      end

      args = default_args.merge(command: command)

      parser = OptionParser.new do |opts|
        opts.banner = "Usage: mlx_lm #{command} [options]"

        opts.on("--model MODEL", "Model path or HuggingFace ID") { |v| args[:model] = v }
        opts.on("--prompt PROMPT", "Input prompt") { |v| args[:prompt] = v }
        opts.on("--max-tokens N", Integer, "Maximum tokens to generate") { |v| args[:max_tokens] = v }
        opts.on("--temp TEMP", Float, "Sampling temperature") { |v| args[:temp] = v }
        opts.on("--top-p P", Float, "Top-p (nucleus) sampling") { |v| args[:top_p] = v }
        opts.on("--seed N", Integer, "Random seed") { |v| args[:seed] = v }
        opts.on("--repetition-penalty F", Float, "Repetition penalty") { |v| args[:repetition_penalty] = v }
        opts.on("--repetition-context-size N", Integer, "Repetition context size") { |v| args[:repetition_context_size] = v }
        opts.on("--host HOST", "Server host") { |v| args[:host] = v }
        opts.on("--port PORT", Integer, "Server port") { |v| args[:port] = v }
        opts.on("--system-prompt PROMPT", "System prompt for chat") { |v| args[:system_prompt] = v }
        opts.on("--verbose", "Verbose output") { args[:verbose] = true }
      end

      parser.parse!(argv)
      args
    end

    def default_args
      {
        command: nil,
        model: nil,
        prompt: "",
        max_tokens: 256,
        temp: 0.0,
        top_p: 1.0,
        seed: nil,
        repetition_penalty: nil,
        repetition_context_size: 20,
        host: "127.0.0.1",
        port: 8080,
        system_prompt: nil,
        verbose: false,
      }
    end

    def run(argv = ARGV)
      args = parse_args(argv.dup)

      case args[:command]
      when "generate"
        run_generate(args)
      when "chat"
        run_chat(args)
      when "server"
        run_server(args)
      end
    end

    def run_generate(args)
      model, tokenizer = LoadUtils.load(args[:model])
      sampler = SampleUtils.make_sampler(temp: args[:temp], top_p: args[:top_p])
      text = Generate.generate(model, tokenizer, args[:prompt],
        max_tokens: args[:max_tokens], sampler: sampler, verbose: args[:verbose])
      puts text unless args[:verbose]
    end

    def run_chat(args)
      model, tokenizer = LoadUtils.load(args[:model])
      messages = []
      if args[:system_prompt]
        messages << { "role" => "system", "content" => args[:system_prompt] }
      end

      loop do
        print "> "
        $stdout.flush
        input = $stdin.gets
        break if input.nil?
        input = input.strip
        break if input.empty? || input == "exit" || input == "quit"

        messages << { "role" => "user", "content" => input }
        prompt = ChatTemplate.apply(messages)

        sampler = SampleUtils.make_sampler(temp: args[:temp])
        text = ""
        Generate.stream_generate(model, tokenizer, prompt,
          max_tokens: args[:max_tokens], sampler: sampler).each do |resp|
          print resp.text
          $stdout.flush
          text += resp.text
        end
        puts

        messages << { "role" => "assistant", "content" => text }
      end
    end

    def run_server(args)
      Server.start(model_path: args[:model], host: args[:host], port: args[:port])
    end
  end
end
