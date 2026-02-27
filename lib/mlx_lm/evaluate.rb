module MlxLm
  class MLXLM
    DEFAULT_MAX_TOKENS = 8192

    attr_reader :model, :tokenizer, :max_tokens, :batch_size, :use_chat_template

    def initialize(
      path_or_hf_repo,
      max_tokens: nil,
      batch_size: 8,
      use_chat_template: nil,
      trust_remote_code: false,
      sampler: nil
    )
      tokenizer_config = trust_remote_code ? { "trust_remote_code" => true } : nil
      @model, @tokenizer = LoadUtils.load(path_or_hf_repo, tokenizer_config: tokenizer_config)
      @max_tokens = max_tokens
      @batch_size = batch_size
      @sampler = sampler
      @use_chat_template = if use_chat_template.nil?
        tokenizer.respond_to?(:has_chat_template) && tokenizer.has_chat_template
      else
        use_chat_template
      end
    end

    def tokenizer_name
      name = if tokenizer.respond_to?(:name_or_path)
        tokenizer.name_or_path
      else
        tokenizer.class.name
      end
      name.to_s.gsub("/", "__")
    end

    def generate(prompt, max_tokens: nil, sampler: nil, **kwargs)
      options = kwargs.dup
      options[:max_tokens] = max_tokens || self.max_tokens || DEFAULT_MAX_TOKENS
      options[:sampler] = sampler || @sampler
      Generate.generate(model, tokenizer, prompt, **options)
    end

    def stream_generate(prompt, max_tokens: nil, sampler: nil, **kwargs)
      options = kwargs.dup
      options[:max_tokens] = max_tokens || self.max_tokens || DEFAULT_MAX_TOKENS
      options[:sampler] = sampler || @sampler
      Generate.stream_generate(model, tokenizer, prompt, **options)
    end
  end
end
