require "tokenizers"
require "json"

module MlxLm
  # Wraps a HuggingFace tokenizer (loaded via the tokenizers gem)
  # providing encode/decode and metadata access.
  class TokenizerWrapper
    attr_reader :tokenizer

    # Can be initialized with:
    # 1. A path string (directory containing tokenizer.json)
    # 2. A Tokenizers::Tokenizer object (with optional eos_token/eos_token_id)
    def initialize(path_or_tokenizer, eos_token: nil, eos_token_id: nil)
      if path_or_tokenizer.is_a?(String)
        tokenizer_json = File.join(path_or_tokenizer, "tokenizer.json")
        @tokenizer = Tokenizers::Tokenizer.from_file(tokenizer_json)

        config_path = File.join(path_or_tokenizer, "tokenizer_config.json")
        @config = File.exist?(config_path) ? JSON.parse(File.read(config_path)) : {}
      else
        @tokenizer = path_or_tokenizer
        @config = {}
      end

      @eos_token_override = eos_token
      @eos_token_id_override = eos_token_id

      @_detokenizer = nil
    end

    def encode(text, add_special_tokens: true)
      @tokenizer.encode(text, add_special_tokens: add_special_tokens).ids
    end

    def decode(ids, skip_special_tokens: false)
      @tokenizer.decode(ids, skip_special_tokens: skip_special_tokens)
    end

    def eos_token
      return @eos_token_override if @eos_token_override
      token = @config["eos_token"]
      token = token["content"] if token.is_a?(Hash)
      token
    end

    def eos_token_id
      # Try override ids first
      if @eos_token_id_override && !@eos_token_id_override.empty?
        return @eos_token_id_override.first
      end

      # Try config
      if @config["eos_token"]
        token = @config["eos_token"]
        token = token["content"] if token.is_a?(Hash)
        id = @tokenizer.token_to_id(token)
        return id if id
      end

      # Try eos_token string override
      if @eos_token_override
        id = @tokenizer.token_to_id(@eos_token_override)
        return id if id
      end

      nil
    end

    # Returns a Set of all EOS token IDs
    def eos_token_ids
      ids = Set.new
      if @eos_token_id_override
        @eos_token_id_override.each { |id| ids << id if id }
      end
      base_id = eos_token_id
      ids << base_id if base_id
      ids
    end

    def bos_token
      token = @config["bos_token"]
      token = token["content"] if token.is_a?(Hash)
      token
    end

    def bos_token_id
      if @config["bos_token"]
        token = @config["bos_token"]
        token = token["content"] if token.is_a?(Hash)
        id = @tokenizer.token_to_id(token)
        return id if id
      end
      nil
    end

    def vocab_size
      @tokenizer.vocab_size
    end

    def id_to_token(id)
      @tokenizer.id_to_token(id)
    end

    def token_to_id(token)
      @tokenizer.token_to_id(token)
    end

    def detokenizer
      @_detokenizer ||= StreamingDetokenizer.new(self)
    end

    def has_chat_template
      !!@config["chat_template"]
    end
  end

  # Streaming detokenizer that incrementally decodes tokens without O(T^2) cost.
  # Uses a simple approach: maintain a buffer of token IDs, decode the full buffer,
  # and emit only the new characters since the last decode.
  class StreamingDetokenizer
    attr_reader :last_segment

    def initialize(tokenizer_wrapper)
      @tokenizer = tokenizer_wrapper
      @token_ids = []
      @prev_text = ""
      @last_segment = ""
    end

    # Add a token and record the new text segment
    def add_token(token_id)
      @token_ids << token_id
      current_text = @tokenizer.decode(@token_ids)
      @last_segment = current_text[@prev_text.length..] || ""
      @prev_text = current_text
      @last_segment
    end

    # Finalize and record any remaining text
    def finalize
      return "" if @token_ids.empty?
      final = @tokenizer.decode(@token_ids)
      @last_segment = final[@prev_text.length..] || ""
      @prev_text = final
      @last_segment
    end

    def text
      @prev_text
    end

    def reset
      @token_ids = []
      @prev_text = ""
      @last_segment = ""
    end
  end
end
