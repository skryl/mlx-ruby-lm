require "tokenizers"
require "json"

module MlxLm
  # Wraps a HuggingFace tokenizer (loaded via the tokenizers gem)
  # providing encode/decode and metadata access.
  class TokenizerWrapper
    attr_reader :tokenizer

    def initialize(path)
      tokenizer_json = File.join(path, "tokenizer.json")
      @tokenizer = Tokenizers::Tokenizer.from_file(tokenizer_json)

      config_path = File.join(path, "tokenizer_config.json")
      @config = File.exist?(config_path) ? JSON.parse(File.read(config_path)) : {}
    end

    def encode(text)
      @tokenizer.encode(text).ids
    end

    def decode(ids, skip_special_tokens: false)
      @tokenizer.decode(ids, skip_special_tokens: skip_special_tokens)
    end

    def eos_token_id
      # Try config first, then fall back to model
      if @config["eos_token"]
        token = @config["eos_token"]
        token = token["content"] if token.is_a?(Hash)
        id = @tokenizer.token_to_id(token)
        return id if id
      end
      nil
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
  end

  # Streaming detokenizer that incrementally decodes tokens without O(T^2) cost.
  # Uses a simple approach: maintain a buffer of token IDs, decode the full buffer,
  # and emit only the new characters since the last decode.
  class StreamingDetokenizer
    def initialize(tokenizer_wrapper)
      @tokenizer = tokenizer_wrapper
      @token_ids = []
      @prev_text = ""
    end

    # Add a token and return the new text segment (may be empty string)
    def add_token(token_id)
      @token_ids << token_id
      current_text = @tokenizer.decode(@token_ids)
      new_segment = current_text[@prev_text.length..]
      @prev_text = current_text
      new_segment || ""
    end

    # Finalize and return any remaining text
    def finalize
      return "" if @token_ids.empty?
      final = @tokenizer.decode(@token_ids)
      remaining = final[@prev_text.length..]
      remaining || ""
    end

    def reset
      @token_ids = []
      @prev_text = ""
    end
  end
end
