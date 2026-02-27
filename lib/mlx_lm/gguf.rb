require "json"
require "pathname"
require "set"

module MlxLm
  class TokenType
    NORMAL = 1
    UNKNOWN = 2
    CONTROL = 3
    USER_DEFINED = 4
    UNUSED = 5
    BYTE = 6
  end

  class GGMLFileType
    GGML_TYPE_F16 = 1
  end

  class HfVocab
    BYTE_TOKEN_PATTERN = /\A<0x[0-9A-Fa-f]{2}>\z/

    attr_reader :tokenizer, :added_tokens_list, :added_tokens_dict, :added_tokens_ids,
      :specials, :special_ids, :vocab_size_base, :vocab_size, :fname_tokenizer,
      :fname_added_tokens

    def initialize(fname_tokenizer, fname_added_tokens = nil)
      @fname_tokenizer = Pathname.new(fname_tokenizer.to_s)
      tokenizer_path = @fname_tokenizer.directory? ? @fname_tokenizer : @fname_tokenizer.dirname
      @tokenizer = TokenizerWrapper.new(tokenizer_path.to_s)

      @fname_added_tokens = normalize_added_tokens_path(fname_added_tokens, tokenizer_path)
      @added_tokens_list = []
      @added_tokens_dict = {}
      @added_tokens_ids = Set.new

      load_added_tokens!

      @specials = {}
      register_special(@tokenizer.bos_token)
      register_special(@tokenizer.eos_token)
      @special_ids = Set.new(@specials.values.compact)

      @vocab_size_base = @tokenizer.vocab_size
      @vocab_size = @vocab_size_base + @added_tokens_list.length
    end

    def hf_tokens
      Enumerator.new do |yielder|
        @vocab_size_base.times do |token_id|
          next if @added_tokens_ids.include?(token_id)

          token_text = @tokenizer.id_to_token(token_id)
          yielder << [
            token_text,
            get_token_score(token_id),
            get_token_type(token_id, token_text, @special_ids),
          ]
        end
      end
    end

    def get_token_type(token_id, token_text, special_ids)
      return TokenType::BYTE if token_text&.match?(BYTE_TOKEN_PATTERN)

      special_ids.include?(token_id) ? TokenType::CONTROL : TokenType::NORMAL
    end

    def get_token_score(_token_id)
      -1000.0
    end

    def added_tokens
      Enumerator.new do |yielder|
        @added_tokens_list.each do |text|
          if @specials.key?(text)
            token_id = @specials[text]
            yielder << [text, get_token_score(token_id), get_token_type(token_id, "", @special_ids)]
          else
            yielder << [text, -1000.0, TokenType::USER_DEFINED]
          end
        end
      end
    end

    def has_newline_token
      !@tokenizer.token_to_id("<0x0A>").nil? || !@tokenizer.token_to_id("\n").nil?
    end

    def all_tokens
      Enumerator.new do |yielder|
        hf_tokens.each { |row| yielder << row }
        added_tokens.each { |row| yielder << row }
      end
    end

    def inspect
      "<HfVocab with #{@vocab_size_base} base tokens and #{@added_tokens_list.length} added tokens>"
    end
    alias_method :to_s, :inspect

    def self.load(path)
      path = Pathname.new(path.to_s)
      tokenizer_path = path.directory? ? path : path.dirname
      added_tokens_path = tokenizer_path.join("added_tokens.json")
      new(path, added_tokens_path.exist? ? added_tokens_path : nil)
    end

    private

    def normalize_added_tokens_path(path, tokenizer_path)
      return nil if path.nil?

      path = Pathname.new(path.to_s)
      path.relative? ? tokenizer_path.join(path) : path
    end

    def register_special(token)
      return if token.nil?

      token_id = @tokenizer.token_to_id(token)
      return if token_id.nil?

      @specials[token] = token_id
    end

    def load_added_tokens!
      return if @fname_added_tokens.nil? || !@fname_added_tokens.exist?

      raw = JSON.parse(@fname_added_tokens.read)
      pairs = raw.map { |token, token_id| [token, token_id.to_i] }.sort_by { |(_token, token_id)| token_id }
      pairs.each do |token, token_id|
        next unless token_id >= @tokenizer.vocab_size

        @added_tokens_list << token
        @added_tokens_dict[token] = token_id
        @added_tokens_ids << token_id
      end
    rescue JSON::ParserError
      @added_tokens_list = []
      @added_tokens_dict = {}
      @added_tokens_ids = Set.new
    end
  end
end
