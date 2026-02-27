module MlxLm
  module Tuner
    module DatasetHelpers
      private

      def fetch_key(container, key)
        return nil if container.nil?
        return container[key] if container.respond_to?(:key?) && container.key?(key)

        symbol_key = key.to_sym
        if container.respond_to?(:key?) && container.key?(symbol_key)
          container[symbol_key]
        elsif container.respond_to?(:[])
          container[key]
        end
      end

      def encode(text)
        encoded = tokenizer.encode(text.to_s)
        encoded.respond_to?(:ids) ? encoded.ids : encoded
      end

      def tokenizer_eos_id
        if tokenizer.respond_to?(:eos_token_id)
          tokenizer.eos_token_id
        elsif tokenizer.respond_to?(:eos_token_ids)
          ids = tokenizer.eos_token_ids
          ids.respond_to?(:first) ? ids.first : nil
        end
      end

      def apply_chat_template(messages, tools: nil, add_generation_prompt: false)
        if tokenizer.respond_to?(:apply_chat_template)
          out = tokenizer.apply_chat_template(
            messages,
            tools: tools,
            add_generation_prompt: add_generation_prompt,
            return_dict: false
          )
          return out.respond_to?(:ids) ? out.ids : out
        end

        normalized = Array(messages)
        text = normalized.map { |m| "#{fetch_key(m, 'role')}: #{fetch_key(m, 'content')}" }.join("\n")
        encode(text)
      end
    end

    class TextDataset
      include DatasetHelpers

      attr_reader :tokenizer, :text_key

      def initialize(data, tokenizer, text_key: "text")
        @data = data
        @tokenizer = tokenizer
        @text_key = text_key
      end

      def process(datum)
        tokens = encode(fetch_key(datum, text_key))
        eos = tokenizer_eos_id
        tokens << eos if eos && (tokens.empty? || tokens[-1] != eos)
        [tokens, 0]
      end

      def [](idx)
        @data[idx]
      end

      def length
        @data.length
      end
      alias_method :size, :length
    end

    class ChatDataset
      include DatasetHelpers

      attr_reader :tokenizer, :chat_key, :mask_prompt

      def initialize(data, tokenizer, chat_key: "messages", mask_prompt: false)
        @data = data
        @tokenizer = tokenizer
        @chat_key = chat_key
        @mask_prompt = mask_prompt
      end

      def process(datum)
        messages = fetch_key(datum, chat_key)
        tools = fetch_key(datum, "tools")
        tokens = apply_chat_template(messages, tools: tools)

        return [tokens, 0] unless mask_prompt

        head = messages[0...-1]
        add_generation_prompt = fetch_key(messages[-1], "role") == "assistant"
        offset_tokens = apply_chat_template(head, tools: tools, add_generation_prompt: add_generation_prompt)
        [tokens, offset_tokens.length]
      end

      def [](idx)
        @data[idx]
      end

      def length
        @data.length
      end
      alias_method :size, :length
    end

    class CompletionsDataset
      include DatasetHelpers

      attr_reader :tokenizer, :prompt_key, :completion_key, :mask_prompt

      def initialize(data, tokenizer, prompt_key:, completion_key:, mask_prompt:)
        @data = data
        @tokenizer = tokenizer
        @prompt_key = prompt_key
        @completion_key = completion_key
        @mask_prompt = mask_prompt
      end

      def process(datum)
        tools = fetch_key(datum, "tools")
        messages = [
          { "role" => "user", "content" => fetch_key(datum, prompt_key) },
          { "role" => "assistant", "content" => fetch_key(datum, completion_key) },
        ]
        tokens = apply_chat_template(messages, tools: tools)
        return [tokens, 0] unless mask_prompt

        offset_tokens = apply_chat_template([messages[0]], tools: tools, add_generation_prompt: true)
        [tokens, offset_tokens.length]
      end

      def [](idx)
        @data[idx]
      end

      def length
        @data.length
      end
      alias_method :size, :length
    end

    class ConcatenatedDataset
      include DatasetHelpers

      def initialize(data)
        @data = data
        @length = @data.sum(&:length)
      end

      def [](idx)
        raise IndexError, "index #{idx} out of bounds" if idx.negative? || idx >= length

        data_idx = nil
        datum = nil
        remaining = idx

        @data.each_with_index do |dataset, i|
          if remaining < dataset.length
            datum = dataset[remaining]
            data_idx = i
            break
          end
          remaining -= dataset.length
        end

        if datum.is_a?(Hash)
          out = datum.dup
          out["_dataset"] = data_idx
          out
        else
          { "value" => datum, "_dataset" => data_idx }
        end
      end

      def process(datum)
        dataset_idx = fetch_key(datum, "_dataset")
        dataset = @data.fetch(dataset_idx)
        payload = datum.is_a?(Hash) && datum.key?("value") ? datum["value"] : datum
        dataset.process(payload)
      end

      def length
        @length
      end
      alias_method :size, :length
    end

    class CacheDataset
      def initialize(data)
        @data = data
        @processed = Array.new(data.length)
      end

      def itemlen(idx)
        item = @data[idx]
        item = item[0] if item.is_a?(Array) && item.length == 2 && item[0].respond_to?(:length)
        item.length
      end

      def [](idx)
        @processed[idx] ||= @data.process(@data[idx])
      end

      def length
        @data.length
      end
      alias_method :size, :length
    end
  end
end
