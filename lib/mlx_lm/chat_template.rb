module MlxLm
  module ChatTemplate
    module_function

    # Apply a simple chat template to format messages into a prompt string.
    # This is a default/fallback template. Model-specific templates (like
    # Jinja-based ones from tokenizer_config.json) can override this.
    def apply(messages, template: :default)
      case template
      when :default
        apply_default(messages)
      when :chatml
        apply_chatml(messages)
      else
        apply_default(messages)
      end
    end

    # Default template: ChatML-like format
    # <|im_start|>system
    # content<|im_end|>
    # <|im_start|>user
    # content<|im_end|>
    # <|im_start|>assistant
    def apply_default(messages)
      parts = []
      messages.each do |msg|
        role = msg["role"] || msg[:role]
        content = msg["content"] || msg[:content]
        parts << "<|im_start|>#{role}\n#{content}<|im_end|>"
      end
      parts << "<|im_start|>assistant"
      parts.join("\n")
    end

    # ChatML template (same as default, widely used)
    def apply_chatml(messages)
      apply_default(messages)
    end
  end
end
