require "json"

module MlxLm
  module LoadUtils
    module_function

    # Load a model and tokenizer from a local directory.
    #
    # @param model_path [String] Path to the model directory
    # @param tokenizer_config [Hash] Additional tokenizer config overrides
    # @return [Array(nn::Module, TokenizerWrapper)] The loaded model and tokenizer
    def load(model_path, tokenizer_config: nil)
      model, _config = load_model(model_path)
      tokenizer = load_tokenizer(model_path)
      [model, tokenizer]
    end

    # Load model from a local directory containing config.json and safetensors.
    #
    # @param model_path [String] Path to the model directory
    # @return [Array(nn::Module, Hash)] The loaded model and config
    def load_model(model_path)
      config = Config.load(model_path)

      # Get model and args classes from registry
      model_class, args_class = Models.get_classes(config)

      # Instantiate model args from config
      model_args = args_class.from_dict(config)

      # Create model
      model = model_class.new(model_args)

      # Load weights
      weights = WeightUtils.load_sharded_safetensors(model_path)

      # Apply model-specific weight sanitization
      if model.respond_to?(:sanitize)
        weights = model.sanitize(weights)
      end

      # Apply quantization if config specifies it
      quantization = config["quantization"]
      if quantization
        group_size = quantization["group_size"] || 64
        bits = quantization["bits"] || 4
        Quantize.quantize_model(model, group_size: group_size, bits: bits, weights: weights)
      end

      # Load weights into model
      model.load_weights(weights, strict: false)

      [model, config]
    end

    # Load tokenizer from a local directory.
    #
    # @param model_path [String] Path containing tokenizer files
    # @return [TokenizerWrapper] The loaded tokenizer
    def load_tokenizer(model_path)
      tokenizer_path = File.join(model_path, "tokenizer.json")
      raise "Tokenizer not found at #{tokenizer_path}" unless File.exist?(tokenizer_path)

      tokenizer = Tokenizers::Tokenizer.from_file(tokenizer_path)

      # Try to load tokenizer config for EOS token
      config_path = File.join(model_path, "tokenizer_config.json")
      eos_token = nil
      eos_token_id = nil
      if File.exist?(config_path)
        tc = JSON.parse(File.read(config_path))
        eos_token = tc["eos_token"]
        eos_token = eos_token["content"] if eos_token.is_a?(Hash)
      end

      # Try to get eos_token_id from config.json
      model_config_path = File.join(model_path, "config.json")
      if File.exist?(model_config_path)
        mc = JSON.parse(File.read(model_config_path))
        eos_token_id = mc["eos_token_id"]
        eos_token_id = eos_token_id.is_a?(::Array) ? eos_token_id : [eos_token_id].compact
      end

      TokenizerWrapper.new(tokenizer, eos_token: eos_token, eos_token_id: eos_token_id)
    end
  end
end
