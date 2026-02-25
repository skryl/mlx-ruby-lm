module MlxLm
  module Models
    # Model registry: maps architecture name to [Model, ModelArgs] classes.
    # Additional architectures register themselves here.
    REGISTRY = {}

    # Remapping for architectures that share implementation
    REMAPPING = {
      "mistral" => "llama",
    }.freeze

    module_function

    def register(name, model_class, args_class)
      REGISTRY[name] = [model_class, args_class]
    end

    def get_classes(config)
      model_type = config["model_type"]
      raise ArgumentError, "config.json missing 'model_type' field" unless model_type

      # Apply remapping
      canonical = REMAPPING.fetch(model_type, model_type)

      unless REGISTRY.key?(canonical)
        raise ArgumentError, "Model architecture '#{model_type}' (canonical: '#{canonical}') not found in registry. Available: #{REGISTRY.keys.join(', ')}"
      end

      REGISTRY[canonical]
    end
  end
end
