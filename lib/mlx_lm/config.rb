require "json"

module MlxLm
  module Config
    module_function

    # Load model config from a directory containing config.json
    # and optionally generation_config.json.
    # Mirrors Python mlx_lm.utils.load_config
    def load(model_path)
      config_path = File.join(model_path, "config.json")
      config = JSON.parse(File.read(config_path))

      gen_config_path = File.join(model_path, "generation_config.json")
      if File.exist?(gen_config_path)
        begin
          gen_config = JSON.parse(File.read(gen_config_path))
        rescue JSON::ParserError
          gen_config = {}
        end

        if (eos = gen_config["eos_token_id"])
          config["eos_token_id"] = eos
        end
      end

      config
    end
  end
end
