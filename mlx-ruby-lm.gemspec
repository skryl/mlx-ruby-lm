require_relative "lib/mlx_lm/version"

Gem::Specification.new do |s|
  s.name        = "mlx-ruby-lm"
  s.version     = MlxLm::VERSION
  s.summary     = "LLM inference and fine-tuning on MLX for Ruby"
  s.description = "A Ruby port of mlx-lm providing large language model inference, " \
                  "quantization, LoRA fine-tuning, and an OpenAI-compatible server " \
                  "built on the mlx gem. Supports Llama, Gemma, Qwen2, Phi3, Mixtral, " \
                  "DeepSeek, and many more architectures."
  s.authors     = ["Alex Skryl"]
  s.email       = ["rut216@gmail.com"]
  s.homepage    = "https://github.com/skryl/mlx-ruby-lm"
  s.license     = "MIT"
  s.required_ruby_version = ">= 3.4"

  s.metadata = {
    "homepage_uri" => s.homepage,
    "source_code_uri" => "https://github.com/skryl/mlx-ruby-lm",
    "changelog_uri" => "https://github.com/skryl/mlx-ruby-lm/blob/main/CHANGELOG.md",
  }

  s.files = Dir.chdir(__dir__) do
    Dir["{lib,exe}/**/*", "LICENSE.txt", "README.md"]
  end
  s.bindir = "exe"
  s.executables = s.files.grep(%r{\Aexe/}) { |f| File.basename(f) }
  s.require_paths = ["lib"]

  s.add_dependency "mlx", ">= 0.30.7.5", "< 1.0"
  s.add_dependency "safetensors", "~> 0.2"
  s.add_dependency "tokenizers", "~> 0.6"

  s.add_development_dependency "minitest", "~> 5.20"
  s.add_development_dependency "ostruct"
  s.add_development_dependency "rake", "~> 13.0"
end
