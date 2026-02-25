Gem::Specification.new do |s|
  s.name        = "mlx-ruby-lm"
  s.version     = File.read(File.expand_path("lib/mlx_lm/version.rb", __dir__))[/VERSION\s*=\s*"(.+?)"/, 1]
  s.summary     = "LLM inference and fine-tuning on MLX for Ruby"
  s.description = "A Ruby port of mlx-lm providing large language model inference and fine-tuning built on the mlx gem."
  s.authors     = ["MLX Ruby Contributors"]
  s.license     = "MIT"
  s.required_ruby_version = ">= 3.3"

  s.files = Dir["lib/**/*.rb"]

  s.add_dependency "safetensors", ">= 0.2"
  s.add_dependency "tokenizers", ">= 0.6"
end
