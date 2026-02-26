# frozen_string_literal: true

require "json"
require "pathname"

class ParityInventoryTask
  ROOT = Pathname.new(__dir__).join("..").expand_path
  DEFAULT_OUTPUT = ROOT.join("test", "reports", "python_ruby_parity_inventory_snapshot.json")

  PY_MODELS_DIR = ROOT.join("mlx-lm", "mlx_lm", "models")
  RB_MODELS_DIR = ROOT.join("lib", "mlx_lm", "models")
  RB_REGISTRY_FILE = ROOT.join("lib", "mlx_lm", "models.rb")

  PY_INFRA_FILES = %w[
    activations.py
    base.py
    bitlinear_layers.py
    cache.py
    gated_delta.py
    mla.py
    pipeline.py
    rope_utils.py
    ssm.py
    switch_layers.py
  ].freeze

  RB_INFRA_FILES = %w[
    cache.rb
    switch_layers.rb
  ].freeze

  def self.run!(check: false, output: DEFAULT_OUTPUT)
    output_path = resolve_output(output)
    snapshot = build_snapshot

    if check
      check_snapshot(output_path, snapshot)
    else
      write_snapshot(output_path, snapshot)
      puts "wrote parity inventory snapshot: #{output_path}"
      true
    end
  end

  class << self
    private

    def resolve_output(path)
      output = path.is_a?(Pathname) ? path : Pathname.new(path.to_s)
      output.relative? ? ROOT.join(output) : output
    end

    def model_files(dir, ext)
      Dir.glob(dir.join("*.#{ext}").to_s)
         .map { |path| File.basename(path) }
         .reject { |name| name == "__init__.py" }
         .sort
    end

    def parse_registered_model_keys
      keys = []
      Dir.glob(RB_MODELS_DIR.join("*.rb").to_s).sort.each do |path|
        File.read(path).scan(/Models\.register\("([^"]+)"/) do |match|
          keys << match.first
        end
      end
      keys.uniq.sort
    end

    def parse_remappings
      content = File.read(RB_REGISTRY_FILE)
      remap_block = content[/REMAPPING\s*=\s*\{(.*?)\}\s*\.freeze/m, 1]
      return {} if remap_block.nil?

      remappings = {}
      remap_block.scan(/"([^"]+)"\s*=>\s*"([^"]+)"/) do |from, to|
        remappings[from] = to
      end
      remappings
    end

    def build_snapshot
      py_all = model_files(PY_MODELS_DIR, "py")
      py_arch = py_all - PY_INFRA_FILES
      py_arch_keys = py_arch.map { |name| name.sub(/\.py\z/, "") }

      rb_all = model_files(RB_MODELS_DIR, "rb")
      rb_arch = rb_all - RB_INFRA_FILES
      rb_arch_keys = rb_arch.map { |name| name.sub(/\.rb\z/, "") }
      rb_registered = parse_registered_model_keys
      remappings = parse_remappings
      missing_architecture_files = (py_arch_keys - rb_registered).sort
      extra_registered_model_keys = (rb_registered - py_arch_keys).sort

      {
        "inventory_version" => 1,
        "source_paths" => {
          "python_models_dir" => "mlx-lm/mlx_lm/models",
          "ruby_models_dir" => "lib/mlx_lm/models",
          "ruby_registry_file" => "lib/mlx_lm/models.rb"
        },
        "python" => {
          "model_files_total" => py_all.length,
          "shared_infra_files" => PY_INFRA_FILES,
          "architecture_files_total" => py_arch.length,
          "architecture_files" => py_arch
        },
        "ruby" => {
          "model_files_total" => rb_all.length,
          "shared_infra_files" => RB_INFRA_FILES,
          "architecture_files_total" => rb_arch.length,
          "architecture_files" => rb_arch,
          "architecture_model_keys_total" => rb_arch_keys.length,
          "architecture_model_keys" => rb_arch_keys,
          "registered_model_keys_total" => rb_registered.length,
          "registered_model_keys" => rb_registered,
          "remappings_total" => remappings.length,
          "remappings" => remappings
        },
        "parity" => {
          "missing_architecture_file_count" => missing_architecture_files.length,
          "missing_architecture_files" => missing_architecture_files,
          "extra_registered_model_keys_count" => extra_registered_model_keys.length,
          "extra_registered_model_keys" => extra_registered_model_keys
        }
      }
    end

    def write_snapshot(output, snapshot)
      output.dirname.mkpath
      output.write("#{JSON.pretty_generate(snapshot)}\n")
    end

    def check_snapshot(output, snapshot)
      unless output.exist?
        warn "snapshot file missing: #{output}"
        return false
      end

      current = JSON.parse(output.read)
      if current == snapshot
        puts "parity inventory snapshot is up-to-date"
        true
      else
        warn "parity inventory snapshot is stale: #{output}"
        false
      end
    end
  end
end
