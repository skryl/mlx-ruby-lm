require "fileutils"
require "rbconfig"
require "rake/testtask"
require_relative "tasks/onnx_report_task"
require_relative "tasks/parity_inventory_task"

VENV_DIR = File.expand_path(".venv-test", __dir__)
VENV_PYTHON = File.join(VENV_DIR, "bin", "python")
REQUIREMENTS_FILE = File.expand_path("requirements.txt", __dir__)
TEST_DEVICE_CHOICES = %w[cpu gpu].freeze
DEFAULT_TEST_DEVICES = %w[cpu gpu].freeze
GEMSPEC_FILE = File.expand_path("mlx-ruby-lm.gemspec", __dir__)
GEM_VERSION_FILE = File.expand_path("lib/mlx_lm/version.rb", __dir__)
TEST_GEM_HOME = File.expand_path("tmp/gem_home", __dir__)
TEST_GEM_SCRIPT = File.expand_path("tmp/gem_version_smoke_test.rb", __dir__)

def parse_test_devices(args)
  raw_values = []
  raw_values << args[:devices] if args[:devices]
  raw_values.concat(args.extras) if args.respond_to?(:extras)

  values = if raw_values.empty?
    DEFAULT_TEST_DEVICES.dup
  else
    raw_values
      .flat_map { |value| value.to_s.split(",") }
      .map(&:strip)
      .reject(&:empty?)
      .map(&:downcase)
  end

  values = DEFAULT_TEST_DEVICES.dup if values.empty?
  invalid = values - TEST_DEVICE_CHOICES
  unless invalid.empty?
    raise ArgumentError, "invalid test device(s): #{invalid.join(', ')} (supported: #{TEST_DEVICE_CHOICES.join(', ')})"
  end

  values.uniq
end

Rake::TestTask.new("test:run") do |t|
  t.libs << "test" << "lib"
  t.test_files = FileList["test/**/*_test.rb"]
end

desc "Run tests on devices (default: cpu then gpu). Examples: rake test, rake \"test[cpu]\", rake \"test[cpu,gpu]\""
task :test, [:devices] do |_task, args|
  devices = parse_test_devices(args)

  devices.each do |device|
    puts "\n==> Running test suite on #{device.upcase}"

    previous_mlx_default_device = ENV["MLX_DEFAULT_DEVICE"]
    previous_device = ENV["DEVICE"]

    begin
      ENV["MLX_DEFAULT_DEVICE"] = device
      ENV["DEVICE"] = device
      Rake::Task["test:run"].reenable
      Rake::Task["test:run"].invoke
    ensure
      if previous_mlx_default_device.nil?
        ENV.delete("MLX_DEFAULT_DEVICE")
      else
        ENV["MLX_DEFAULT_DEVICE"] = previous_mlx_default_device
      end

      if previous_device.nil?
        ENV.delete("DEVICE")
      else
        ENV["DEVICE"] = previous_device
      end
    end
  end
end

namespace :test do
  desc "Install Python dependencies required by parity tests"
  task :deps do
    next unless File.exist?(REQUIREMENTS_FILE)

    sh("python3 -m venv #{VENV_DIR}") unless File.exist?(VENV_PYTHON)
    sh("#{VENV_PYTHON} -m pip install --upgrade pip")
    sh("#{VENV_PYTHON} -m pip install -r #{REQUIREMENTS_FILE}")
  end

  Rake::TestTask.new(:parity) do |t|
    t.libs << "test" << "lib"
    t.test_files = FileList["test/parity/**/*_test.rb"]
  end

  desc "Run full test suite including ONNX full export tests"
  task :all do
    previous_full_export = ENV["ONNX_FULL_EXPORT"]
    ENV["ONNX_FULL_EXPORT"] = "1"

    begin
      Rake::Task[:test].reenable
      Rake::Task[:test].invoke
    ensure
      if previous_full_export.nil?
        ENV.delete("ONNX_FULL_EXPORT")
      else
        ENV["ONNX_FULL_EXPORT"] = previous_full_export
      end
    end
  end

  desc "Build gem, install into tmp/gem_home, and print installed gem version"
  task :gem do
    spec = Gem::Specification.load(GEMSPEC_FILE)
    raise "Could not load gemspec: #{GEMSPEC_FILE}" unless spec

    Rake::Task["gem:build"].invoke

    gem_file = File.expand_path("#{spec.name}-#{spec.version}.gem", __dir__)
    raise "Built gem artifact not found: #{gem_file}" unless File.exist?(gem_file)

    FileUtils.rm_rf(TEST_GEM_HOME)
    FileUtils.mkdir_p(TEST_GEM_HOME)
    FileUtils.mkdir_p(File.dirname(TEST_GEM_SCRIPT))

    smoke_script = <<~RUBY
      require "rubygems"
      gem_spec = Gem::Specification.find_by_name("mlx-ruby-lm")
      version_file = File.join(gem_spec.full_gem_path, "lib", "mlx_lm", "version.rb")
      load version_file
      puts MlxLm::VERSION
    RUBY
    File.write(TEST_GEM_SCRIPT, smoke_script)

    gem_env = {
      "GEM_HOME" => TEST_GEM_HOME,
      "GEM_PATH" => TEST_GEM_HOME,
      "RUBYOPT" => nil,
      "BUNDLE_BIN_PATH" => nil,
      "BUNDLE_GEMFILE" => nil,
      "BUNDLE_PATH" => nil,
      "BUNDLE_WITH" => nil,
      "BUNDLE_WITHOUT" => nil,
      "RUBYGEMS_GEMDEPS" => nil,
    }

    run_smoke_test = proc do
      sh(
        gem_env,
        "gem", "install", gem_file,
        "--install-dir", TEST_GEM_HOME,
        "--local",
        "--ignore-dependencies",
        "--no-document"
      )

      sh(gem_env, RbConfig.ruby, TEST_GEM_SCRIPT)
    end

    if defined?(Bundler) && Bundler.respond_to?(:with_unbundled_env)
      Bundler.with_unbundled_env { run_smoke_test.call }
    else
      run_smoke_test.call
    end
  ensure
    FileUtils.rm_f(TEST_GEM_SCRIPT)
  end
end

namespace :parity do
  desc "Regenerate the Python/Ruby parity inventory snapshot"
  task :inventory do
    ParityInventoryTask.run!
  end

  desc "Verify the parity inventory snapshot is up-to-date"
  task :inventory_check do
    next if ParityInventoryTask.run!(check: true)

    raise "parity inventory snapshot is stale"
  end
end

namespace :onnx do
  desc "Run compat-only ONNX suite and generate report artifacts under test/reports"
  task :report do
    OnnxReportTask.run!
  end
end

namespace :gem do
  desc "Bump gem version by 0.0.0.1 in lib/mlx_lm/version.rb."
  task :bump do
    content = File.read(GEM_VERSION_FILE)
    version_pattern = /^(\s*VERSION\s*=\s*")([^"]+)(")\s*$/
    match = content.match(version_pattern)
    raise "Could not find VERSION assignment in #{GEM_VERSION_FILE}" unless match

    old_version = match[2]
    segments = old_version.split(".")
    unless segments.all? { |segment| segment.match?(/\A\d+\z/) } && segments.length <= 4
      raise "Expected VERSION in numeric dotted format with up to 4 segments, got #{old_version.inspect}"
    end

    numeric_segments = segments.map(&:to_i)
    numeric_segments << 0 while numeric_segments.length < 4
    numeric_segments[3] += 1
    new_version = numeric_segments.join(".")

    updated = content.sub(version_pattern) { "#{Regexp.last_match(1)}#{new_version}#{Regexp.last_match(3)}" }
    File.write(GEM_VERSION_FILE, updated)
    puts "Bumped version: #{old_version} -> #{new_version}"
  end

  desc "Build gem package from mlx-ruby-lm.gemspec."
  task :build do
    sh("gem", "build", GEMSPEC_FILE)
  end

  desc "Publish built gem package to RubyGems."
  task push: :build do
    spec = Gem::Specification.load(GEMSPEC_FILE)
    raise "Could not load gemspec: #{GEMSPEC_FILE}" unless spec

    gem_file = File.expand_path("#{spec.name}-#{spec.version}.gem", __dir__)
    raise "Built gem artifact not found: #{gem_file}" unless File.exist?(gem_file)

    sh("gem", "push", gem_file)
  end
end

task default: :test
