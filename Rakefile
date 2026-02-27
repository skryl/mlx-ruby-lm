require "rake/testtask"
require_relative "tasks/onnx_report_task"
require_relative "tasks/parity_inventory_task"

VENV_DIR = File.expand_path(".venv-test", __dir__)
VENV_PYTHON = File.join(VENV_DIR, "bin", "python")
REQUIREMENTS_FILE = File.expand_path("requirements.txt", __dir__)
TEST_DEVICE_CHOICES = %w[cpu gpu].freeze
DEFAULT_TEST_DEVICES = %w[cpu gpu].freeze

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

task default: :test
