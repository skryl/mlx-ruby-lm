require "rake/testtask"
require_relative "tasks/onnx_report_task"
require_relative "tasks/parity_inventory_task"

VENV_DIR = File.expand_path(".venv-test", __dir__)
VENV_PYTHON = File.join(VENV_DIR, "bin", "python")
REQUIREMENTS_FILE = File.expand_path("requirements.txt", __dir__)

Rake::TestTask.new(:test) do |t|
  t.libs << "test" << "lib"
  t.test_files = FileList["test/**/*_test.rb"]
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
