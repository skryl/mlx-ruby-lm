require "rake/testtask"

Rake::TestTask.new(:test) do |t|
  t.libs << "test" << "lib"
  t.test_files = FileList["test/**/*_test.rb"]
end

namespace :test do
  Rake::TestTask.new(:parity) do |t|
    t.libs << "test" << "lib"
    t.test_files = FileList["test/parity/**/*_test.rb"]
  end
end

namespace :parity do
  desc "Regenerate the Python/Ruby parity inventory snapshot"
  task :inventory do
    ruby "scripts/generate_parity_inventory.rb"
  end

  desc "Verify the parity inventory snapshot is up-to-date"
  task :inventory_check do
    ruby "scripts/generate_parity_inventory.rb", "--check"
  end
end

task default: :test
