$LOAD_PATH.unshift File.expand_path("../lib", __dir__)
$LOAD_PATH.unshift File.expand_path("../../mlx-ruby/lib", __dir__)

require "mlx"
require "mlx_lm"
require "minitest/autorun"
require "json"
require "tempfile"
require "fileutils"
require "open3"

module ParityTestHelpers
  FIXTURES_DIR = File.expand_path("fixtures", __dir__)

  def fixtures_dir
    FIXTURES_DIR
  end

  # Run a Python snippet, capture JSON output, return parsed result
  def python_eval(code)
    stdout, stderr, status = Open3.capture3(
      {
        # Linux mlx Python can fail JIT C++ compilation in CI toolchains.
        # Parity checks do not require compiled mode.
        "MLX_DISABLE_COMPILE" => "1",
      },
      "python3", "-c", code
    )
    unless status.success?
      raise <<~MSG
        Python eval failed (exit #{status.exitstatus})
        STDERR:
        #{stderr}
        STDOUT:
        #{stdout}
      MSG
    end

    JSON.parse(stdout)
  rescue Errno::ENOENT
    raise "Python eval failed: python3 is not installed"
  end

  # Assert two flat arrays are element-wise close
  def assert_arrays_close(expected, actual, atol: 1e-5, msg: nil)
    expected = expected.flatten if expected.is_a?(Array) && expected.first.is_a?(Array)
    actual = actual.flatten if actual.is_a?(Array) && actual.first.is_a?(Array)
    assert_equal expected.length, actual.length, "#{msg} - length mismatch"
    expected.zip(actual).each_with_index do |(e, a), i|
      assert_in_delta e, a, atol, "#{msg} - element #{i}: expected #{e}, got #{a}"
    end
  end
end
