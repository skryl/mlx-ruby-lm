# frozen_string_literal: true

require "json"
require "open3"
require "pathname"
require "time"

class OnnxReportTask
  ROOT = Pathname.new(__dir__).join("..").expand_path
  REPORT_DIR = ROOT.join("test", "reports")
  RAW_OUTPUT_PATH = REPORT_DIR.join("onnx_compat_test_output.txt")
  JSON_PATH = REPORT_DIR.join("onnx_compat_full_report.json")
  MARKDOWN_PATH = REPORT_DIR.join("onnx_compat_full_report.md")
  INVOCATIONS_CSV_PATH = REPORT_DIR.join("onnx_compat_missing_ops_invocations.csv")

  COMMAND = [
    "bundle", "exec", "rake", "test",
    "TEST=test/onnx/*_test.rb",
    "TESTOPTS=--name=/test_onnx_compat_report/"
  ].freeze

  STATUS_LINE_RE = /^\s*\[ONNX\]\s+(?<model>[^:]+):\s+(?:(?<status>PASS|FAIL)\s+[—-]\s+)?(?<supported>\d+)\/(?<total>\d+)\s+nodes(?:\s+\((?<pct>[0-9.]+)%\))?\s+[—-]\s+missing:\s*(?<missing>.*?)(?:\s+\((?<suffix>CRASH during export)\))?\s*$/.freeze
  JSON_LINE_RE = /^\s*\[ONNX-JSON\]\s+(?<model>[^:]+):\s+(?<payload>\{.*\})\s*$/.freeze
  SUMMARY_LINE_RE = /(?<runs>\d+)\s+runs,\s+(?<assertions>\d+)\s+assertions,\s+(?<failures>\d+)\s+failures,\s+(?<errors>\d+)\s+errors,\s+(?<skips>\d+)\s+skips/.freeze

  def self.run!
    REPORT_DIR.mkpath
    output, status = run_compat_suite
    RAW_OUTPUT_PATH.write(output)

    models, test_summary, warnings = parse_output(output)
    model_rows = models.values.sort_by { |entry| entry["model_type"] }

    unsupported_union = model_rows
      .flat_map { |entry| entry.fetch("missing_ops", []) }
      .uniq
      .sort

    missing_op_model_counts = Hash.new(0)
    model_rows.each do |entry|
      entry.fetch("missing_ops", []).uniq.each { |op| missing_op_model_counts[op] += 1 }
    end

    summary = {
      "models_total" => model_rows.length,
      "models_with_missing_ops" => model_rows.count { |entry| entry.fetch("missing_ops", []).any? },
      "pass_count" => model_rows.count { |entry| entry["status"] == "PASS" },
      "fail_count" => model_rows.count { |entry| entry["status"] == "FAIL" },
      "crash_count" => model_rows.count { |entry| entry["status"] == "CRASH" },
      "unknown_count" => model_rows.count { |entry| entry["status"] == "UNKNOWN" },
      "unsupported_ops_union_size" => unsupported_union.length
    }

    report_payload = {
      "generated_at" => Time.now.utc.iso8601,
      "command" => COMMAND.map { |value| shell_escape(value) }.join(" "),
      "exit_status" => status.exitstatus,
      "test_summary" => test_summary,
      "summary" => summary,
      "unsupported_ops_union" => unsupported_union,
      "missing_op_model_counts" => missing_op_model_counts.sort.to_h,
      "models" => model_rows,
      "warnings" => warnings
    }

    JSON_PATH.write("#{JSON.pretty_generate(report_payload)}\n")
    write_invocation_csv(INVOCATIONS_CSV_PATH, model_rows)
    MARKDOWN_PATH.write(render_markdown(report_payload))

    puts "\nWrote reports:"
    puts "- #{RAW_OUTPUT_PATH}"
    puts "- #{JSON_PATH}"
    puts "- #{INVOCATIONS_CSV_PATH}"
    puts "- #{MARKDOWN_PATH}"
    puts "Models: #{summary['models_total']} | Missing-op models: #{summary['models_with_missing_ops']} | Unsupported ops: #{unsupported_union.join(', ')}"

    return report_payload if status.success?

    exit_code = status.exitstatus || 1
    raise "ONNX compat suite failed with exit code #{exit_code}"
  end

  class << self
    private

    def parse_missing_ops(value)
      text = value.to_s.strip
      return [] if text.empty? || text == "none"

      text.split(",").map(&:strip).reject(&:empty?).uniq.sort
    end

    def shell_escape(value)
      "'" + value.to_s.gsub("'", %q('"'"')) + "'"
    end

    def markdown_escape(value)
      value.to_s.gsub("|", "\\|")
    end

    def extract_invocations(report)
      return [] unless report.is_a?(Hash)

      nodes = report["nodes"]
      return [] unless nodes.is_a?(Array)

      nodes.filter_map do |node|
        next unless node.is_a?(Hash)
        next unless node["supported"] == false

        {
          "index" => node["index"],
          "op" => node["op"],
          "onnx_op_type" => node["onnx_op_type"]
        }
      end
    end

    def run_compat_suite
      env = {
        "ONNX_COMPAT_REPORT_JSON" => "1",
        "ONNX_LOG_LINES" => "1"
      }
      output = +""
      status = nil

      Open3.popen2e(env, *COMMAND, chdir: ROOT.to_s) do |stdin, stdout_and_stderr, wait_thr|
        stdin.close
        stdout_and_stderr.each_line do |line|
          unless line.include?("[ONNX-JSON]") || line.include?("[ONNX-INV]")
            print line
          end
          output << line
        end
        status = wait_thr.value
      end

      [output, status]
    end

    def parse_output(output)
      models = {}
      warnings = []
      summary = nil

      output.each_line do |line|
        if (m = line.match(STATUS_LINE_RE))
          model = m[:model].strip
          models[model] ||= { "model_type" => model }
          status = if m[:suffix] == "CRASH during export"
            "CRASH"
          else
            m[:status] || "UNKNOWN"
          end
          models[model]["status"] = status
          models[model]["supported_nodes"] = m[:supported].to_i
          models[model]["total_nodes"] = m[:total].to_i
          models[model]["coverage_percent"] = if m[:total].to_i.positive?
            ((m[:supported].to_f / m[:total].to_f) * 100).round(1)
          else
            0.0
          end
          models[model]["missing_ops"] = parse_missing_ops(m[:missing])
          next
        end

        if (m = line.match(JSON_LINE_RE))
          model = m[:model].strip
          models[model] ||= { "model_type" => model }
          begin
            models[model]["compat_report"] = JSON.parse(m[:payload])
          rescue JSON::ParserError => e
            warnings << "#{model}: failed to parse ONNX-JSON line (#{e.message})"
          end
          next
        end

        if (m = line.match(SUMMARY_LINE_RE))
          summary = {
            "runs" => m[:runs].to_i,
            "assertions" => m[:assertions].to_i,
            "failures" => m[:failures].to_i,
            "errors" => m[:errors].to_i,
            "skips" => m[:skips].to_i
          }
        end
      end

      models.each_value do |entry|
        report = entry["compat_report"]
        if report.is_a?(Hash)
          entry["supported_nodes"] ||= report["supported_nodes"]
          entry["total_nodes"] ||= report["total_nodes"]
          entry["missing_ops"] ||= Array(report["unsupported_ops"]).map(&:to_s).sort
          if entry["total_nodes"].to_i.positive?
            entry["coverage_percent"] ||= ((entry["supported_nodes"].to_f / entry["total_nodes"].to_f) * 100).round(1)
          else
            entry["coverage_percent"] ||= 0.0
          end
        else
          entry["missing_ops"] ||= []
        end

        entry["status"] ||= "UNKNOWN"
        entry["unsupported_invocations"] = extract_invocations(report)
      end

      [models, summary, warnings]
    end

    def write_invocation_csv(path, model_rows)
      escape = lambda do |value|
        text = value.nil? ? "" : value.to_s
        if text.include?(",") || text.include?("\"") || text.include?("\n")
          "\"#{text.gsub("\"", "\"\"")}\""
        else
          text
        end
      end

      lines = []
      lines << "model_type,status,missing_op,onnx_op_type,node_index"
      model_rows.each do |row|
        invocations = row.fetch("unsupported_invocations", [])
        if invocations.empty?
          row.fetch("missing_ops", []).each do |op|
            values = [row["model_type"], row["status"], op, nil, nil]
            lines << values.map { |v| escape.call(v) }.join(",")
          end
        else
          invocations.each do |inv|
            values = [row["model_type"], row["status"], inv["op"], inv["onnx_op_type"], inv["index"]]
            lines << values.map { |v| escape.call(v) }.join(",")
          end
        end
      end

      path.write("#{lines.join("\n")}\n")
    end

    def markdown_table_row(values)
      "| #{values.map { |v| markdown_escape(v) }.join(' | ')} |"
    end

    def render_markdown(report_payload)
      model_rows = report_payload.fetch("models")
      invocation_rows = model_rows.flat_map do |model|
        model.fetch("unsupported_invocations", []).map do |inv|
          [model["model_type"], inv["op"], inv["onnx_op_type"], inv["index"]]
        end
      end

      lines = []
      lines << "# ONNX Compat Report"
      lines << ""
      lines << "- Generated at: `#{report_payload.fetch("generated_at")}`"
      lines << "- Command: `#{report_payload.fetch("command")}`"
      lines << "- Exit status: `#{report_payload.fetch("exit_status")}`"
      lines << "- Models: `#{report_payload.dig("summary", "models_total")}`"
      lines << "- Models with missing ops: `#{report_payload.dig("summary", "models_with_missing_ops")}`"
      lines << "- Unsupported op union size: `#{report_payload.dig("summary", "unsupported_ops_union_size")}`"
      lines << ""

      test_summary = report_payload["test_summary"]
      if test_summary
        lines << "## Test Summary"
        lines << ""
        lines << markdown_table_row(%w[Runs Assertions Failures Errors Skips])
        lines << markdown_table_row(%w[--- --- --- --- ---])
        lines << markdown_table_row([
          test_summary["runs"],
          test_summary["assertions"],
          test_summary["failures"],
          test_summary["errors"],
          test_summary["skips"]
        ])
        lines << ""
      end

      lines << "## Per-Model Coverage"
      lines << ""
      lines << markdown_table_row(["Model", "Status", "Supported/Total", "Coverage %", "Missing Ops"])
      lines << markdown_table_row(%w[--- --- --- --- ---])
      model_rows.each do |row|
        lines << markdown_table_row([
          row["model_type"],
          row["status"],
          "#{row["supported_nodes"]}/#{row["total_nodes"]}",
          row["coverage_percent"],
          row.fetch("missing_ops", []).join(", ")
        ])
      end
      lines << ""

      lines << "## Unsupported Ops Union"
      lines << ""
      unsupported_union = report_payload.fetch("unsupported_ops_union")
      if unsupported_union.empty?
        lines << "none"
      else
        unsupported_union.each do |op|
          count = report_payload.dig("missing_op_model_counts", op).to_i
          lines << "- `#{op}`: #{count} model(s)"
        end
      end
      lines << ""

      lines << "## Unsupported Node Invocations"
      lines << ""
      if invocation_rows.empty?
        lines << "none"
      else
        lines << markdown_table_row(["Model", "Op", "ONNX op type", "Node index"])
        lines << markdown_table_row(%w[--- --- --- ---])
        invocation_rows.each do |row|
          lines << markdown_table_row(row)
        end
      end
      lines << ""

      warnings = report_payload.fetch("warnings", [])
      unless warnings.empty?
        lines << "## Warnings"
        lines << ""
        warnings.each { |warning| lines << "- #{warning}" }
        lines << ""
      end

      lines.join("\n")
    end
  end
end
