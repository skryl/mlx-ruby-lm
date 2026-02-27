require_relative "../test_helper"

class PhaseIClassParityClosureSurfaceTest < Minitest::Test
  CHECKLIST_PATH = File.expand_path("../../prd/2026_02_25_python_ruby_parity_checklist.md", __dir__)

  def test_core_runtime_phase_d_classes_exist
    %i[
      GenerationResponse
      BatchStats
      BatchResponse
      Batch
      BatchGenerator
      Response
      NaiveStreamingDetokenizer
      SPMStreamingDetokenizer
      BPEStreamingDetokenizer
      NewlineTokenizer
      ConcatenateKVCache
      BatchKVCache
      BatchRotatingKVCache
    ].each do |const_name|
      assert MlxLm.const_defined?(const_name, false), "expected MlxLm::#{const_name} to be defined"
    end
  end

  def test_server_phase_e_classes_exist
    %i[
      StopCondition
      LRUPromptCache
      CacheEntry
      SearchResult
      ModelDescription
      SamplingArguments
      LogitsProcessorArguments
      GenerationArguments
      CompletionRequest
      GenerationContext
      Response
      TimeBudget
      ModelProvider
      ResponseGenerator
      APIHandler
    ].each do |const_name|
      assert MlxLm::Server.const_defined?(const_name, false),
        "expected MlxLm::Server::#{const_name} to be defined"
    end
  end

  def test_parity_alias_model_classes_exist
    rows = File.readlines(CHECKLIST_PATH).grep(/^\| /)[2..]
    alias_rows = rows.map { |line| line.split("|").map(&:strip) }.select do |cols|
      cols[4] == "Implemented" && cols[6].to_s.include?("parity compatibility alias class")
    end

    assert_operator alias_rows.length, :>, 100, "expected a large alias-backed closure set"

    alias_rows.each do |cols|
      ruby_ref = cols[5]
      py_class = cols[3]
      next unless ruby_ref.start_with?("models/")

      mod = resolve_model_namespace(ruby_ref)
      assert mod, "could not resolve module namespace for #{ruby_ref}"
      assert mod.const_defined?(py_class.to_sym, false),
        "expected #{mod}::#{py_class} to be defined"
    end
  end

  private

  def resolve_model_namespace(ruby_ref)
    case ruby_ref
    when "models/cache.rb"
      return MlxLm
    when "models/switch_layers.rb"
      return MlxLm::Models::SwitchLayers
    when "models/pipeline.rb"
      return MlxLm::Models
    end

    path = File.expand_path("../../lib/mlx_lm/#{ruby_ref}", __dir__)
    return nil unless File.exist?(path)

    text = File.read(path)
    match = text.match(/module\s+MlxLm\s*\n\s*module\s+Models\s*\n\s*module\s+([A-Za-z0-9_]+)/m)
    return nil unless match
    return nil unless MlxLm::Models.const_defined?(match[1].to_sym, false)

    MlxLm::Models.const_get(match[1].to_sym, false)
  end
end
