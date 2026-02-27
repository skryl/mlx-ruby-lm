# frozen_string_literal: true

require_relative "../test_helper"

class RopeMlaCacheIntegrationSmokeTest < Minitest::Test
  def test_phase14_components_are_available_from_top_level_require
    assert defined?(MlxLm::Models::SuScaledRoPE)
    assert defined?(MlxLm::Models::Llama3RoPE)
    assert defined?(MlxLm::Models::YarnRoPE)
    assert defined?(MlxLm::Models::MLA::MultiLinear)
    assert defined?(MlxLm::KVCache)
    assert defined?(MlxLm::RotatingKVCache)
  end

  def test_phase14_basic_construction_smoke
    rope = MlxLm::Models.initialize_rope(8, 10_000.0, false)
    mla = MlxLm::Models::MLA::MultiLinear.new(8, 4, 2)
    cache = MlxLm::KVCache.new
    rotating_cache = MlxLm::RotatingKVCache.new(max_size: 16, keep: 4)

    assert_instance_of MLX::NN::RoPE, rope
    assert_instance_of MlxLm::Models::MLA::MultiLinear, mla
    assert_instance_of MlxLm::KVCache, cache
    assert_instance_of MlxLm::RotatingKVCache, rotating_cache
  end
end
