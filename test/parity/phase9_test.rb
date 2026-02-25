require_relative "../test_helper"

# Phase 9: LoRA & Fine-Tuning
# Tests LoRA layers, model application, training step, and adapter save/load.
class Phase9LoRATest < Minitest::Test
  include ParityTestHelpers

  def setup
    @mx = MLX::Core
  end

  # Test 1: LoRALinear forward pass produces correct shape
  def test_lora_linear_forward
    linear = MLX::NN::Linear.new(32, 64, bias: false)
    @mx.eval(*linear.parameters.values)
    lora = MlxLm::Tuner::LoRALinear.from_base(linear, r: 8, scale: 20.0)
    @mx.eval(*MLX::Utils.tree_flatten(lora.parameters).map { |_, v| v })

    x = @mx.ones([1, 32]).astype(@mx.float32)
    output = lora.call(x)
    assert_equal [1, 64], output.shape
  end

  # Test 2: LoRALinear with zero lora_b produces same output as base
  def test_lora_linear_zero_init
    linear = MLX::NN::Linear.new(32, 64, bias: false)
    @mx.eval(*linear.parameters.values)
    lora = MlxLm::Tuner::LoRALinear.from_base(linear, r: 8, scale: 20.0)
    @mx.eval(*MLX::Utils.tree_flatten(lora.parameters).map { |_, v| v })

    x = @mx.ones([1, 32]).astype(@mx.float32)
    base_out = linear.call(x)
    lora_out = lora.call(x)
    @mx.eval(base_out, lora_out)

    # Since lora_b is initialized to zeros, outputs should match
    diff = @mx.max(@mx.abs(base_out - lora_out)).item
    assert diff < 1e-5, "Zero-initialized LoRA should produce same output, diff=#{diff}"
  end

  # Test 3: LoRALinear fuse merges weights correctly
  def test_lora_linear_fuse
    linear = MLX::NN::Linear.new(32, 64, bias: false)
    @mx.eval(*linear.parameters.values)
    lora = MlxLm::Tuner::LoRALinear.from_base(linear, r: 8, scale: 20.0)
    @mx.eval(*MLX::Utils.tree_flatten(lora.parameters).map { |_, v| v })

    x = @mx.ones([1, 32]).astype(@mx.float32)
    lora_out = lora.call(x)

    fused = lora.fuse
    fused_out = fused.call(x)
    @mx.eval(lora_out, fused_out)

    diff = @mx.max(@mx.abs(lora_out - fused_out)).item
    assert diff < 1e-4, "Fused output should match LoRA output, diff=#{diff}"
  end

  # Test 4: LoRAEmbedding forward pass
  def test_lora_embedding_forward
    embed = MLX::NN::Embedding.new(100, 32)
    @mx.eval(*embed.parameters.values)
    lora_embed = MlxLm::Tuner::LoRAEmbedding.from_base(embed, r: 8, scale: 20.0)
    @mx.eval(*MLX::Utils.tree_flatten(lora_embed.parameters).map { |_, v| v })

    ids = @mx.array([[1, 2, 3]]).astype(@mx.int32)
    output = lora_embed.call(ids)
    assert_equal [1, 3, 32], output.shape
  end

  # Test 5: apply_lora_layers converts model layers
  def test_apply_lora_layers
    args = MlxLm::Models::Llama::ModelArgs.from_dict({
      "model_type" => "llama",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 128,
      "vocab_size" => 128,
      "rms_norm_eps" => 1e-6,
      "tie_word_embeddings" => true,
    })
    model = MlxLm::Models::Llama::Model.new(args)

    lora_config = { "rank" => 8, "scale" => 20.0, "dropout" => 0.0 }
    MlxLm::Tuner.apply_lora_layers(model, num_layers: 2, config: lora_config)

    # Check that some linear layers are now LoRA
    has_lora = false
    model.named_modules.each do |name, mod|
      if mod.is_a?(MlxLm::Tuner::LoRALinear)
        has_lora = true
        break
      end
    end
    assert has_lora, "Model should have LoRA layers after apply_lora_layers"
  end

  # Test 6: Freeze non-LoRA parameters
  def test_freeze_non_lora
    args = MlxLm::Models::Llama::ModelArgs.from_dict({
      "model_type" => "llama",
      "hidden_size" => 64,
      "num_hidden_layers" => 2,
      "num_attention_heads" => 2,
      "num_key_value_heads" => 2,
      "intermediate_size" => 128,
      "vocab_size" => 128,
      "rms_norm_eps" => 1e-6,
      "tie_word_embeddings" => true,
    })
    model = MlxLm::Models::Llama::Model.new(args)
    @mx.eval(*MLX::Utils.tree_flatten(model.parameters).map { |_, v| v })

    MlxLm::Tuner.apply_lora_layers(model, num_layers: 2, config: { "rank" => 8 })

    # Freeze base model, only LoRA params should be trainable
    model.freeze
    model.named_modules.each do |name, mod|
      if mod.is_a?(MlxLm::Tuner::LoRALinear)
        mod.unfreeze(keys: ["lora_a", "lora_b"])
      end
    end

    trainable = MLX::Utils.tree_flatten(model.trainable_parameters)
    total = MLX::Utils.tree_flatten(model.parameters)

    assert trainable.length > 0, "Should have trainable LoRA parameters"
    assert trainable.length < total.length, "Trainable params should be fewer than total"
  end

  # Test 7: One training step computes gradients
  def test_training_step
    linear = MLX::NN::Linear.new(4, 2, bias: false)
    @mx.eval(*linear.parameters.values)
    lora = MlxLm::Tuner::LoRALinear.from_base(linear, r: 2, scale: 1.0)
    @mx.eval(*MLX::Utils.tree_flatten(lora.parameters).map { |_, v| v })

    loss_fn = MLX::NN.value_and_grad(lora, ->(m, x) {
      out = m.call(x)
      @mx.mean(out)
    })

    x = @mx.ones([1, 4]).astype(@mx.float32)
    loss, grads = loss_fn.call(lora, x)
    @mx.eval(loss)
    assert loss.item.is_a?(Numeric), "Loss should be a number"
  end

  # Test 8: Adapter save and load round-trip
  def test_adapter_save_load
    linear = MLX::NN::Linear.new(32, 64, bias: false)
    @mx.eval(*linear.parameters.values)
    lora = MlxLm::Tuner::LoRALinear.from_base(linear, r: 8, scale: 20.0)
    @mx.eval(*MLX::Utils.tree_flatten(lora.parameters).map { |_, v| v })

    # Get the LoRA weights
    lora_a_before = lora.lora_a
    @mx.eval(lora_a_before)

    # Save adapter weights using Ruby safetensors gem
    dir = File.join(Dir.tmpdir, "mlx_lm_lora_test_#{$$}")
    FileUtils.mkdir_p(dir)
    begin
      adapter_path = File.join(dir, "adapters.safetensors")
      weights = { "lora_a" => lora.lora_a, "lora_b" => lora.lora_b }
      @mx.eval(*weights.values)

      # Serialize using Safetensors gem
      tensors = {}
      weights.each do |name, arr|
        arr = arr.astype(@mx.float32) unless arr.dtype == @mx.float32
        @mx.eval(arr)
        data = arr.tolist
        data = data.flatten if data.is_a?(::Array) && data.first.is_a?(::Array)
        data = [data].flatten
        binary = data.pack("e*")
        tensors[name] = { "dtype" => "float32", "shape" => arr.shape, "data" => binary }
      end
      File.binwrite(adapter_path, Safetensors.serialize(tensors))

      # Load and verify
      loaded = MlxLm::WeightUtils.load_safetensors(adapter_path)
      @mx.eval(*loaded.values)
      diff = @mx.max(@mx.abs(loaded["lora_a"] - lora_a_before)).item
      assert diff < 1e-6, "Loaded adapter weights should match saved weights"
    ensure
      FileUtils.rm_rf(dir)
    end
  end
end
