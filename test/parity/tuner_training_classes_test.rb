require_relative "../test_helper"

class TunerTrainingClassesTest < Minitest::Test
  include ParityTestHelpers

  class FakeTokenizer
    attr_reader :eos_token_id

    def initialize
      @eos_token_id = 99
    end

    def encode(text, add_special_tokens: true)
      text.to_s.bytes.map { |b| b % 31 }
    end

    def apply_chat_template(messages, tools: nil, add_generation_prompt: false, return_dict: false)
      Array(messages)
        .map { |m| "#{m['role'] || m[:role]}:#{m['content'] || m[:content]}" }
        .join("|")
        .bytes
        .map { |b| b % 29 }
    end
  end

  class RecordingClient
    attr_reader :inits, :logs

    def initialize
      @inits = []
      @logs = []
    end

    def init(**kwargs)
      @inits << kwargs
    end

    def log(payload, step: nil)
      @logs << [payload, step]
    end
  end

  def setup
    @mx = MLX::Core
  end

  def test_training_args_defaults_and_overrides
    defaults = MlxLm::Tuner::TrainingArgs.new
    assert_equal 4, defaults.batch_size
    assert_equal 100, defaults.iters
    assert_equal "adapters.safetensors", defaults.adapter_file
    assert_equal false, defaults.grad_checkpoint

    overrides = MlxLm::Tuner::TrainingArgs.new(batch_size: 8, iters: 42)
    assert_equal 8, overrides.batch_size
    assert_equal 42, overrides.iters
  end

  def test_get_reporting_callbacks_builds_chain
    wandb = RecordingClient.new
    swanlab = RecordingClient.new
    callback = MlxLm::Tuner.get_reporting_callbacks(
      report_to: "wandb, swanlab",
      project_name: "proj",
      log_dir: "/tmp/run",
      config: { "lr" => 1e-4 },
      clients: {
        "wandb" => wandb,
        "swanlab" => swanlab,
      }
    )

    payload = { "iteration" => 3, "train_loss" => 0.123 }
    callback.on_train_loss_report(payload)
    callback.on_val_loss_report(payload)

    assert_equal 1, wandb.inits.length
    assert_equal 1, swanlab.inits.length
    assert_equal 2, wandb.logs.length
    assert_equal 2, swanlab.logs.length
  end

  def test_text_chat_and_completions_datasets
    tokenizer = FakeTokenizer.new

    text_ds = MlxLm::Tuner::TextDataset.new(
      [{ "text" => "hello" }],
      tokenizer
    )
    text_tokens, text_offset = text_ds.process(text_ds[0])
    assert_equal 0, text_offset
    assert_equal tokenizer.eos_token_id, text_tokens[-1]

    chat_ds = MlxLm::Tuner::ChatDataset.new(
      [{ "messages" => [{ "role" => "user", "content" => "hi" }, { "role" => "assistant", "content" => "hello" }] }],
      tokenizer,
      mask_prompt: true
    )
    chat_tokens, chat_offset = chat_ds.process(chat_ds[0])
    assert_operator chat_tokens.length, :>, 0
    assert_operator chat_offset, :>, 0

    completion_ds = MlxLm::Tuner::CompletionsDataset.new(
      [{ "prompt" => "Q", "completion" => "A" }],
      tokenizer,
      prompt_key: "prompt",
      completion_key: "completion",
      mask_prompt: true
    )
    comp_tokens, comp_offset = completion_ds.process(completion_ds[0])
    assert_operator comp_tokens.length, :>, 0
    assert_operator comp_offset, :>, 0
  end

  def test_concatenated_and_cache_dataset
    tokenizer = FakeTokenizer.new
    d1 = MlxLm::Tuner::TextDataset.new([{ "text" => "a" }, { "text" => "b" }], tokenizer)
    d2 = MlxLm::Tuner::TextDataset.new([{ "text" => "c" }], tokenizer)
    concat = MlxLm::Tuner::ConcatenatedDataset.new([d1, d2])

    entry = concat[2]
    assert_equal 1, entry["_dataset"]

    cache = MlxLm::Tuner::CacheDataset.new(concat)
    first = cache[2]
    second = cache[2]
    assert_equal first, second
    assert_operator cache.itemlen(2), :>, 0
  end

  def test_dora_linear_forward_and_fuse
    linear = MLX::NN::Linear.new(8, 4, bias: false)
    @mx.eval(*linear.parameters.values)
    dora = MlxLm::Tuner::DoRALinear.from_base(linear, r: 2, scale: 1.0)
    @mx.eval(*MLX::Utils.tree_flatten(dora.parameters).map { |_, v| v })

    x = @mx.ones([1, 8]).astype(@mx.float32)
    base = linear.call(x)
    out = dora.call(x)
    fused = dora.fuse
    fused_out = fused.call(x)
    @mx.eval(base, out, fused_out)

    diff = @mx.max(@mx.abs(base - out)).item
    fuse_diff = @mx.max(@mx.abs(out - fused_out)).item

    assert diff < 1e-5, "DoRA with zero-initialized lora_b should match base linear output"
    assert fuse_diff < 1e-4, "Fused DoRA linear should match unfused output"
  end

  def test_dora_embedding_forward_and_fuse_shape
    embed = MLX::NN::Embedding.new(32, 8)
    @mx.eval(*embed.parameters.values)
    dora = MlxLm::Tuner::DoRAEmbedding.from_base(embed, r: 2, scale: 1.0)
    @mx.eval(*MLX::Utils.tree_flatten(dora.parameters).map { |_, v| v })

    ids = @mx.array([[1, 2, 3]], dtype: @mx.int32)
    out = dora.call(ids)
    fused = dora.fuse
    fused_out = fused.call(ids)
    @mx.eval(out, fused_out)

    assert_equal [1, 3, 8], out.shape
    assert_equal [1, 3, 8], fused_out.shape
  end
end
