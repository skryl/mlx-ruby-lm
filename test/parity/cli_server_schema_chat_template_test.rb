require_relative "../test_helper"

# Tests CLI parsing, server request/response schemas, and chat templates.

class CliServerSchemaChatTemplateTest < Minitest::Test
  # Test 1: CLI generate command parses all standard flags
  def test_cli_generate_flags
    args = MlxLm::CLI.parse_args([
      "generate",
      "--model", "mlx-community/Llama-3-8B-4bit",
      "--prompt", "Hello world",
      "--max-tokens", "100",
      "--temp", "0.7",
      "--top-p", "0.9",
      "--seed", "42",
      "--repetition-penalty", "1.1",
      "--repetition-context-size", "20",
    ])
    assert_equal "generate", args[:command]
    assert_equal "mlx-community/Llama-3-8B-4bit", args[:model]
    assert_equal "Hello world", args[:prompt]
    assert_equal 100, args[:max_tokens]
    assert_in_delta 0.7, args[:temp]
    assert_in_delta 0.9, args[:top_p]
    assert_equal 42, args[:seed]
    assert_in_delta 1.1, args[:repetition_penalty]
    assert_equal 20, args[:repetition_context_size]
  end

  # Test 2: CLI server command parses host and port
  def test_cli_server_flags
    args = MlxLm::CLI.parse_args([
      "server",
      "--model", "mlx-community/Llama-3-8B-4bit",
      "--host", "0.0.0.0",
      "--port", "8080",
    ])
    assert_equal "server", args[:command]
    assert_equal "0.0.0.0", args[:host]
    assert_equal 8080, args[:port]
  end

  # Test 3: CLI chat command parses system prompt
  def test_cli_chat_flags
    args = MlxLm::CLI.parse_args([
      "chat",
      "--model", "mlx-community/Llama-3-8B-4bit",
      "--max-tokens", "512",
    ])
    assert_equal "chat", args[:command]
    assert_equal 512, args[:max_tokens]
  end

  # Test 4: CLI defaults are sensible
  def test_cli_defaults
    args = MlxLm::CLI.parse_args(["generate", "--model", "test-model"])
    assert_equal 256, args[:max_tokens]
    assert_in_delta 0.0, args[:temp]
    assert_equal "127.0.0.1", args[:host]
    assert_equal 8080, args[:port]
  end

  # Test 5: CLI unknown command raises
  def test_cli_unknown_command
    assert_raises(ArgumentError) do
      MlxLm::CLI.parse_args(["unknown_command"])
    end
  end
end

class CliServerSchemaChatTemplateChatCompletionRequestParsingTest < Minitest::Test
  # Test 6: Chat completion request schema parsed correctly
  def test_chat_completion_request_parsing
    request_body = {
      "model" => "test-model",
      "messages" => [
        { "role" => "system", "content" => "You are a helpful assistant." },
        { "role" => "user", "content" => "Hello!" },
      ],
      "max_tokens" => 100,
      "temperature" => 0.7,
      "top_p" => 0.9,
      "stream" => false,
    }
    req = MlxLm::Server::ChatCompletionRequest.from_hash(request_body)
    assert_equal "test-model", req.model
    assert_equal 2, req.messages.length
    assert_equal "system", req.messages[0]["role"]
    assert_equal "user", req.messages[1]["role"]
    assert_equal 100, req.max_tokens
    assert_in_delta 0.7, req.temperature
    assert_in_delta 0.9, req.top_p
    assert_equal false, req.stream
  end

  # Test 7: Chat completion response matches OpenAI schema
  def test_chat_completion_response_schema
    resp = MlxLm::Server::ChatCompletionResponse.new(
      model: "test-model",
      content: "Hello! How can I help?",
      prompt_tokens: 10,
      completion_tokens: 5,
      finish_reason: "stop"
    )
    json = resp.to_hash

    assert_equal "chat.completion", json["object"]
    assert json.key?("id")
    assert json.key?("created")
    assert_equal "test-model", json["model"]
    assert_equal 1, json["choices"].length

    choice = json["choices"][0]
    assert_equal 0, choice["index"]
    assert_equal "stop", choice["finish_reason"]
    assert_equal "assistant", choice["message"]["role"]
    assert_equal "Hello! How can I help?", choice["message"]["content"]

    usage = json["usage"]
    assert_equal 10, usage["prompt_tokens"]
    assert_equal 5, usage["completion_tokens"]
    assert_equal 15, usage["total_tokens"]
  end

  # Test 8: Streaming chunk response matches OpenAI SSE format
  def test_streaming_chunk_response
    chunk = MlxLm::Server::ChatCompletionChunk.new(
      model: "test-model",
      content: "Hello",
      finish_reason: nil
    )
    json = chunk.to_hash

    assert_equal "chat.completion.chunk", json["object"]
    assert json.key?("id")
    assert_equal "test-model", json["model"]

    delta = json["choices"][0]["delta"]
    assert_equal "Hello", delta["content"]
    assert_nil json["choices"][0]["finish_reason"]
  end

  # Test 9: Streaming final chunk has finish_reason
  def test_streaming_final_chunk
    chunk = MlxLm::Server::ChatCompletionChunk.new(
      model: "test-model",
      content: "",
      finish_reason: "stop"
    )
    json = chunk.to_hash
    assert_equal "stop", json["choices"][0]["finish_reason"]
  end

  # Test 10: Models list response matches OpenAI schema
  def test_models_list_response
    resp = MlxLm::Server::ModelsListResponse.new(models: ["model-a", "model-b"])
    json = resp.to_hash

    assert_equal "list", json["object"]
    assert_equal 2, json["data"].length
    assert_equal "model-a", json["data"][0]["id"]
    assert_equal "model", json["data"][0]["object"]
    assert_equal "model-b", json["data"][1]["id"]
  end
end

class CliServerSchemaChatTemplateDefaultChatTemplateTest < Minitest::Test
  # Test 11: Default chat template formats messages correctly
  def test_default_chat_template
    messages = [
      { "role" => "system", "content" => "You are helpful." },
      { "role" => "user", "content" => "Hi" },
    ]
    prompt = MlxLm::ChatTemplate.apply(messages)
    assert prompt.include?("You are helpful.")
    assert prompt.include?("Hi")
  end

  # Test 12: Chat template with multi-turn conversation
  def test_multi_turn_chat_template
    messages = [
      { "role" => "user", "content" => "Hello" },
      { "role" => "assistant", "content" => "Hi there!" },
      { "role" => "user", "content" => "How are you?" },
    ]
    prompt = MlxLm::ChatTemplate.apply(messages)
    assert prompt.include?("Hello")
    assert prompt.include?("Hi there!")
    assert prompt.include?("How are you?")
  end
end
