require "json"
require "securerandom"

module MlxLm
  module Server
    # Request schema for POST /v1/chat/completions
    class ChatCompletionRequest
      attr_reader :model, :messages, :max_tokens, :temperature, :top_p, :stream, :stop

      def self.from_hash(h)
        new(
          model: h["model"],
          messages: h["messages"] || [],
          max_tokens: h["max_tokens"] || 256,
          temperature: h["temperature"] || 0.0,
          top_p: h["top_p"] || 1.0,
          stream: h.fetch("stream", false),
          stop: h["stop"]
        )
      end

      def initialize(model:, messages:, max_tokens: 256, temperature: 0.0, top_p: 1.0, stream: false, stop: nil)
        @model = model
        @messages = messages
        @max_tokens = max_tokens
        @temperature = temperature
        @top_p = top_p
        @stream = stream
        @stop = stop
      end
    end

    # Response schema for non-streaming chat completion
    class ChatCompletionResponse
      def initialize(model:, content:, prompt_tokens:, completion_tokens:, finish_reason: "stop")
        @model = model
        @content = content
        @prompt_tokens = prompt_tokens
        @completion_tokens = completion_tokens
        @finish_reason = finish_reason
        @id = "chatcmpl-#{SecureRandom.hex(12)}"
        @created = Time.now.to_i
      end

      def to_hash
        {
          "id" => @id,
          "object" => "chat.completion",
          "created" => @created,
          "model" => @model,
          "choices" => [
            {
              "index" => 0,
              "message" => {
                "role" => "assistant",
                "content" => @content,
              },
              "finish_reason" => @finish_reason,
            }
          ],
          "usage" => {
            "prompt_tokens" => @prompt_tokens,
            "completion_tokens" => @completion_tokens,
            "total_tokens" => @prompt_tokens + @completion_tokens,
          }
        }
      end

      def to_json
        JSON.generate(to_hash)
      end
    end

    # Streaming chunk response
    class ChatCompletionChunk
      def initialize(model:, content:, finish_reason: nil)
        @model = model
        @content = content
        @finish_reason = finish_reason
        @id = "chatcmpl-#{SecureRandom.hex(12)}"
        @created = Time.now.to_i
      end

      def to_hash
        {
          "id" => @id,
          "object" => "chat.completion.chunk",
          "created" => @created,
          "model" => @model,
          "choices" => [
            {
              "index" => 0,
              "delta" => {
                "content" => @content,
              },
              "finish_reason" => @finish_reason,
            }
          ]
        }
      end

      def to_sse
        "data: #{JSON.generate(to_hash)}\n\n"
      end
    end

    # GET /v1/models response
    class ModelsListResponse
      def initialize(models:)
        @models = models
      end

      def to_hash
        {
          "object" => "list",
          "data" => @models.map { |m|
            {
              "id" => m,
              "object" => "model",
              "created" => Time.now.to_i,
              "owned_by" => "mlx-lm",
            }
          }
        }
      end

      def to_json
        JSON.generate(to_hash)
      end
    end

    module_function

    def start(model_path:, host: "127.0.0.1", port: 8080)
      require "webrick"

      model, tokenizer = LoadUtils.load(model_path)

      server = WEBrick::HTTPServer.new(Port: port, BindAddress: host)

      server.mount_proc "/v1/models" do |req, res|
        res["Content-Type"] = "application/json"
        resp = ModelsListResponse.new(models: [model_path])
        res.body = resp.to_json
      end

      server.mount_proc "/v1/chat/completions" do |req, res|
        body = JSON.parse(req.body)
        chat_req = ChatCompletionRequest.from_hash(body)

        prompt = ChatTemplate.apply(chat_req.messages)
        sampler = SampleUtils.make_sampler(temp: chat_req.temperature, top_p: chat_req.top_p)

        if chat_req.stream
          res["Content-Type"] = "text/event-stream"
          res["Cache-Control"] = "no-cache"

          res.body = Enumerator.new { |yielder|
            Generate.stream_generate(model, tokenizer, prompt,
              max_tokens: chat_req.max_tokens, sampler: sampler).each do |resp|
              chunk = ChatCompletionChunk.new(
                model: chat_req.model,
                content: resp.text,
                finish_reason: resp.finish_reason
              )
              yielder << chunk.to_sse
            end
            yielder << "data: [DONE]\n\n"
          }
        else
          text = Generate.generate(model, tokenizer, prompt,
            max_tokens: chat_req.max_tokens, sampler: sampler)

          res["Content-Type"] = "application/json"
          resp = ChatCompletionResponse.new(
            model: chat_req.model,
            content: text,
            prompt_tokens: prompt.length,
            completion_tokens: text.length,
            finish_reason: "stop"
          )
          res.body = resp.to_json
        end
      end

      trap("INT") { server.shutdown }
      server.start
    end
  end
end
