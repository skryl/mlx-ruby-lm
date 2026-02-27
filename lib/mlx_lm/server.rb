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

    class StopCondition
      attr_reader :stop_strings, :max_tokens

      def initialize(stop: nil, max_tokens: nil)
        @stop_strings = Array(stop).compact.map(&:to_s)
        @max_tokens = max_tokens
      end

      def met?(text:, generated_tokens:)
        return true if max_tokens && generated_tokens.to_i >= max_tokens.to_i
        return false if stop_strings.empty?

        stop_strings.any? { |stop| text.include?(stop) }
      end
    end

    CacheEntry = Struct.new(
      :key,
      :value,
      :created_at,
      :last_access_at,
      keyword_init: true
    )

    SearchResult = Struct.new(
      :entry,
      :score,
      keyword_init: true
    )

    class LRUPromptCache
      def initialize(max_entries: 128)
        @max_entries = max_entries
        @store = {}
        @lru = []
      end

      def get(key)
        entry = @store[key]
        return nil unless entry

        entry.last_access_at = Time.now
        touch(key)
        entry.value
      end

      def put(key, value)
        now = Time.now
        entry = CacheEntry.new(key: key, value: value, created_at: now, last_access_at: now)
        @store[key] = entry
        touch(key)
        evict! while @store.length > @max_entries
        entry
      end

      def search(prefix)
        key = @store.keys.find { |k| k.start_with?(prefix.to_s) }
        return nil unless key

        entry = @store[key]
        SearchResult.new(entry: entry, score: entry.value.to_s.length)
      end

      def size
        @store.size
      end

      private

      def touch(key)
        @lru.delete(key)
        @lru << key
      end

      def evict!
        oldest = @lru.shift
        @store.delete(oldest)
      end
    end

    class ModelDescription
      attr_reader :id, :object, :created, :owned_by

      def initialize(id:, object: "model", created: Time.now.to_i, owned_by: "mlx-lm")
        @id = id
        @object = object
        @created = created
        @owned_by = owned_by
      end

      def to_hash
        {
          "id" => id,
          "object" => object,
          "created" => created,
          "owned_by" => owned_by,
        }
      end
    end

    SamplingArguments = Struct.new(
      :temperature,
      :top_p,
      :top_k,
      :min_p,
      :seed,
      :repetition_penalty,
      :repetition_context_size,
      keyword_init: true
    )

    LogitsProcessorArguments = Struct.new(
      :repetition_penalty,
      :repetition_context_size,
      keyword_init: true
    )

    GenerationArguments = Struct.new(
      :max_tokens,
      :sampler,
      :logits_processors,
      :stop,
      keyword_init: true
    )

    class CompletionRequest < ChatCompletionRequest
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
    end

    GenerationContext = Struct.new(
      :model,
      :tokenizer,
      :request,
      :prompt,
      :sampling,
      :generation,
      :prompt_cache,
      :time_budget,
      keyword_init: true
    )

    class Response < ChatCompletionResponse
    end

    class TimeBudget
      def initialize(seconds: nil, deadline: nil)
        @deadline = deadline || (seconds ? Time.now + seconds : nil)
      end

      def expired?
        return false if @deadline.nil?

        Time.now >= @deadline
      end

      def remaining_seconds
        return nil if @deadline.nil?

        [@deadline - Time.now, 0.0].max
      end
    end

    class ModelProvider
      def initialize(loader: nil)
        @loader = loader || ->(path) { LoadUtils.load(path) }
        @cache = {}
      end

      def load(model_path)
        @cache[model_path] ||= @loader.call(model_path)
      end
    end

    class ResponseGenerator
      def initialize(model_provider: ModelProvider.new)
        @model_provider = model_provider
      end

      def generate(request, model_path:)
        model, tokenizer = @model_provider.load(model_path)
        prompt = ChatTemplate.apply(request.messages)
        sampler = SampleUtils.make_sampler(temp: request.temperature, top_p: request.top_p)
        text = Generate.generate(model, tokenizer, prompt, max_tokens: request.max_tokens, sampler: sampler)
        Response.new(
          model: request.model,
          content: text,
          prompt_tokens: prompt.length,
          completion_tokens: text.length,
          finish_reason: "stop"
        )
      end

      def stream(request, model_path:)
        model, tokenizer = @model_provider.load(model_path)
        prompt = ChatTemplate.apply(request.messages)
        sampler = SampleUtils.make_sampler(temp: request.temperature, top_p: request.top_p)
        Generate.stream_generate(model, tokenizer, prompt, max_tokens: request.max_tokens, sampler: sampler)
      end
    end

    class APIHandler
      def initialize(response_generator: ResponseGenerator.new)
        @response_generator = response_generator
      end

      def parse_request(payload)
        CompletionRequest.from_hash(payload)
      end

      def handle_chat_completion(payload, model_path:)
        request = parse_request(payload)
        if request.stream
          stream = @response_generator.stream(request, model_path: model_path)
          Enumerator.new do |yielder|
            stream.each do |resp|
              chunk = ChatCompletionChunk.new(
                model: request.model,
                content: resp.text,
                finish_reason: resp.finish_reason
              )
              yielder << chunk.to_sse
            end
            yielder << "data: [DONE]\n\n"
          end
        else
          @response_generator.generate(request, model_path: model_path).to_hash
        end
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
