# Server

[Back to Documentation Index](index.md)

This page documents the current HTTP server in `lib/mlx_lm/server.rb`.

## Startup API

```ruby
MlxLm::Server.start(model_path:, host: "127.0.0.1", port: 8080)
```

`start` behavior:

1. Loads model/tokenizer once at boot via `MlxLm::LoadUtils.load(model_path)`.
2. Starts a `WEBrick::HTTPServer`.
3. Mounts:
   - `GET /v1/models`
   - `POST /v1/chat/completions`
4. Traps `INT` for graceful `server.shutdown`.

## Endpoint: `GET /v1/models`

Response content type: `application/json`

Response schema (`MlxLm::Server::ModelsListResponse#to_hash`):

```json
{
  "object": "list",
  "data": [
    {
      "id": "/path/or/id/passed-as-model_path",
      "object": "model",
      "created": 1700000000,
      "owned_by": "mlx-lm"
    }
  ]
}
```

Notes:

- Response always contains a single model entry built from startup `model_path`.
- `created` is generated at response time.

## Endpoint: `POST /v1/chat/completions`

### Request Schema

Parsed by `MlxLm::Server::ChatCompletionRequest.from_hash`.

Accepted fields:

- `model` (string, optional in parser but echoed in responses)
- `messages` (array, default `[]`)
- `max_tokens` (integer, default `256`)
- `temperature` (float, default `0.0`)
- `top_p` (float, default `1.0`)
- `stream` (boolean, default `false`)
- `stop` (accepted and stored, currently unused)

Prompt building and sampling:

- Prompt is always created by `MlxLm::ChatTemplate.apply(messages)`.
- Sampler is built with `MlxLm::SampleUtils.make_sampler(temp: temperature, top_p: top_p)`.

### Non-Streaming Response (`stream: false`)

Response content type: `application/json`

Schema (`MlxLm::Server::ChatCompletionResponse#to_hash`):

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "request-model-field",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "generated text"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 123,
    "completion_tokens": 45,
    "total_tokens": 168
  }
}
```

### Streaming SSE Response (`stream: true`)

Headers:

- `Content-Type: text/event-stream`
- `Cache-Control: no-cache`

Body format:

- Each chunk: `data: {json}\n\n`
- Final sentinel: `data: [DONE]\n\n`

Chunk schema (`MlxLm::Server::ChatCompletionChunk#to_hash`):

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion.chunk",
  "created": 1700000000,
  "model": "request-model-field",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "next text segment"
      },
      "finish_reason": null
    }
  ]
}
```

`finish_reason` may be `"stop"` on the final generated chunk before `[DONE]`.

## Quick Example

```bash
curl -s http://127.0.0.1:8080/v1/models | jq .
```

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model":"local-model",
    "messages":[{"role":"user","content":"Say hello"}],
    "max_tokens":64,
    "temperature":0.7,
    "top_p":0.9,
    "stream":false
  }' | jq .
```

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model":"local-model",
    "messages":[{"role":"user","content":"Stream a short response"}],
    "stream":true
  }'
```

## Caveats

- Only `GET /v1/models` and `POST /v1/chat/completions` are implemented. Endpoints like `/v1/completions` are not present.
- `model` in request is not used for model routing/validation; one model is loaded at startup and used for all requests.
- `stop` is parsed in request schema but is not applied during generation.
- Request parsing and validation are minimal; invalid JSON or malformed payloads are not handled with structured API errors.
- Usage counters in non-streaming responses are not tokenizer token counts:
  - `prompt_tokens` is prompt string length (`prompt.length`)
  - `completion_tokens` is generated string length (`text.length`)
- Prompt formatting always uses `MlxLm::ChatTemplate.apply` default path; tokenizer-provided chat templates are not applied here.
- Streaming chunk IDs/timestamps are generated per chunk object; chunk identity is not stable across a completion.

[Back to Documentation Index](index.md)
