# CLI Usage

[Back to Documentation Index](index.md)

Executable: `mlx_lm`

Commands:

- `mlx_lm generate`
- `mlx_lm chat`
- `mlx_lm server`

## Command Help

Each command uses the same `OptionParser`, so `--help` output shows the same option
set for `generate`, `chat`, and `server`.

Example:

```bash
bundle exec ruby -Ilib exe/mlx_lm generate --help
```

## Quick Examples

### Generate

```bash
mlx_lm generate --model /path/to/model --prompt "Hello" --max-tokens 64 --temp 0.7 --top-p 0.9
```

### Chat (interactive REPL)

```bash
mlx_lm chat --model /path/to/model --system-prompt "You are concise." --max-tokens 256 --temp 0.7
```

In chat mode, enter `exit`, `quit`, empty input, or EOF (`Ctrl-D`) to stop.

### Server

```bash
mlx_lm server --model /path/to/model --host 127.0.0.1 --port 8080
```

Serves:

- `GET /v1/models`
- `POST /v1/chat/completions`

## Defaults

Current defaults (`MlxLm::CLI.default_args`):

- `prompt: ""`
- `max_tokens: 256`
- `temp: 0.0`
- `top_p: 1.0`
- `seed: nil`
- `repetition_penalty: nil`
- `repetition_context_size: 20`
- `host: "127.0.0.1"`
- `port: 8080`
- `system_prompt: nil`
- `verbose: false`

## Option Behavior Matrix

`Used` means command logic currently consumes the parsed value. `Parsed only` means
the option is accepted by parser/help but not used by that command implementation.

| Option | Default | `generate` | `chat` | `server` | Notes |
| --- | --- | --- | --- | --- | --- |
| `--model` | `nil` | Used | Used | Used | Required in practice; passed to `LoadUtils.load` / `Server.start`. |
| `--prompt` | `""` | Used | Parsed only | Parsed only | Chat reads from stdin instead of `--prompt`. |
| `--max-tokens` | `256` | Used | Used | Parsed only | Server request body provides per-request `max_tokens`. |
| `--temp` | `0.0` | Used | Used | Parsed only | Server request body provides `temperature`. |
| `--top-p` | `1.0` | Used | Parsed only | Parsed only | Chat sampler currently ignores `top_p`. |
| `--seed` | `nil` | Parsed only | Parsed only | Parsed only | Parsed, never applied in current command paths. |
| `--repetition-penalty` | `nil` | Parsed only | Parsed only | Parsed only | Parsed, never wired into `make_logits_processors`. |
| `--repetition-context-size` | `20` | Parsed only | Parsed only | Parsed only | Parsed, never wired into `make_logits_processors`. |
| `--host` | `127.0.0.1` | Parsed only | Parsed only | Used | Only used by `server`. |
| `--port` | `8080` | Parsed only | Parsed only | Used | Only used by `server`. |
| `--system-prompt` | `nil` | Parsed only | Used | Parsed only | Prepended as a system message for chat. |
| `--verbose` | `false` | Used | Parsed only | Parsed only | Only affects `generate` output mode/stats. |

## Caveats

- Help text says `--model` can be a model path or HuggingFace ID, but current runtime
  loading expects a local model directory (`config.json`, tokenizer files, safetensors).
- Repetition and seed flags are currently parser-level only and do not change runtime generation behavior.
- `chat` always builds prompts from chat turns via `MlxLm::ChatTemplate.apply`; `--prompt` is ignored.
- `server` serves OpenAI-style endpoints but does not consume CLI generation flags (temperature/top-p/etc.).

## Related Docs

- [Documentation Index](index.md)
- [Installation](installation.md)
- [Ruby APIs](ruby-apis.md)
- [Models](models.md)
