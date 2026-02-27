# mlx-ruby-lm

[![Tests](https://github.com/skryl/mlx-ruby-lm/actions/workflows/ci.yml/badge.svg)](https://github.com/skryl/mlx-ruby-lm/actions/workflows/ci.yml) [![Gem Version](https://badge.fury.io/rb/mlx-ruby-lm.svg)](https://rubygems.org/gems/mlx-ruby-lm)

Ruby LLM inference toolkit built on the `mlx` gem.

## Index

- [Documentation Index](docs/index.md)
- [Installation](docs/installation.md)
- [CLI Usage](docs/cli.md)
- [Ruby APIs](docs/ruby-apis.md)
- [Models](docs/models.md)

For full reference pages and deep dives, start at [docs/index.md](docs/index.md).

## Installation

```bash
gem install mlx-ruby-lm
```

Or add it to a project:

```bash
bundle add mlx-ruby-lm
```

See [docs/installation.md](docs/installation.md) for requirements and source installs.

## CLI Usage

Executable: `mlx_lm`

Commands:

- `mlx_lm generate`
- `mlx_lm chat`
- `mlx_lm server`

Quick examples:

```bash
mlx_lm generate --model /path/to/model --prompt "Hello"
mlx_lm chat --model /path/to/model --system-prompt "You are concise."
mlx_lm server --model /path/to/model --host 127.0.0.1 --port 8080
```

See [docs/cli.md](docs/cli.md) for options, defaults, and current parser/behavior caveats.

## High-Level Ruby API Usage

```ruby
require "mlx"
require "mlx_lm"

model, tokenizer = MlxLm::LoadUtils.load("/path/to/model")
text = MlxLm::Generate.generate(model, tokenizer, "Hello", max_tokens: 64)
puts text
```

Streaming:

```ruby
MlxLm::Generate.stream_generate(model, tokenizer, "Hello", max_tokens: 64).each do |resp|
  print resp.text
end
puts
```

See [docs/ruby-apis.md](docs/ruby-apis.md) for the full API inventory.

## High-Level Model Usage

`LoadUtils.load` expects a local model directory with files such as `config.json`,
`tokenizer.json`, and `model*.safetensors`.

To inspect supported model keys at runtime:

```ruby
require "mlx_lm"
puts MlxLm::Models::REGISTRY.keys.sort
```

See [docs/models.md](docs/models.md) for full registry keys and remapping behavior.
