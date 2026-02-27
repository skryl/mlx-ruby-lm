# Installation

[Back to Documentation Index](index.md)

## Requirements

- Ruby `>= 3.4`
- Runtime gem dependencies:
  - `mlx` `>= 0.30.7.5`, `< 1.0`
  - `safetensors` `~> 0.2`
  - `tokenizers` `~> 0.6`

`mlx-ruby-lm` depends on the `mlx` gem runtime and follows its platform/runtime constraints.

## Install from RubyGems

```bash
gem install mlx-ruby-lm
```

## Install with Bundler

```bash
bundle add mlx-ruby-lm
bundle install
```

## Install from Source (Local Development)

From the repository root:

```bash
bundle install
bundle exec ruby -Ilib exe/mlx_lm generate --help
```

## Verify Installation

```bash
bundle exec ruby -Ilib -e 'require "mlx"; require "mlx_lm"; puts MlxLm::VERSION'
```

## Next Steps

- [Documentation Index](index.md)
- [CLI Usage](cli.md)
- [Ruby APIs](ruby-apis.md)
- [Models](models.md)
