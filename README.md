[![CI](https://github.com/yaak-ai/rbyte/actions/workflows/ci.yaml/badge.svg)](https://github.com/yaak-ai/rbyte/actions/workflows/ci.yaml) [![build](https://github.com/yaak-ai/rbyte/actions/workflows/build.yaml/badge.svg)](https://github.com/yaak-ai/rbyte/actions/workflows/build.yaml)

# rbyte

Multimodal dataset library.

## Installation

```bash
uv add https://github.com/yaak-ai/rbyte/releases/latest/download/rbyte-X.Y.Z-py3-none-any.whl [--extra visualize]
```

## Usage

See [examples/config_templates/dataset](examples/config_templates/dataset).

### Visualization

1. Create a [`hydra`](https://hydra.cc) config `config.yaml` with the following structure (see [examples/config_templates/visualize.yaml](examples/config_templates/visualize.yaml)):
```yaml
dataloader: ???
logger: ???
```

2. Run using the config from step 1:
```bash
 uv run rbyte-visualize --config-name config.yaml [--config-path /path/to/config]
```

## Development

### Setup

Requirements:
- [`uv`](https://github.com/astral-sh/uv)
- [`just`](https://github.com/casey/just)

```bash
just setup
```
