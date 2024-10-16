export PYTHONOPTIMIZE := "1"
export HATCH_BUILD_CLEAN := "1"

_default:
    @just --list --unsorted

sync:
    uv sync --all-extras --dev

install-tools:
    for tool in basedpyright ruff pre-commit; do uv tool install --force --upgrade $tool;  done

setup: sync install-tools
    git submodule update --init --recursive --remote
    uvx pre-commit install --install-hooks

clean:
    uvx --from hatch hatch clean

build:
    uv build

build-protos:
    uvx --from hatch hatch build --clean --hooks-only --target sdist

pre-commit *ARGS: build-protos
    uvx pre-commit run --all-files --color=always {{ ARGS }}

generate-example-config:
    ytt --ignore-unknown-comments \
        --file {{ justfile_directory() }}/examples/config_templates \
        --output-files examples/config \
        --output yaml \
        --strict

[group('scripts')]
visualize *ARGS: generate-example-config
    uv run rbyte-visualize \
        --config-path {{ justfile_directory() }}/examples/config \
        --config-name visualize.yaml \
        hydra/hydra_logging=disabled \
        hydra/job_logging=disabled \
        {{ ARGS }}

[group('scripts')]
build-table *ARGS: generate-example-config
    uv run rbyte-build-table \
        --config-path {{ justfile_directory() }}/examples/config \
        --config-name build_table.yaml \
        hydra/hydra_logging=disabled \
        hydra/job_logging=disabled \
        {{ ARGS }}

[group('scripts')]
read-frames *ARGS: generate-example-config
    uv run rbyte-read-frames \
        --config-path {{ justfile_directory() }}/examples/config \
        --config-name read_frames.yaml \
        hydra/hydra_logging=disabled \
        hydra/job_logging=disabled \
        {{ ARGS }}

# rerun server and viewer
rerun bind="0.0.0.0" port="9876" ws-server-port="9877" web-viewer-port="9090":
    RUST_LOG=debug uv run rerun \
    	--bind {{ bind }} \
    	--port {{ port }} \
    	--ws-server-port {{ ws-server-port }} \
    	--web-viewer \
    	--web-viewer-port {{ web-viewer-port }} \
    	--memory-limit 95% \
    	--server-memory-limit 95% \
    	--expect-data-soon \
