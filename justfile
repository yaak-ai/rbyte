export HYDRA_FULL_ERROR := "1"
export PYTHONOPTIMIZE := "1"

_default:
    @just --list --unsorted

setup:
    git submodule update --init --recursive --remote
    uv sync --all-extras --dev
    for tool in basedpyright ruff pre-commit; do uv tool install --force --upgrade $tool;  done
    uvx pre-commit install --install-hooks

build:
    uvx --from build pyproject-build --installer uv --wheel

pre-commit:
    uvx pre-commit validate-config
    uvx pre-commit install --install-hooks
    uvx pre-commit run --all-files --color=always

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
        {{ ARGS }}

[group('scripts')]
build-table *ARGS: generate-example-config
    uv run rbyte-build-table \
        --config-path {{ justfile_directory() }}/examples/config \
        --config-name table.yaml \
        {{ ARGS }}

# rerun server and viewer
rerun bind="0.0.0.0" port="9876" ws-server-port="9877" web-viewer-port="9090":
    uv run rerun \
    	--bind {{ bind }} \
    	--port {{ port }} \
    	--ws-server-port {{ ws-server-port }} \
    	--web-viewer \
    	--web-viewer-port {{ web-viewer-port }} \
    	--memory-limit 95% \
    	--server-memory-limit 95% \
    	--expect-data-soon \
