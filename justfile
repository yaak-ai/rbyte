export PYTHONOPTIMIZE := "1"
export HATCH_BUILD_CLEAN := "1"
export HYDRA_FULL_ERROR := "1"
export TQDM_DISABLE := "1"

_default:
    @just --list --unsorted

sync:
    uv sync --all-extras --all-groups

install-tools:
    uv tool install --force --upgrade ruff
    uv tool install --force --upgrade basedpyright
    uv tool install --force --upgrade pre-commit --with pre-commit-uv

install-duckdb-extensions:
    uv run python -c "import duckdb; duckdb.connect().install_extension('spatial')"

setup: sync install-tools install-duckdb-extensions
    git submodule update --init --recursive --force --remote
    git lfs pull
    uvx pre-commit install --install-hooks

build:
    uv build

format *ARGS:
    uvx ruff format {{ ARGS }}

lint *ARGS:
    uvx ruff check {{ ARGS }}

typecheck *ARGS:
    uvx basedpyright {{ ARGS }}

pre-commit *ARGS: build
    uvx pre-commit run --all-files --color=always {{ ARGS }}

generate-config:
    ytt --ignore-unknown-comments \
        --file {{ justfile_directory() }}/config/_templates \
        --output-files {{ justfile_directory() }}/config \
        --output yaml \
        --strict

test *ARGS: build generate-config
    uv run --all-extras pytest --capture=no {{ ARGS }}

notebook FILE *ARGS: sync generate-config
    uv run --all-extras --with=jupyter,jupyterlab-vim,rerun-notebook jupyter lab {{ FILE }} {{ ARGS }}

[group('scripts')]
visualize *ARGS: generate-config
    uv run rbyte-visualize \
        --config-path {{ justfile_directory() }}/config \
        --config-name visualize.yaml \
        hydra/hydra_logging=disabled \
        hydra/job_logging=disabled \
        {{ ARGS }}

[group('visualize')]
visualize-yaak *ARGS:
    just visualize dataset=yaak logger=console ++data_dir={{ justfile_directory() }}/tests/data/yaak {{ ARGS }}

[group('visualize')]
visualize-carla-garage *ARGS:
    just visualize dataset=carla_garage logger=rerun/carla_garage ++data_dir={{ justfile_directory() }}/tests/data/carla_garage {{ ARGS }}

[group('visualize')]
visualize-zod *ARGS:
    just visualize dataloader=unbatched dataset=zod logger=rerun/zod ++data_dir={{ justfile_directory() }}/tests/data/zod {{ ARGS }}

[group('visualize')]
visualize-mimicgen *ARGS:
    just visualize dataset=mimicgen logger=rerun/mimicgen ++data_dir={{ justfile_directory() }}/tests/data/mimicgen {{ ARGS }}

[group('visualize')]
visualize-nuscenes *ARGS:
    just visualize dataset=nuscenes logger=rerun/nuscenes ++data_dir={{ justfile_directory() }}/tests/data/nuscenes {{ ARGS }}

[group('visualize')]
visualize-all: visualize-yaak visualize-zod visualize-mimicgen visualize-nuscenes

# rerun server and viewer
rerun bind="0.0.0.0" port="9876" web-viewer-port="9090":
    RUST_LOG=debug uv run rerun \
    	--bind {{ bind }} \
    	--port {{ port }} \
    	--web-viewer \
    	--web-viewer-port {{ web-viewer-port }} \
    	--memory-limit 95% \
    	--server-memory-limit 95% \
    	--expect-data-soon \
