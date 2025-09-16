export PYTHONOPTIMIZE := "1"
export HATCH_BUILD_CLEAN := "1"
export HYDRA_FULL_ERROR := "1"
export TQDM_DISABLE := "1"

_default:
    @just --choose --chooser sk

sync:
    uv sync --all-extras --all-groups

install-duckdb-extensions:
    uv run python -c "import duckdb; duckdb.connect().install_extension('spatial')"

setup: sync install-duckdb-extensions
    git submodule update --init --recursive --force --remote
    git lfs pull
    uvx --with=pre-commit-uv pre-commit install --install-hooks

build:
    uv build

format *ARGS:
    uvx ruff format {{ ARGS }}

lint *ARGS:
    uvx ruff check {{ ARGS }}

typecheck *ARGS:
    uvx ty@latest check {{ ARGS }}

pre-commit *ARGS: build
    uvx --with=pre-commit-uv pre-commit run --all-files --color=always {{ ARGS }}

generate-config:
    ytt --file {{ justfile_directory() }}/config/_templates \
        --output-files {{ justfile_directory() }}/config \
        --output yaml \
        --strict

test *ARGS: build generate-config
    uv run --all-extras pytest --capture=no -v {{ ARGS }}

notebook FILE *ARGS: sync generate-config
    uv run --all-extras --with=jupyter,jupyterlab-vim,rerun-notebook jupyter lab {{ FILE }} {{ ARGS }}

[group('scripts')]
_visualize *ARGS:
    uv run rbyte-visualize \
        --config-path {{ justfile_directory() }}/config \
        --config-name visualize.yaml \
        hydra/hydra_logging=disabled \
        hydra/job_logging=disabled \
        {{ ARGS }}

[group('scripts')]
visualize dataset *ARGS: generate-config
    just _visualize dataset={{ dataset }} ++data_dir={{ justfile_directory() }}/tests/data/{{ dataset }} {{ ARGS }}

[group('scripts')]
visualize-all: generate-config
    just visualize yaak
    just visualize zod batch_size=1
    just visualize mimicgen
    just visualize nuscenes
    just visualize carla_garage

benchmark-dataloader *ARGS: generate-config
    uv run rbyte-benchmark-dataloader \
        --config-path {{ justfile_directory() }}/config \
        --config-name benchmark_dataloader.yaml \
        hydra/hydra_logging=disabled \
        hydra/job_logging=disabled \
        {{ ARGS }}

rerun *ARGS:
    uv run rerun --serve-web {{ ARGS }}
