set shell := ["nu", "-c"]
set script-interpreter := ["nu"]
set lazy

export PYTHONOPTIMIZE := "1"
export HATCH_BUILD_CLEAN := "1"
export HYDRA_FULL_ERROR := "1"
export TQDM_DISABLE := "1"
export PYTHONBREAKPOINT := "patdb.debug"
export PATDB_CODE_STYLE := "vim"
export BETTER_EXCEPTIONS := "1"
export LOVELY_TENSORS := "1"
export RERUN_STRICT := "1"

_default:
    @just --choose --chooser sk

sync:
    uv sync --all-extras --all-groups

setup: sync
    git submodule update --init --recursive --force --remote
    git lfs pull
    prek install --overwrite

build:
    uv build

check:
    uv format --check
    uv run --group check ruff check
    uv check

prek *ARGS: build
    prek --all-files {{ ARGS }}

[script]
generate-config:
    (
    ytt --file {{ justfile_directory() }}/config/_templates
        --output-files {{ justfile_directory() }}/config
        --output yaml
        --strict
    )

[script]
generate-test-data-yaak-mp4-frame-mappings:
    ls -f tests/data/yaak/**/*.mp4 | get name | par-each { |video|
        let output = $"($video).frames.json";
        print $"creating: ($output)";
        ffprobe -hide_banner -loglevel fatal -i $video -show_frames -show_entries frame=pts,duration,key_frame -of json | save -f $output;
    }

test *ARGS: build generate-config generate-test-data-yaak-mp4-frame-mappings
    uv run --all-extras --group test pytest --capture=no -v {{ ARGS }}

notebook FILE *ARGS: sync generate-config
    uv run --all-extras --with=jupyter,jupyterlab-vim,rerun-notebook jupyter lab {{ FILE }} {{ ARGS }}

[script]
_visualize *ARGS:
    (
    uv run
        --extra visualize
        rbyte-visualize
        --config-path {{ justfile_directory() }}/config
        --config-name visualize.yaml
        hydra/hydra_logging=disabled
        hydra/job_logging=disabled
        {{ ARGS }}
    )

[script]
visualize dataset *ARGS: generate-config
    match "{{ dataset }}" {
        "yaak" => { just generate-test-data-yaak-mp4-frame-mappings }
    }
    just _visualize dataset={{ dataset }} ++data_dir={{ justfile_directory() }}/tests/data/{{ dataset }} {{ ARGS }}

visualize-all: generate-config
    just visualize yaak
    just visualize zod batch_size=1
    just visualize mimicgen
    just visualize nuscenes
    just visualize carla_garage

[script]
benchmark-dataloader *ARGS: generate-config
    (
    uv run rbyte-benchmark-dataloader
        --config-path {{ justfile_directory() }}/config
        --config-name benchmark_dataloader.yaml
        hydra/hydra_logging=disabled
        hydra/job_logging=disabled
        {{ ARGS }}
    )

rerun *ARGS:
    uv run rerun --serve-web {{ ARGS }}
