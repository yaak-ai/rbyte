set script-interpreter := ["nu"]
set default-script
set default-list
set unstable
set lists

export HYDRA_FULL_ERROR := "1"
export TQDM_DISABLE := "1"
export PYTHONBREAKPOINT := "patdb.debug"
export PATDB_CODE_STYLE := "vim"
export BETTER_EXCEPTIONS := "1"
export LOVELY_TENSORS := "1"
export RERUN_STRICT := "1"

sync:
    uv sync --all-extras

setup: sync
    git lfs pull
    prek install --overwrite

build:
    uv build

check:
    uv format --check
    uv run --group check ruff check
    uv check

prek *ARGS:
    prek --all-files {{ quote(ARGS) }}

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

test *ARGS: generate-config generate-test-data-yaak-mp4-frame-mappings
    uv run --all-extras pytest --capture=no {{ quote(ARGS) }}

notebook FILE *ARGS: generate-config
    uv run --all-extras jupyter lab {{ quote(FILE) }} {{ quote(ARGS) }}

[script]
_visualize *ARGS: generate-config
    (
    uv run
        --extra visualize
        rbyte-visualize
        --config-path {{ justfile_directory() }}/config
        --config-name visualize.yaml
        hydra/hydra_logging=disabled
        hydra/job_logging=disabled
        {{ quote(ARGS) }}
    )

[script]
visualize dataset *ARGS:
    match {{ quote(dataset) }} {
        "yaak" => { just generate-test-data-yaak-mp4-frame-mappings }
    }
    just _visualize {{ quote("dataset=" + dataset) }} {{ quote("++data_dir=" + justfile_directory() + "/tests/data/" + dataset) }} {{ quote(ARGS) }}

visualize-all: generate-config
    just visualize yaak
    just visualize zod batch_size=1
    just visualize mimicgen
    just visualize nuscenes

[script]
benchmark-dataloader *ARGS: generate-config
    (
    uv run rbyte-benchmark-dataloader
        --config-path {{ justfile_directory() }}/config
        --config-name benchmark_dataloader.yaml
        hydra/hydra_logging=disabled
        hydra/job_logging=disabled
        {{ quote(ARGS) }}
    )
