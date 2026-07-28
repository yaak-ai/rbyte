from pathlib import Path

import numpy as np
import polars as pl
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from optree import PyTree

from rbyte.config import PipelineHydraConfig

DATA_DIR = Path(__file__).resolve().parent / "data"
CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def _run_pipeline(name: str, output_names: set[str]) -> PyTree[pl.DataFrame]:
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR.as_posix()):
        cfg = compose(
            "dataset", overrides=[f"dataset={name}", f"+data_dir={DATA_DIR / name}"]
        )

    samples = PipelineHydraConfig.model_validate(
        OmegaConf.to_container(cfg.dataset.samples, resolve=True)
    )
    pipeline = samples.pipeline.instantiate().subpipeline(output_names=output_names)
    results = pipeline.map(
        inputs={
            name: value
            for name, value in samples.inputs.items()
            if name in pipeline.topological_generations.root_args
        },
        parallel=False,
    )

    outputs = {}
    for output_name in output_names:
        output = results[output_name].output
        outputs[output_name] = (
            output.item() if isinstance(output, np.ndarray) else output
        )

    return outputs  # ty:ignore[invalid-return-type]


def test_samples_mimicgen() -> None:
    input_ids = ("coffee/data/demo_0", "coffee/data/demo_1")
    samples: pl.DataFrame = _run_pipeline("mimicgen", {k := "samples_aggregated"})[k]  # ty:ignore[invalid-assignment]

    assert samples.shape == (419, 3)
    assert samples.schema == pl.Schema({
        "_idx_": pl.UInt32(),
        "obs/robot0_eef_pos": pl.Array(pl.Float64(), 3),
        "input_id": pl.Enum(input_ids),
    })


def test_samples_nuscenes() -> None:
    reference_topic = "/CAM_FRONT/image_rect_compressed"
    camera_topics = (
        reference_topic,
        "/CAM_FRONT_LEFT/image_rect_compressed",
        "/CAM_FRONT_RIGHT/image_rect_compressed",
    )
    outputs = _run_pipeline("nuscenes", {"data", "indexed", "aligned"})
    raw: dict[str, pl.DataFrame] = outputs["data"]  # ty:ignore[invalid-assignment]
    indexed: dict[str, pl.DataFrame] = outputs["indexed"]  # ty:ignore[invalid-assignment]
    aligned: pl.DataFrame = outputs["aligned"]  # ty:ignore[invalid-assignment]

    assert {topic: len(df) for topic, df in raw.items()} == {
        reference_topic: 12,
        "/CAM_FRONT_RIGHT/image_rect_compressed": 12,
        "/CAM_FRONT_LEFT/image_rect_compressed": 11,
        "/odom": 47,
    }
    assert len(aligned) == 12  # ruff:ignore[magic-value-comparison]
    assert aligned[f"{reference_topic}/_idx_"].to_list() == list(range(12))

    reference_time = aligned[f"{reference_topic}/log_time"]
    for topic in camera_topics[1:]:
        matched_time = indexed[topic]["log_time"].gather(aligned[f"{topic}/_idx_"])
        assert (
            (matched_time - reference_time).dt.total_nanoseconds().abs().max()
            <= 40_000_000  # ruff:ignore[magic-value-comparison]  # ty:ignore[unsupported-operator]
        )

    assert aligned.null_count().row(0) == (0, 0, 0, 0, 2)
    assert aligned["/odom/vel.x"].is_null().to_list() == [True, True, *([False] * 10)]


def test_samples_zod() -> None:
    camera_time = "camera_front_blur_meta/timestamp"
    lidar_time = "lidar_velodyne_meta/timestamp"
    controls_time = "vehicle_data/ego_vehicle_controls/timestamp/nanoseconds/value"
    outputs = _run_pipeline(
        "zod", {"camera_front_blur_meta", "lidar_velodyne_meta", "aligned"}
    )
    camera: pl.DataFrame = outputs["camera_front_blur_meta"]  # ty:ignore[invalid-assignment]
    lidar: pl.DataFrame = outputs["lidar_velodyne_meta"]  # ty:ignore[invalid-assignment]
    aligned: pl.DataFrame = outputs["aligned"]  # ty:ignore[invalid-assignment]

    assert len(camera) == 10  # ruff:ignore[magic-value-comparison]
    assert len(lidar) == 9  # ruff:ignore[magic-value-comparison]
    assert aligned.shape == (10, 6)
    assert aligned.schema == pl.Schema({
        camera_time: pl.Datetime("ns"),
        lidar_time: pl.Datetime("ns"),
        controls_time: pl.Datetime("ns"),
        "vehicle_data/ego_vehicle_controls/acceleration_pedal/ratio/unitless/value": pl.Float32(),  # ruff:ignore[line-too-long]
        "vehicle_data/ego_vehicle_controls/steering_wheel_angle/angle/radians/value": pl.Float32(),  # ruff:ignore[line-too-long]
        "vehicle_data/satellite/speed/meters_per_second/value": pl.Float32(),
    })
    for column in (lidar_time, controls_time):
        assert (
            (
                (aligned[column] - aligned[camera_time])
                .dt.total_nanoseconds()
                .abs()
                .max()
            )
            <= 100_000_000  # ruff:ignore[magic-value-comparison]  # ty:ignore[unsupported-operator]
        )
    assert aligned.null_count().row(0) == (0,) * len(aligned.columns)


def test_samples_yaak() -> None:
    outputs = _run_pipeline("yaak", {"meta_raw", "waypoints"})
    metadata: dict[str, pl.DataFrame] = outputs["meta_raw"]  # ty:ignore[invalid-assignment]
    waypoints: pl.DataFrame = outputs["waypoints"]  # ty:ignore[invalid-assignment]

    assert {name: len(df) for name, df in metadata.items()} == {
        "VehicleMotion": 1024,
        "Gnss": 218,
        "ImageMetadata.cam_front_center": 621,
        "ImageMetadata.cam_front_left": 621,
        "ImageMetadata.cam_front_right": 621,
        "ImageMetadata.cam_rear": 620,
        "ImageMetadata.cam_right_forward": 619,
        "ImageMetadata.cam_left_backward": 619,
        "ImageMetadata.cam_right_backward": 619,
        "ImageMetadata.cam_left_forward": 619,
    }
    assert len(waypoints) == 218  # ruff:ignore[magic-value-comparison]

    positions = waypoints["waypoints/position"].to_numpy()
    assert positions.shape == (218, 10, 2)
    assert np.isfinite(positions).all()

    segment_lengths = np.linalg.norm(np.diff(positions, axis=1), axis=2)
    assert (segment_lengths > 0).all()
    assert (segment_lengths <= 10 + 1e-6).all()
    assert np.median(segment_lengths) == pytest.approx(10, abs=1e-6)
