from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from pipefunc import PipeFunc, Pipeline
from pipefunc.helpers import collect_kwargs
from pytest_lazy_fixtures import lf
from structlog import get_logger
from torch import Tensor

from rbyte import Dataset
from rbyte.config.base import HydraConfig
from rbyte.dataset import PipelineInstanceConfig, SourceConfig, SourcesConfig
from rbyte.io import (
    DataFrameAligner,
    DataFrameDuckDbQuery,
    DataFrameGroupByDynamic,
    DuckDbDataFrameBuilder,
    McapDataFrameBuilder,
    ProtobufMcapDecoderFactory,
    TorchCodecFrameSource,
    VideoDataFrameBuilder,
    WaypointBuilder,
    Wgs84ToUtm,
    YaakMetadataDataFrameBuilder,
)
from rbyte.io.dataframe.aligner import (
    AlignConfig,
    AsofColumnAlignConfig,
    InterpColumnAlignConfig,
)

logger = get_logger(__name__)

CONFIG_PATH = "../config"
DATA_DIR = Path(__file__).resolve().parent / "data"


@pytest.fixture
def yaak_pydantic() -> Dataset:
    data_dir = DATA_DIR / "yaak"
    drive_ids = ["Niro098-HQ/2024-06-18--13-39-54"]
    cameras = ["cam_front_left", "cam_left_backward", "cam_right_backward"]

    samples = PipelineInstanceConfig(
        inputs={
            input_id: {
                "meta_path": data_dir / input_id / "metadata.log",
                "mcap_path": data_dir / input_id / "ai.mcap",
                "waypoints_path": data_dir / input_id / "waypoints.json",
            }
            | {
                f"{camera}_path": data_dir / input_id / f"{camera}.pii.mp4"
                for camera in cameras
            }
            for input_id in drive_ids
        },
        parallel=False,  # pyright: ignore[reportCallIssue]
        storage="dict",  # pyright: ignore[reportCallIssue]
        pipeline=Pipeline(
            validate_type_annotations=False,
            functions=[
                PipeFunc(
                    renames={"path": "meta_path"},
                    output_name="meta",
                    mapspec="meta_path[i] -> meta[i]",
                    func=YaakMetadataDataFrameBuilder(
                        fields={
                            "rbyte.io.yaak.proto.sensor_pb2.ImageMetadata": {  # pyright: ignore[reportArgumentType]
                                "time_stamp": pl.Datetime(),
                                "frame_idx": pl.Int32(),
                                "camera_name": pl.Enum((
                                    "cam_front_center",
                                    "cam_front_left",
                                    "cam_front_right",
                                    "cam_left_forward",
                                    "cam_right_forward",
                                    "cam_left_backward",
                                    "cam_right_backward",
                                    "cam_rear",
                                )),
                            },
                            "rbyte.io.yaak.proto.can_pb2.VehicleMotion": {
                                "time_stamp": pl.Datetime(),
                                "speed": pl.Float32(),
                            },
                            "rbyte.io.yaak.proto.sensor_pb2.Gnss": {
                                "time_stamp": pl.Datetime(),
                                "latitude": pl.Float32(),
                                "longitude": pl.Float32(),
                            },
                        }
                    ),
                ),
                *(
                    PipeFunc(
                        renames={"path": f"{camera}_path"},
                        output_name=f"{camera}_meta",
                        mapspec=f"{camera}_path[i] -> {camera}_meta[i]",
                        func=VideoDataFrameBuilder(fields={"frame_idx": pl.Int32()}),
                    )
                    for camera in cameras
                ),
                PipeFunc(
                    renames={"path": "mcap_path"},
                    output_name="mcap",
                    mapspec="mcap_path[i] -> mcap[i]",
                    func=McapDataFrameBuilder(
                        decoder_factories=[ProtobufMcapDecoderFactory],
                        fields={
                            "/ai/safety_score": {
                                "clip.end_timestamp": pl.Datetime(),
                                "score": pl.Float32(),
                            }
                        },
                    ),
                ),
                PipeFunc(
                    renames={"path": "waypoints_path"},
                    output_name="waypoints_raw",
                    mapspec="waypoints_path[i] -> waypoints_raw[i]",
                    func=DuckDbDataFrameBuilder(udfs=[Wgs84ToUtm]),
                    bound={
                        "query": """
LOAD spatial;
SET TimeZone = 'UTC';
SELECT TO_TIMESTAMP(timestamp)::TIMESTAMP as timestamp,
   heading,
   ST_Wgs84ToUtm(ST_AsWKB(geom)) AS geometry
FROM ST_Read('{path}')
"""
                    },
                ),
                PipeFunc(
                    renames={"input": "waypoints_raw"},
                    output_name="waypoints",
                    mapspec="waypoints_raw[i] -> waypoints[i]",
                    func=WaypointBuilder(
                        length=10,
                        columns=WaypointBuilder.Columns(
                            points="geometry", output="waypoints"
                        ),
                    ),
                ),
                PipeFunc(
                    output_name="data",
                    mapspec="meta[i], mcap[i], waypoints[i] -> data[i]",
                    func=collect_kwargs(parameters=("meta", "mcap", "waypoints")),
                ),
                PipeFunc(
                    renames={"input": "data"},
                    output_name="aligned",
                    mapspec="data[i] -> aligned[i]",
                    func=DataFrameAligner(
                        separator="/",
                        fields=OrderedDict({
                            "meta": OrderedDict({
                                **{
                                    f"ImageMetadata.{camera}": AlignConfig(
                                        key="time_stamp",
                                        columns=OrderedDict(
                                            {}
                                            if i == 0
                                            else {
                                                "frame_idx": AsofColumnAlignConfig(
                                                    strategy="nearest", tolerance="20ms"
                                                )
                                            }
                                        ),
                                    )
                                    for i, camera in enumerate(cameras)
                                },
                                "VehicleMotion": AlignConfig(
                                    key="time_stamp",
                                    columns=OrderedDict(
                                        speed=InterpColumnAlignConfig()
                                    ),
                                ),
                                "Gnss": AlignConfig(
                                    key="time_stamp",
                                    columns=OrderedDict(
                                        latitude=AsofColumnAlignConfig(
                                            strategy="nearest", tolerance="500ms"
                                        ),
                                        longitude=AsofColumnAlignConfig(
                                            strategy="nearest", tolerance="500ms"
                                        ),
                                    ),
                                ),
                            }),
                            "mcap": OrderedDict({
                                "/ai/safety_score": AlignConfig(
                                    key="clip.end_timestamp",
                                    columns=OrderedDict({
                                        "clip.end_timestamp": AsofColumnAlignConfig(
                                            strategy="nearest", tolerance="500ms"
                                        ),
                                        "score": AsofColumnAlignConfig(
                                            strategy="nearest", tolerance="500ms"
                                        ),
                                    }),
                                )
                            }),
                            "waypoints": AlignConfig(
                                key="timestamp",
                                columns=OrderedDict({
                                    "heading": AsofColumnAlignConfig(
                                        strategy="nearest"
                                    ),
                                    "waypoints": AsofColumnAlignConfig(
                                        strategy="nearest"
                                    ),
                                }),
                            ),
                        }),
                    ),
                ),
                PipeFunc(
                    output_name="query_context",
                    mapspec=(
                        ", ".join(["aligned[i]", *map("{}_meta[i]".format, cameras)])
                        + " -> query_context[i]"
                    ),
                    func=collect_kwargs(
                        parameters=("aligned", *map("{}_meta".format, cameras))
                    ),
                ),
                PipeFunc(
                    renames={"context": "query_context"},
                    output_name="filtered",
                    mapspec="query_context[i] -> filtered[i]",
                    func=DataFrameDuckDbQuery(udfs=[Wgs84ToUtm]),
                    bound={
                        "query": """
LOAD spatial;
WITH
  base_data AS (
    SELECT
      *,
      ST_GeomFromWKB(
        ST_Wgs84ToUtm(
          ST_AsWKB(
            ST_Point(
              "meta/Gnss/longitude",
              "meta/Gnss/latitude"
            )
          )
        )
      ) AS ego_geom,
      ST_GeomFromWKB("waypoints/waypoints") AS waypoints_geom
    FROM
      aligned
      SEMI JOIN cam_front_left_meta
        ON aligned."meta/ImageMetadata.cam_front_left/frame_idx"
        = cam_front_left_meta.frame_idx
      SEMI JOIN cam_left_backward_meta
        ON aligned."meta/ImageMetadata.cam_left_backward/frame_idx"
        = cam_left_backward_meta.frame_idx
      SEMI JOIN cam_right_backward_meta
        ON aligned."meta/ImageMetadata.cam_right_backward/frame_idx"
        = cam_right_backward_meta.frame_idx
    WHERE
      COLUMNS (*) IS NOT NULL
      AND "meta/VehicleMotion/speed" > 44
  ),
  normalized_geometries AS (
    SELECT
      *,
      ST_Rotate(
        ST_Translate(
          waypoints_geom,
          -ST_X(ego_geom),
          -ST_Y(ego_geom)
        ),
        radians("waypoints/heading")
      ) AS normalized_waypoints_geom
    FROM
      base_data
  )
SELECT
  * EXCLUDE (
    waypoints_geom,
    normalized_waypoints_geom
  ),
  (
    SELECT
      list(
        [ST_X(p.point_struct.geom), ST_Y(p.point_struct.geom)]
        ORDER BY
          p.point_struct.path
      )
    FROM
      UNNEST(ST_Dump(normalized_waypoints_geom)) AS p(point_struct)
  ) AS "waypoints/waypoints_normalized"
FROM
  normalized_geometries
WHERE
  ST_Contains(
    ST_MakeEnvelope(-150, -150, 150, 150),
    normalized_waypoints_geom
  )
ORDER BY
  "meta/ImageMetadata.cam_front_left/time_stamp";
"""
                    },
                ),
                PipeFunc(
                    renames={"input": "filtered"},
                    output_name="samples",
                    mapspec="filtered[i] -> samples[i]",
                    func=DataFrameGroupByDynamic(
                        index_column=f"meta/ImageMetadata.{cameras[0]}/frame_idx",
                        every="6i",
                        period="6i",
                    ),
                ),
                PipeFunc(
                    renames={"df": "samples"},
                    output_name="samples_cast",
                    mapspec="samples[i] -> samples_cast[i]",
                    func=DataFrameDuckDbQuery(),
                    bound={
                        "query": """
SELECT
   "meta/ImageMetadata.cam_front_left/time_stamp"::TIMESTAMP[6]
        AS "meta/ImageMetadata.cam_front_left/time_stamp",
   "meta/ImageMetadata.cam_front_left/frame_idx"::INT32[6]
        AS "meta/ImageMetadata.cam_front_left/frame_idx",
   "meta/ImageMetadata.cam_left_backward/frame_idx"::INT32[6]
        AS "meta/ImageMetadata.cam_left_backward/frame_idx",
   "meta/ImageMetadata.cam_right_backward/frame_idx"::INT32[6]
        AS "meta/ImageMetadata.cam_right_backward/frame_idx",
   "meta/VehicleMotion/speed"::FLOAT[6]
        AS "meta/VehicleMotion/speed",
   "mcap//ai/safety_score/clip.end_timestamp"::TIMESTAMP[6]
        AS "mcap//ai/safety_score/clip.end_timestamp",
   "mcap//ai/safety_score/score"::FLOAT[6]
        AS "mcap//ai/safety_score/score",
   "waypoints/heading"::FLOAT[6]
        AS "waypoints/heading",
   "waypoints/waypoints_normalized"::FLOAT[2][10][6]
        AS "waypoints/waypoints_normalized"
FROM df
WHERE len("meta/ImageMetadata.cam_front_left/frame_idx") == 6
"""
                    },
                ),
            ],
        ),
    )

    sources = SourcesConfig({
        drive_id: {
            camera: SourceConfig(
                index_column=f"meta/ImageMetadata.{camera}/frame_idx",
                source=HydraConfig(
                    target=TorchCodecFrameSource,
                    source=data_dir / drive_id / f"{camera}.pii.mp4",  # pyright: ignore[reportCallIssue]
                ),
            )
            for camera in cameras
        }
        for drive_id in drive_ids
    })

    return Dataset(samples=samples, sources=sources)


@pytest.fixture
def yaak_hydra() -> Dataset:
    with initialize(version_base=None, config_path=CONFIG_PATH):
        cfg = compose(
            "visualize", overrides=["dataset=yaak", f"+data_dir={DATA_DIR}/yaak"]
        )

    return instantiate(cfg.dataset)


@pytest.mark.parametrize("dataset", [lf("yaak_hydra"), lf("yaak_pydantic")])
def test_yaak(dataset: Dataset) -> None:
    index = [0, 2]
    c = SimpleNamespace(B=len(index))

    match (batch := dataset.get_batch(index)).to_dict():
        case {
            "data": {
                "cam_front_left": Tensor(shape=[c.B, *_]),
                "cam_left_backward": Tensor(shape=[c.B, *_]),
                "cam_right_backward": Tensor(shape=[c.B, *_]),
                "meta/ImageMetadata.cam_front_left/frame_idx": Tensor(shape=[c.B, *_]),
                "meta/ImageMetadata.cam_front_left/time_stamp": Tensor(shape=[c.B, *_]),
                "meta/ImageMetadata.cam_left_backward/frame_idx": Tensor(
                    shape=[c.B, *_]
                ),
                "meta/ImageMetadata.cam_right_backward/frame_idx": Tensor(
                    shape=[c.B, *_]
                ),
                "meta/VehicleMotion/speed": Tensor(shape=[c.B, *_]),
                "mcap//ai/safety_score/clip.end_timestamp": Tensor(shape=[c.B, *_]),
                "mcap//ai/safety_score/score": Tensor(shape=[c.B, *_]),
                "waypoints/heading": Tensor(shape=[c.B, *_]),
                "waypoints/waypoints_normalized": Tensor(shape=[c.B, *_]),
                **data_rest,
            },
            "meta": {"input_id": [*_], "sample_idx": Tensor(shape=[c.B]), **meta_rest},
            **batch_rest,
        } if not any((batch_rest, data_rest, meta_rest)):
            waypoints_normalized: Tensor = batch.data["waypoints/waypoints_normalized"]  # pyright: ignore[reportUnknownVariableType, reportOptionalSubscript]

            assert waypoints_normalized.shape[2:] == (10, 2), "invalid waypoints shape"  # noqa: S101 # pyright: ignore[reportUnknownMemberType] # pyright: ignore[reportUnknownArgumentType]

            assert not (waypoints_normalized == 0.0).all(), "waypoints are all zero"  # noqa: S101 # pyright: ignore[reportUnknownMemberType]

            atol_relative = 1
            relative_distances = torch.linalg.norm(  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
                torch.diff(waypoints_normalized, dim=2),  # pyright: ignore[reportUnknownArgumentType]
                dim=3,
                ord=2,
            )

            # since we duplicate waypoints at the end of the ride
            relative_distances = torch.where(
                relative_distances != 0.0,  # pyright: ignore[reportUnknownArgumentType]
                relative_distances,  # pyright: ignore[reportUnknownArgumentType]
                10.0,
            )

            assert torch.allclose(  # noqa: S101
                relative_distances,
                torch.full_like(relative_distances, 10.0),
                atol=atol_relative,
            ), (
                f"Expected relative distances to be 10 +- {atol_relative}, "
                f"but max distance is {relative_distances.max().item()} "
                f"and min distance is {relative_distances.min().item()}"
            )

            waypoints_radius = torch.linalg.norm(waypoints_normalized, dim=3, ord=2)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            max_radius = 150.0
            assert torch.all(waypoints_radius <= max_radius).item(), (  # noqa: S101 # pyright: ignore[reportUnknownArgumentType]
                f"Expected all waypoints to be within radius {max_radius}, "
                f"but max radius is {waypoints_radius.max().item()}"  # pyright: ignore[reportUnknownMemberType]
            )

            atol_origin = 10
            first_waypoints = waypoints_radius[..., 0]  # pyright: ignore[reportUnknownVariableType]
            assert torch.allclose(  # noqa: S101
                first_waypoints,  # pyright: ignore[reportUnknownArgumentType]
                torch.zeros_like(first_waypoints),  # pyright: ignore[reportUnknownArgumentType]
                atol=atol_origin,
            ), (
                f"Expected first waypoint to be near origin (0,0) "
                f"with tolerance {atol_origin}, "
                f"but found at least one waypoint at {first_waypoints.max().item()}"  # pyright: ignore[reportUnknownMemberType]
            )

        case _:
            logger.error(msg := "invalid batch structure", batch=batch)

            raise AssertionError(msg)

    match (
        batch := dataset.get_batch(
            index,
            keys={"data", ("data", "meta/VehicleMotion/speed"), "meta"},  # pyright: ignore[reportArgumentType]
        )
    ).to_dict():
        case {
            "data": {"meta/VehicleMotion/speed": Tensor(shape=[c.B, *_]), **data_rest},
            "meta": {"input_id": [*_], "sample_idx": Tensor(shape=[c.B]), **meta_rest},
            **batch_rest,
        } if not any((batch_rest, data_rest, meta_rest)):
            pass

        case _:
            logger.error(msg := "invalid batch structure", batch=batch)

            raise AssertionError(msg)

    match (
        batch := dataset.get_batch(index, keys={("data", "meta/VehicleMotion/speed")})  # pyright: ignore[reportArgumentType]
    ).to_dict():
        case {
            "data": {"meta/VehicleMotion/speed": Tensor(shape=[c.B, *_]), **data_rest},
            "meta": None,
            **batch_rest,
        } if not any((batch_rest, data_rest)):
            pass

        case _:
            logger.error(msg := "invalid batch structure", batch=batch)

            raise AssertionError(msg)

    match (batch := dataset.get_batch(index, keys={("meta", "input_id")})).to_dict():  # pyright: ignore[reportArgumentType]
        case {"data": None, "meta": {"input_id": [*_]}, **batch_rest} if not batch_rest:
            pass

        case _:
            logger.error(msg := "invalid batch structure", batch=batch)

            raise AssertionError(msg)
