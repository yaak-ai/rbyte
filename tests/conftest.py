from collections import OrderedDict
from pathlib import Path

import polars as pl
import pytest
from hydra import compose, initialize
from hydra.utils import instantiate
from makefun import with_signature
from pipefunc import PipeFunc, Pipeline
from structlog import get_logger

from rbyte import Dataset
from rbyte.config import HydraConfig, PipelineInstanceConfig, StreamConfig
from rbyte.io import (
    DataFrameAligner,
    DataFrameDuckDbQuery,
    DataFrameGroupByDynamic,
    DuckDbDataFrameBuilder,
    McapDataFrameBuilder,  # ty: ignore[possibly-unbound-import]
    ProtobufMcapDecoderFactory,  # ty: ignore[possibly-unbound-import]
    TorchCodecFrameSource,  # ty: ignore[possibly-unbound-import]
    VideoDataFrameBuilder,  # ty: ignore[possibly-unbound-import]
    WaypointBuilder,  # ty: ignore[possibly-unbound-import]
    YaakMetadataDataFrameBuilder,  # ty: ignore[possibly-unbound-import]
)
from rbyte.io.dataframe.aligner import (
    AlignConfig,
    AsofColumnAlignConfig,
    InterpColumnAlignConfig,
)
from rbyte.io.dataframe.concater import DataFrameConcater
from rbyte.viz.loggers.rerun_logger import RerunLogger

logger = get_logger(__name__)

CONFIG_PATH = "../config"
DATA_DIR = Path(__file__).resolve().parent / "data"


def _build_dataset(name: str) -> Dataset:
    with initialize(version_base=None, config_path=CONFIG_PATH):
        cfg = compose(
            "dataset", overrides=[f"dataset={name}", f"+data_dir={DATA_DIR}/{name}"]
        )

    return instantiate(cfg.dataset)


# TODO: cleaner way of doing this while preserving fixture caching?  # noqa: FIX002
@pytest.fixture(scope="session")
def carla_garage_dataset() -> Dataset:
    return _build_dataset("carla_garage")


@pytest.fixture(scope="session")
def mimicgen_dataset() -> Dataset:
    return _build_dataset("mimicgen")


@pytest.fixture(scope="session")
def nuscenes_dataset() -> Dataset:
    return _build_dataset("nuscenes")


@pytest.fixture(scope="session")
def yaak_dataset() -> Dataset:
    return _build_dataset("yaak")


@pytest.fixture(scope="session")
def zod_dataset() -> Dataset:
    return _build_dataset("zod")


@pytest.fixture(scope="session")
def yaak_dataset_pydantic() -> Dataset:
    data_dir = DATA_DIR / "yaak"
    input_ids = ["Niro098-HQ/2024-06-18--13-39-54"]
    cameras = ["cam_front_left", "cam_left_backward", "cam_right_backward"]

    samples = PipelineInstanceConfig(
        inputs={
            "input_id": input_ids,
            "meta_path": [data_dir / i / "metadata.log" for i in input_ids],
            "mcap_path": [data_dir / i / "ai.mcap" for i in input_ids],
            "waypoints_path": [data_dir / i / "waypoints.json" for i in input_ids],
        }
        | {
            f"{camera}_path": [data_dir / i / f"{camera}.pii.mp4" for i in input_ids]
            for camera in cameras
        },
        parallel=False,  # ty: ignore[unknown-argument]
        pipeline=Pipeline(
            validate_type_annotations=False,
            functions=[
                PipeFunc(
                    renames={"path": "meta_path"},
                    output_name="meta",
                    mapspec="meta_path[i] -> meta[i]",
                    func=YaakMetadataDataFrameBuilder(
                        fields={
                            "rbyte.io.yaak.proto.sensor_pb2.ImageMetadata": {
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
                    func=DuckDbDataFrameBuilder(),
                    bound={
                        "query": """
LOAD spatial;
SET TimeZone = 'UTC';
SELECT TO_TIMESTAMP(timestamp)::TIMESTAMP as timestamp,
   heading,
   ST_AsWKB(
       ST_Transform(geom, 'EPSG:4326', 'EPSG:25832', always_xy := true)) AS geometry
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
                    output_name="aligned",
                    mapspec="meta[i], mcap[i], waypoints[i] -> aligned[i]",
                    func=with_signature("align(*, meta, mcap, waypoints)")(
                        DataFrameAligner(
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
                                                        strategy="nearest",
                                                        tolerance="20ms",
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
                        )
                    ),
                ),
                PipeFunc(
                    output_name="filtered",
                    mapspec=(
                        ", ".join(["aligned[i]", *map("{}_meta[i]".format, cameras)])
                        + " -> filtered[i]"
                    ),
                    func=with_signature(
                        "df_query(*, query, aligned, cam_front_left_meta, cam_left_backward_meta, cam_right_backward_meta)"  # noqa: E501
                    )(DataFrameDuckDbQuery()),
                    bound={
                        "query": """
LOAD spatial;
WITH
  base_data AS (
    SELECT
      *,
      ST_Transform(
        ST_Point("meta/Gnss/longitude", "meta/Gnss/latitude"),
        'EPSG:4326', 'EPSG:25832', always_xy := true
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
                    output_name="samples_cast",
                    mapspec="samples[i] -> samples_cast[i]",
                    func=with_signature("df_query(*, query, samples)")(
                        DataFrameDuckDbQuery()
                    ),
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
FROM samples
WHERE len("meta/ImageMetadata.cam_front_left/frame_idx") == 6
"""
                    },
                ),
                PipeFunc(
                    renames={"keys": "input_id", "values": "samples_cast"},
                    output_name="samples_aggregated",
                    func=DataFrameConcater(key_column="input_id"),
                ),
            ],
        ),
    )

    streams = {
        camera: StreamConfig(
            index=f"meta/ImageMetadata.{camera}/frame_idx",
            sources={
                input_id: HydraConfig(
                    target=TorchCodecFrameSource,
                    source=(data_dir / input_id / f"{camera}.pii.mp4").as_posix(),  # ty: ignore[unknown-argument]
                )
                for input_id in input_ids
            },
        )
        for camera in cameras
    }

    return Dataset.from_config(samples=samples, streams=streams)


@pytest.fixture(params=["carla_garage", "mimicgen", "nuscenes", "yaak", "zod"])
def rerun_logger(request: pytest.FixtureRequest) -> RerunLogger:
    name = request.param
    with initialize(version_base=None, config_path=f"{CONFIG_PATH}/logger/rerun"):
        cfg = compose(f"{name}", overrides=["++spawn=false"])

    return instantiate(cfg)
