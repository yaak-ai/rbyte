import multiprocessing
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl
import pytest
from hydra import compose, initialize
from hydra.utils import instantiate
from m2df import MessageType
from makefun import with_signature
from pipefunc import PipeFunc, Pipeline
from structlog import get_logger

from rbyte import Dataset
from rbyte.config import HydraConfig, PipelineInstanceConfig, StreamConfig
from rbyte.io import (
    DataFrameAligner,
    DataFrameGroupByDynamic,
    DuckDBDataFrameQuery,
    McapDataFrameBuilder,
    ProtobufMcapDecoderFactory,
    RouteMatchedWaypointGenerator,
    TorchCodecFrameSource,
    TreeBroadcastMapper,
    TreeItemGetter,
    VideoDataFrameBuilder,
    YaakMetadataDataFrameBuilder,
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


# TODO: cleaner way of doing this while preserving fixture caching?  # ruff:ignore[line-contains-todo]
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
    drive_queries: dict[str, str] = {
        "Niro098-HQ/2024-06-18--13-39-54": "SELECT * FROM self WHERE time_stamp BETWEEN TIMESTAMP('2024-06-18 13:39:55') AND TIMESTAMP('2024-06-18 13:40:15')"  # ruff:ignore[line-too-long]
    }
    cameras = ["cam_front_left", "cam_left_backward", "cam_right_backward"]

    samples = PipelineInstanceConfig(
        inputs={
            "input_id": list(drive_queries.keys()),
            "meta_path": [data_dir / i / "metadata.log" for i in drive_queries],
            "meta_query": list(drive_queries.values()),
            "mcap_path": [data_dir / i / "ai.mcap" for i in drive_queries],
            "route_path": [data_dir / i / "map-matched.json" for i in drive_queries],
        }
        | {
            f"{camera}_path": [
                data_dir / i / f"{camera}.pii.mp4" for i in drive_queries
            ]
            for camera in cameras
        },
        executor=ProcessPoolExecutor(
            max_workers=1, mp_context=multiprocessing.get_context("forkserver")
        ),
        pipeline=Pipeline(
            validate_type_annotations=False,
            functions=[
                PipeFunc(
                    renames={"path": "meta_path"},
                    output_name="meta_raw",
                    mapspec="meta_path[i] -> meta_raw[i]",
                    func=YaakMetadataDataFrameBuilder(
                        messages={
                            MessageType.ImageMetadata: {
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
                            MessageType.VehicleMotion: {
                                "time_stamp": pl.Datetime(),
                                "speed": pl.Float32(),
                            },
                            MessageType.Gnss: {
                                "time_stamp": pl.Datetime(),
                                "latitude": pl.Float64(),
                                "longitude": pl.Float64(),
                                "heading": pl.Float64(),
                            },
                        }
                    ),
                ),
                PipeFunc(
                    renames={"left": "meta_raw", "right": "meta_query"},
                    output_name="meta",
                    mapspec="meta_raw[i], meta_query[i] -> meta[i]",
                    func=TreeBroadcastMapper(),
                    bound={"func": pl.DataFrame.sql},
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
                    renames={"mapping": "meta_raw"},
                    output_name="gnss",
                    mapspec="meta_raw[i] -> gnss[i]",
                    func=TreeItemGetter(key_path=("Gnss",)),
                ),
                PipeFunc(
                    output_name="route",
                    mapspec="route_path[i] -> route[i]",
                    func=with_signature("build_route(*, route_path)")(
                        DuckDBDataFrameQuery(
                            extensions=["spatial"],
                            config={"TimeZone": "UTC"},
                            query="""
WITH coordinates AS (
  SELECT coordinate, ordinality
  FROM read_json_auto($route_path),
       UNNEST(features[1].geometry.coordinates) WITH ORDINALITY
         AS coordinates(coordinate, ordinality)
),
route_metric AS (
  SELECT
    ST_Transform(
      ST_Point(coordinate[1], coordinate[2]),
      'EPSG:4326', 'EPSG:25832', always_xy := true
    ) AS geom,
    ordinality
  FROM coordinates
)
SELECT
  ST_X(geom) AS easting,
  ST_Y(geom) AS northing
FROM route_metric
ORDER BY ordinality
""",
                        )
                    ),
                ),
                PipeFunc(
                    output_name="gnss_metric",
                    mapspec="gnss[i] -> gnss_metric[i]",
                    func=with_signature("build_gnss_metric(*, gnss)")(
                        DuckDBDataFrameQuery(
                            extensions=["spatial"],
                            config={"TimeZone": "UTC"},
                            query="""
WITH projected AS (
  SELECT
    * EXCLUDE (latitude, longitude),
    ST_Transform(
      ST_Point(longitude, latitude),
      'EPSG:4326', 'EPSG:25832', always_xy := true
    ) AS geom
  FROM gnss
)
SELECT
  time_stamp,
  ST_X(geom) AS easting,
  ST_Y(geom) AS northing,
  heading
FROM projected
ORDER BY time_stamp
""",
                        )
                    ),
                ),
                PipeFunc(
                    renames={"gnss": "gnss_metric"},
                    output_name="waypoints",
                    mapspec="route[i], gnss_metric[i] -> waypoints[i]",
                    func=RouteMatchedWaypointGenerator(
                        config=RouteMatchedWaypointGenerator.Config(
                            waypoint_offsets_m=(1, 11, 21, 31, 41, 51, 61, 71, 81, 91)
                        )
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
                                    key="time_stamp",
                                    columns=OrderedDict({
                                        "easting": AsofColumnAlignConfig(
                                            strategy="nearest"
                                        ),
                                        "northing": AsofColumnAlignConfig(
                                            strategy="nearest"
                                        ),
                                        "heading": AsofColumnAlignConfig(
                                            strategy="nearest"
                                        ),
                                        "waypoints/position": AsofColumnAlignConfig(
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
                        "filter(*, aligned, cam_front_left_meta, cam_left_backward_meta, cam_right_backward_meta)"  # ruff:ignore[line-too-long]
                    )(
                        DuckDBDataFrameQuery(
                            extensions=["spatial"],
                            query="""
WITH
  base_data AS (
    SELECT
      *,
      ST_Point("waypoints/easting", "waypoints/northing") AS ego_geom,
      ST_Collect(
        (
          SELECT list(
            ST_Point(position[1], position[2])
            ORDER BY ordinality
          )
          FROM UNNEST(
            CAST(aligned."waypoints/waypoints/position" AS DOUBLE[][])
          ) WITH ORDINALITY AS waypoint(position, ordinality)
        )
      ) AS waypoints_geom
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
      ST_Transform(
        waypoints_geom,
        'EPSG:25832', 'EPSG:4326', always_xy := true
      ) AS wgs84_waypoints_geom,
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
    wgs84_waypoints_geom,
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
  ) AS "waypoints/xy_normalized",
  (
    SELECT
      list(
        [ST_Y(p.point_struct.geom), ST_X(p.point_struct.geom)]
        ORDER BY
          p.point_struct.path
      )
    FROM
      UNNEST(ST_Dump(wgs84_waypoints_geom)) AS p(point_struct)
  ) AS "waypoints/WGS84"
FROM
  normalized_geometries
WHERE
  ST_Contains(
    ST_MakeEnvelope(-150, -150, 150, 150),
    normalized_waypoints_geom
  )
ORDER BY
  "meta/ImageMetadata.cam_front_left/time_stamp";
""",
                        )
                    ),
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
                    func=with_signature("cast(*, samples)")(
                        DuckDBDataFrameQuery(
                            query="""
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
   "waypoints/xy_normalized"::FLOAT[2][10][6]
        AS "waypoints/xy_normalized",
   "waypoints/WGS84"::DOUBLE[2][10][6]
        AS "waypoints/WGS84"
FROM samples
WHERE len("meta/ImageMetadata.cam_front_left/frame_idx") == 6
"""
                        )
                    ),
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
                    source=(data_dir / input_id / f"{camera}.pii.mp4").as_posix(),
                    custom_frame_mappings=(
                        data_dir / input_id / f"{camera}.pii.mp4.frames.json"
                    ).as_posix(),
                    transforms=[
                        {"_target_": "torchcodec.transforms.Resize", "size": [324, 576]}
                    ],
                )
                for input_id in drive_queries
            },
        )
        for camera in cameras
    }

    return Dataset.from_config(samples=samples, streams=streams)


@pytest.fixture(params=["mimicgen", "nuscenes", "yaak", "zod"])
def rerun_logger(request: pytest.FixtureRequest) -> RerunLogger:
    name = request.param
    with initialize(version_base=None, config_path=f"{CONFIG_PATH}/logger/rerun"):
        cfg = compose(f"{name}", overrides=["++spawn=false"])

    return instantiate(cfg)
