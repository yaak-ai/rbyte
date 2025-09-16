from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal

from rbyte.io import (
    PathDataFrameBuilder,
    YaakMetadataDataFrameBuilder,  # ty: ignore[possibly-unbound-import]
)

DATA_DIR = Path(__file__).resolve().parent / "data"
CAMERA_ENUM = pl.Enum(
    categories=[
        "cam_front_center",
        "cam_front_left",
        "cam_front_right",
        "cam_left_forward",
        "cam_right_forward",
        "cam_left_backward",
        "cam_right_backward",
        "cam_rear",
    ]
)


def test_PathDataFrameBuilder() -> None:  # noqa: N802
    path = DATA_DIR / "yaak"

    builder = PathDataFrameBuilder(
        fields={"car": pl.String(), "drive": None, "camera": CAMERA_ENUM},
        pattern=r"(?<car>[^/]+)/(?<drive>[^/]+)/(?<camera>\w+)\.pii\.mp4",
    )
    assert builder.__pipefunc_hash__() == "9f8f175d5d4541be"

    df = builder(path)

    assert_frame_equal(
        df,
        pl.DataFrame(
            {
                "car": "Niro098-HQ",
                "drive": "2024-06-18--13-39-54",
                "camera": ["cam_front_left", "cam_left_backward", "cam_right_backward"],
            },
            schema={"car": pl.String(), "drive": pl.String(), "camera": CAMERA_ENUM},
        ),
        check_row_order=False,
    )


def test_YaakMetadataDataFrameBuilder() -> None:  # noqa: N802
    path = DATA_DIR / "yaak" / "Niro098-HQ" / "2024-06-18--13-39-54" / "metadata.log"

    builder = YaakMetadataDataFrameBuilder(
        fields={
            "rbyte.io.yaak.proto.sensor_pb2.ImageMetadata": {
                "time_stamp": pl.Datetime(time_unit="us"),
                "camera_name": CAMERA_ENUM,
            },
            "rbyte.io.yaak.proto.sensor_pb2.Gnss": {
                "time_stamp": pl.Datetime(time_unit="us"),
                "latitude": pl.Float32(),
            },
            "rbyte.io.yaak.proto.can_pb2.VehicleMotion": {
                "time_stamp": pl.Datetime(time_unit="us"),
                "speed": None,
            },
        }
    )

    assert builder.__pipefunc_hash__() == "83f4fddef74a39bc"

    dfs = builder(path)
    match dfs:
        case {
            "VehicleMotion": pl.DataFrame(
                schema={
                    "time_stamp": pl.Datetime(time_unit="us"),
                    "speed": pl.Float32(),
                }
            ),
            "Gnss": pl.DataFrame(
                schema={
                    "time_stamp": pl.Datetime(time_unit="us"),
                    "latitude": pl.Float32(),
                }
            ),
            "ImageMetadata.cam_front_center": pl.DataFrame(
                schema={"time_stamp": pl.Datetime(time_unit="us")}
            ),
            "ImageMetadata.cam_front_left": pl.DataFrame(
                schema={"time_stamp": pl.Datetime(time_unit="us")}
            ),
            "ImageMetadata.cam_front_right": pl.DataFrame(
                schema={"time_stamp": pl.Datetime(time_unit="us")}
            ),
            "ImageMetadata.cam_left_forward": pl.DataFrame(
                schema={"time_stamp": pl.Datetime(time_unit="us")}
            ),
            "ImageMetadata.cam_right_forward": pl.DataFrame(
                schema={"time_stamp": pl.Datetime(time_unit="us")}
            ),
            "ImageMetadata.cam_left_backward": pl.DataFrame(
                schema={"time_stamp": pl.Datetime(time_unit="us")}
            ),
            "ImageMetadata.cam_right_backward": pl.DataFrame(
                schema={"time_stamp": pl.Datetime(time_unit="us")}
            ),
            "ImageMetadata.cam_rear": pl.DataFrame(
                schema={"time_stamp": pl.Datetime(time_unit="us")}
            ),
            **extra,
        } if not extra:
            pass

        case _:
            msg = "unexpected dataframe schemas"
            raise AssertionError(msg)
