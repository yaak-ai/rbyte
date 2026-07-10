from pathlib import Path

import polars as pl
from m2df import MessageType
from polars.testing import assert_frame_equal

from rbyte.io import PathDataFrameBuilder, YaakMetadataDataFrameBuilder

DATA_DIR = Path(__file__).resolve().parent / "data"
TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "config" / "_templates"
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


def test_PathDataFrameBuilder() -> None:  # ruff:ignore[invalid-function-name]
    path = DATA_DIR / "yaak"

    builder = PathDataFrameBuilder(
        fields={"car": pl.String(), "drive": None, "camera": CAMERA_ENUM},
        pattern=r"(?<car>[^/]+)/(?<drive>[^/]+)/(?<camera>\w+)\.pii\.mp4$",
    )
    assert builder.__pipefunc_hash__() == "fbdd8b0111ed61bf"

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


def test_YaakMetadataDataFrameBuilder() -> None:  # ruff:ignore[invalid-function-name]
    path = DATA_DIR / "yaak" / "Niro098-HQ" / "2024-06-18--13-39-54" / "metadata.log"

    builder = YaakMetadataDataFrameBuilder(
        messages={
            MessageType.ImageMetadata: {
                "time_stamp": pl.Datetime(time_unit="us"),
                "camera_name": CAMERA_ENUM,
            },
            MessageType.Gnss: {
                "time_stamp": pl.Datetime(time_unit="us"),
                "latitude": pl.Float32(),
            },
            MessageType.VehicleMotion: {
                "time_stamp": pl.Datetime(time_unit="us"),
                "speed": None,
            },
        }
    )

    assert builder.__pipefunc_hash__() == "00f2bcc91542f647"

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
