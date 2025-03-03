from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal

from rbyte.io import PathDataFrameBuilder

DATA_DIR = Path(__file__).resolve().parent / "data"


def test_PathDataFrameBuilder() -> None:  # noqa: N802
    builder = PathDataFrameBuilder(
        fields={"car": None, "drive": None, "camera": None},
        pattern=r"(?<car>[^/]+)/(?<drive>[^/]+)/(?<camera>\w+)\.pii\.mp4",
    )
    df = builder(DATA_DIR / "yaak")

    assert_frame_equal(
        df,
        pl.DataFrame({
            "car": "Niro098-HQ",
            "drive": "2024-06-18--13-39-54",
            "camera": ["cam_front_left", "cam_left_backward", "cam_right_backward"],
        }),
        check_row_order=False,
    )
