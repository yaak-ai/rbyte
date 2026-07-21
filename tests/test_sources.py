from functools import partial
from pathlib import Path
from typing import cast

import polars as pl
import simplejpeg
import torch
from mcap_protobuf.decoder import DecoderFactory

from rbyte.samples.dataframe import DataFrameRowIndexer
from rbyte.samples.hdf5 import Hdf5Reader
from rbyte.samples.mcap import McapReader, ProtobufDecoderFactory
from rbyte.samples.path import PathScanner
from rbyte.samples.video import VideoFrameIndexer
from rbyte.streams.hdf5 import Hdf5Source
from rbyte.streams.mcap import McapSource
from rbyte.streams.numpy import NpySource
from rbyte.streams.path import PathSource
from rbyte.streams.video import TorchCodecVideoSource
from rbyte.utils import datetime_from_nanos

DATA_DIR = Path(__file__).resolve().parent / "data"
ZOD_PATTERN = r"000002_romeo_(?<timestamp>.+)Z"


def test_source_hdf5() -> None:
    path = DATA_DIR / "mimicgen" / "coffee.hdf5"
    prefix = "/data/demo_0"
    samples = cast(
        "pl.DataFrame",
        DataFrameRowIndexer(name="_idx_")(
            Hdf5Reader(fields={"obs/agentview_image": None})(path, prefix=prefix)
        ),
    )
    source = Hdf5Source(path=path, key=f"{prefix}/obs/agentview_image")
    indexes = samples["_idx_"].gather([0, 99, 199]).to_list()

    assert samples.schema == pl.Schema({
        "_idx_": pl.UInt32(),
        "obs/agentview_image": pl.Array(pl.UInt8(), (84, 84, 3)),
    })

    batch = source[indexes]
    assert batch.dtype == torch.uint8
    assert batch.shape == (3, 84, 84, 3)
    assert torch.equal(batch, torch.stack([source[index] for index in indexes]))


def test_source_mcap() -> None:
    path = DATA_DIR / "nuscenes" / "nuScenes-v1.0-mini-scene-0061-cut.mcap"
    topic = "/CAM_FRONT/image_rect_compressed"
    samples = cast(
        "pl.DataFrame",
        DataFrameRowIndexer(name="_idx_")(
            McapReader(
                decoder_factories=[ProtobufDecoderFactory],
                fields={topic: {"log_time": pl.Datetime("ns")}},
            )(path)[topic]  # ty:ignore[invalid-argument-type]
        ),
    )
    source = McapSource(
        path=path,
        topic=topic,
        decoder_factory=DecoderFactory,
        decoder=partial(
            simplejpeg.decode_jpeg, colorspace="rgb", fastdct=True, fastupsample=True
        ),
    )
    assert samples.schema == pl.Schema({
        "_idx_": pl.UInt32(),
        "log_time": pl.Datetime("ns"),
    })

    for indexes in ([0, 5, 11], [11, 0, 5], [0, 0]):
        batch = source[indexes]
        assert batch.dtype == torch.uint8
        assert batch.shape == (len(indexes), 900, 1600, 3)
        assert torch.equal(batch, torch.stack([source[index] for index in indexes]))


def test_source_video() -> None:
    path = (
        DATA_DIR
        / "yaak"
        / "Niro098-HQ"
        / "2024-06-18--13-39-54"
        / "cam_front_left.pii.mp4"
    )
    samples = VideoFrameIndexer(fields={"frame_idx": pl.Int32()})(path)
    source = TorchCodecVideoSource(
        source=path, custom_frame_mappings=Path(f"{path}.frames.json")
    )
    indexes = samples["frame_idx"].gather([0, 2, 4]).to_list()

    assert samples.schema == pl.Schema({"frame_idx": pl.Int32()})

    batch = source[indexes]
    assert batch.dtype == torch.uint8
    assert batch.shape == (3, 3, 1080, 1920)
    assert torch.equal(batch, torch.stack([source[index] for index in indexes]))


def test_source_path() -> None:
    path = DATA_DIR / "zod" / "sequences" / "000002_short" / "camera_front_blur"
    samples = PathScanner(fields={"timestamp": pl.Datetime("ns")}, pattern=ZOD_PATTERN)(
        path
    ).sort("timestamp")
    source = PathSource(
        path=path / "000002_romeo_{:%Y-%m-%dT%H:%M:%S.%f}Z.jpg",
        index_transform=datetime_from_nanos,
        decoder=partial(
            simplejpeg.decode_jpeg, colorspace="rgb", fastdct=True, fastupsample=True
        ),
    )
    indexes = samples["timestamp"].to_physical().gather([0, len(samples) - 1]).to_list()

    assert len(samples) == 10  # ruff:ignore[magic-value-comparison]
    assert samples.schema == pl.Schema({"timestamp": pl.Datetime("ns")})

    batch = source[indexes]
    assert batch.dtype == torch.uint8
    assert batch.shape == (2, 2168, 3848, 3)
    assert torch.equal(batch, torch.stack([source[index] for index in indexes]))


def test_source_npy() -> None:
    path = DATA_DIR / "zod" / "sequences" / "000002_short" / "lidar_velodyne"
    samples = PathScanner(fields={"timestamp": pl.Datetime("ns")}, pattern=ZOD_PATTERN)(
        path
    ).sort("timestamp")
    source = NpySource(
        path=path / "000002_romeo_{:%Y-%m-%dT%H:%M:%S.%f}Z.npy",
        select=["x", "y", "z"],
        index_transform=datetime_from_nanos,
    )
    indexes = samples["timestamp"].to_physical().gather([0, 1]).to_list()
    point_clouds = [source[index] for index in indexes]

    assert len(samples) == 9  # ruff:ignore[magic-value-comparison]
    assert samples.schema == pl.Schema({"timestamp": pl.Datetime("ns")})
    assert all(
        point_cloud.dtype == torch.float32
        and point_cloud.ndim == 2  # ruff:ignore[magic-value-comparison]
        and point_cloud.shape[1] == 3  # ruff:ignore[magic-value-comparison]
        for point_cloud in point_clouds
    )
    assert point_clouds[0].shape[0] != point_clouds[1].shape[0]
