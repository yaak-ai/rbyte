from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

from hydra import compose, initialize
from hydra.utils import instantiate
from structlog import get_logger
from torch import Tensor

if TYPE_CHECKING:
    from rbyte.batch import Batch
    from rbyte.viz.loggers.base import Logger

logger = get_logger(__name__)

CONFIG_PATH = "../config"
DATA_DIR = Path(__file__).resolve().parent / "data"


def test_carla_garage() -> None:
    with initialize(version_base=None, config_path=CONFIG_PATH):
        cfg = compose(
            "visualize",
            overrides=[
                "dataset=carla_garage",
                "logger=rerun/carla_garage",
                f"+data_dir={DATA_DIR}/carla_garage",
            ],
        )

    dataloader = instantiate(cfg.dataloader)

    c = SimpleNamespace(B=cfg.dataloader.batch_size)

    batch = next(iter(dataloader))
    match batch.to_dict():
        case {
            "data": {
                "rgb": Tensor(shape=[c.B, *_]),
                "measurements/_idx_": Tensor(shape=[c.B, *_]),
                "measurements/brake": Tensor(shape=[c.B, *_]),
                "measurements/steer": Tensor(shape=[c.B, *_]),
                "measurements/throttle": Tensor(shape=[c.B, *_]),
                "measurements/speed": Tensor(shape=[c.B, *_]),
                "waypoints/heading": Tensor(shape=[c.B, *_]),
                "waypoints/waypoints_normalized": Tensor(shape=[c.B, *_]),
                **data_rest,
            },
            "meta": {
                "input_id": input_id,
                "sample_idx": Tensor(shape=[c.B]),
                **meta_rest,
            },
            **batch_rest,
        } if set(input_id).issubset(cfg.dataloader.dataset.sources) and not any((
            batch_rest,
            data_rest,
            meta_rest,
        )):
            pass

        case _:
            logger.error(msg := "invalid batch structure", batch=batch)

            raise AssertionError(msg)

    batch_logger: Logger[Batch] = instantiate(cfg.logger, spawn=False)
    batch_logger.log(batch)


def test_mimicgen() -> None:
    with initialize(version_base=None, config_path=CONFIG_PATH):
        cfg = compose(
            "visualize",
            overrides=[
                "dataset=mimicgen",
                "logger=rerun/mimicgen",
                f"+data_dir={DATA_DIR}/mimicgen",
            ],
        )

    dataloader = instantiate(cfg.dataloader)

    c = SimpleNamespace(B=cfg.dataloader.batch_size)

    batch = next(iter(dataloader))
    match batch.to_dict():
        case {
            "data": {
                "obs/agentview_image": Tensor(shape=[c.B, *_]),
                "_idx_": Tensor(shape=[c.B, *_]),
                "obs/robot0_eef_pos": Tensor(shape=[c.B, *_]),
                **data_rest,
            },
            "meta": {
                "input_id": input_id,
                "sample_idx": Tensor(shape=[c.B]),
                **meta_rest,
            },
            **batch_rest,
        } if set(input_id).issubset(cfg.dataloader.dataset.sources) and not any((
            batch_rest,
            data_rest,
            meta_rest,
        )):
            pass

        case _:
            logger.error(msg := "invalid batch structure", batch=batch)

            raise AssertionError(msg)

    batch_logger: Logger[Batch] = instantiate(cfg.logger, spawn=False)
    batch_logger.log(batch)


def test_nuscenes() -> None:
    with initialize(version_base=None, config_path=CONFIG_PATH):
        cfg = compose(
            "visualize",
            overrides=[
                "dataset=nuscenes",
                "logger=rerun/nuscenes",
                f"+data_dir={DATA_DIR}/nuscenes",
            ],
        )

    dataloader = instantiate(cfg.dataloader)

    c = SimpleNamespace(B=cfg.dataloader.batch_size)

    batch = next(iter(dataloader))
    match batch.to_dict():
        case {
            "data": {
                "CAM_FRONT": Tensor(shape=[c.B, *_]),
                "CAM_FRONT_LEFT": Tensor(shape=[c.B, *_]),
                "CAM_FRONT_RIGHT": Tensor(shape=[c.B, *_]),
                "/CAM_FRONT/image_rect_compressed/_idx_": Tensor(shape=[c.B, *_]),
                "/CAM_FRONT/image_rect_compressed/log_time": Tensor(shape=[c.B, *_]),
                "/CAM_FRONT_LEFT/image_rect_compressed/_idx_": Tensor(shape=[c.B, *_]),
                "/CAM_FRONT_RIGHT/image_rect_compressed/_idx_": Tensor(shape=[c.B, *_]),
                "/odom/vel.x": Tensor(shape=[c.B, *_]),
                **data_rest,
            },
            "meta": {
                "input_id": input_id,
                "sample_idx": Tensor(shape=[c.B]),
                **meta_rest,
            },
            **batch_rest,
        } if set(input_id).issubset(cfg.dataloader.dataset.sources) and not any((
            batch_rest,
            data_rest,
            meta_rest,
        )):
            pass

        case _:
            logger.error(msg := "invalid batch structure", batch=batch)

            raise AssertionError(msg)

    batch_logger: Logger[Batch] = instantiate(cfg.logger, spawn=False)
    batch_logger.log(batch)


def test_yaak(tmp_path: Path) -> None:
    with initialize(version_base=None, config_path=CONFIG_PATH):
        cfg = compose(
            "visualize",
            overrides=[
                "dataset=yaak",
                f"dataset.samples.run_folder={tmp_path}",
                "logger=rerun/yaak",
                f"+data_dir={DATA_DIR}/yaak",
            ],
        )

    dataloader = instantiate(cfg.dataloader)

    c = SimpleNamespace(B=cfg.dataloader.batch_size)

    batch = next(iter(dataloader))
    match batch.to_dict():
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
            "meta": {
                "input_id": input_id,
                "sample_idx": Tensor(shape=[c.B]),
                **meta_rest,
            },
            **batch_rest,
        } if set(input_id).issubset(cfg.dataloader.dataset.sources) and not any((
            batch_rest,
            data_rest,
            meta_rest,
        )):
            pass

        case _:
            logger.error(msg := "invalid batch structure", batch=batch)

            raise AssertionError(msg)

    batch_logger: Logger[Batch] = instantiate(cfg.logger, spawn=False)
    batch_logger.log(batch)


def test_zod() -> None:
    with initialize(version_base=None, config_path=CONFIG_PATH):
        cfg = compose(
            "visualize",
            overrides=[
                "dataloader=unbatched",
                "dataset=zod",
                "logger=rerun/zod",
                f"+data_dir={DATA_DIR}/zod",
            ],
        )

    dataloader = instantiate(cfg.dataloader)

    c = SimpleNamespace(B=cfg.dataloader.batch_size)

    batch = next(iter(dataloader))
    match batch.to_dict():
        case {
            "data": {
                "camera_front_blur": Tensor(shape=[c.B, *_]),
                "camera_front_blur_meta/timestamp": Tensor(shape=[c.B, *_]),
                "lidar_velodyne": Tensor(shape=[c.B, *_]),
                "lidar_velodyne_meta/timestamp": Tensor(shape=[c.B, *_]),
                "vehicle_data/ego_vehicle_controls/acceleration_pedal/ratio/unitless/value": Tensor(  # noqa: E501
                    shape=[c.B, *_]
                ),
                "vehicle_data/ego_vehicle_controls/steering_wheel_angle/angle/radians/value": Tensor(  # noqa: E501
                    shape=[c.B, *_]
                ),
                "vehicle_data/ego_vehicle_controls/timestamp/nanoseconds/value": Tensor(
                    shape=[c.B, *_]
                ),
                "vehicle_data/satellite/speed/meters_per_second/value": Tensor(
                    shape=[c.B, *_]
                ),
                **data_rest,
            },
            "meta": {
                "input_id": input_id,
                "sample_idx": Tensor(shape=[c.B, *_]),
                **meta_rest,
            },
            **batch_rest,
        } if set(input_id).issubset(cfg.dataloader.dataset.sources) and not any((
            batch_rest,
            data_rest,
            meta_rest,
        )):
            pass

        case _:
            logger.error(msg := "invalid batch structure", batch=batch)

            raise AssertionError(msg)

    batch_logger: Logger[Batch] = instantiate(cfg.logger, spawn=False)
    batch_logger.log(batch)
