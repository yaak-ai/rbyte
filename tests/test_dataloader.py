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
                "meta/_idx_": Tensor(shape=[c.B, *_]),
                "meta/brake": Tensor(shape=[c.B, *_]),
                "meta/pos_global_x": Tensor(shape=[c.B, *_]),
                "meta/pos_global_y": Tensor(shape=[c.B, *_]),
                "meta/steer": Tensor(shape=[c.B, *_]),
                "meta/throttle": Tensor(shape=[c.B, *_]),
                "meta/speed": Tensor(shape=[c.B, *_]),
                "waypoints/heading": Tensor(shape=[c.B, *_]),
                "waypoints/waypoints": Tensor(shape=[c.B, *_]),
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


def test_nuscenes_mcap() -> None:
    with initialize(version_base=None, config_path=CONFIG_PATH):
        cfg = compose(
            "visualize",
            overrides=[
                "dataset=nuscenes/mcap",
                "logger=rerun/nuscenes/mcap",
                f"+data_dir={DATA_DIR}/nuscenes/mcap",
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


def test_nuscenes_rrd() -> None:
    with initialize(version_base=None, config_path=CONFIG_PATH):
        cfg = compose(
            "visualize",
            overrides=[
                "dataset=nuscenes/rrd",
                "logger=rerun/nuscenes/rrd",
                f"+data_dir={DATA_DIR}/nuscenes/rrd",
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
                "/world/ego_vehicle/CAM_FRONT/timestamp": Tensor(shape=[c.B, *_]),
                "/world/ego_vehicle/CAM_FRONT/_idx_": Tensor(shape=[c.B, *_]),
                "/world/ego_vehicle/CAM_FRONT_LEFT/_idx_": Tensor(shape=[c.B, *_]),
                "/world/ego_vehicle/CAM_FRONT_RIGHT/_idx_": Tensor(shape=[c.B, *_]),
                "/world/ego_vehicle/LIDAR_TOP/Position3D": Tensor(shape=[c.B, *_]),
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


def test_yaak() -> None:
    with initialize(version_base=None, config_path=CONFIG_PATH):
        cfg = compose(
            "visualize",
            overrides=[
                "dataset=yaak",
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
                "meta/VehicleMotion/gear": Tensor(shape=[c.B, *_]),
                "meta/VehicleMotion/speed": Tensor(shape=[c.B, *_]),
                "meta/Gnss/latitude": Tensor(shape=[c.B, *_]),
                "meta/Gnss/longitude": Tensor(shape=[c.B, *_]),
                "mcap//ai/safety_score/clip.end_timestamp": Tensor(shape=[c.B, *_]),
                "mcap//ai/safety_score/score": Tensor(shape=[c.B, *_]),
                "waypoints/heading": Tensor(shape=[c.B, *_]),
                "waypoints/waypoints": Tensor(shape=[c.B, *_]),
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
            overrides=["dataset=zod", "logger=rerun/zod", f"+data_dir={DATA_DIR}/zod"],
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
