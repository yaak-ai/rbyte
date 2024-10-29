from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize
from hydra.utils import instantiate
from structlog import get_logger
from torch import Tensor

logger = get_logger(__name__)

CONFIG_PATH = "../config"
DATA_DIR = Path(__file__).resolve().parent / "data"


def test_mimicgen() -> None:
    with initialize(version_base=None, config_path=CONFIG_PATH):
        cfg = compose(
            "visualize", overrides=["dataset=mimicgen", f"+data_dir={DATA_DIR}"]
        )

    dataloader = instantiate(cfg.dataloader)

    c = SimpleNamespace(
        B=cfg.dataloader.batch_size, S=cfg.dataloader.dataset.sample_builder.length
    )

    batch = next(iter(dataloader))
    match batch.to_dict():
        case {
            "frame": {
                "obs/agentview_image": Tensor(shape=[c.B, c.S, *_]),
                **frame_rest,
            },
            "table": {
                "_idx_": Tensor(shape=[c.B, c.S]),
                "obs/robot0_eef_pos": Tensor(shape=[c.B, c.S, *_]),
                **table_rest,
            },
            "meta": {
                "input_id": input_id,
                "sample_idx": Tensor(shape=[c.B]),
                **meta_rest,
            },
            **batch_rest,
        } if set(input_id).issubset(cfg.dataloader.dataset.inputs) and not any((
            batch_rest,
            frame_rest,
            table_rest,
            meta_rest,
        )):
            pass

        case _:
            logger.error(msg := "invalid batch structure", batch=batch)

            raise AssertionError(msg)


def test_nuscenes_mcap() -> None:
    with initialize(version_base=None, config_path=CONFIG_PATH):
        cfg = compose(
            "visualize", overrides=["dataset=nuscenes_mcap", f"+data_dir={DATA_DIR}"]
        )

    dataloader = instantiate(cfg.dataloader)

    c = SimpleNamespace(
        B=cfg.dataloader.batch_size, S=cfg.dataloader.dataset.sample_builder.length
    )

    batch = next(iter(dataloader))
    match batch.to_dict():
        case {
            "frame": {
                "CAM_FRONT": Tensor(shape=[c.B, c.S, *_]),
                "CAM_FRONT_LEFT": Tensor(shape=[c.B, c.S, *_]),
                "CAM_FRONT_RIGHT": Tensor(shape=[c.B, c.S, *_]),
                **frame_rest,
            },
            "table": {
                "/CAM_FRONT/image_rect_compressed/_idx_": Tensor(shape=[c.B, c.S]),
                "/CAM_FRONT_LEFT/image_rect_compressed/_idx_": Tensor(shape=[c.B, c.S]),
                "/CAM_FRONT_RIGHT/image_rect_compressed/_idx_": Tensor(
                    shape=[c.B, c.S]
                ),
                "/CAM_FRONT/image_rect_compressed/log_time": Tensor(shape=[c.B, c.S]),
                "/CAM_FRONT_LEFT/image_rect_compressed/log_time": Tensor(
                    shape=[c.B, c.S]
                ),
                "/CAM_FRONT_RIGHT/image_rect_compressed/log_time": Tensor(
                    shape=[c.B, c.S]
                ),
                "/odom/vel.x": Tensor(shape=[c.B, c.S]),
                **table_rest,
            },
            "meta": {
                "input_id": input_id,
                "sample_idx": Tensor(shape=[c.B]),
                **meta_rest,
            },
            **batch_rest,
        } if set(input_id).issubset(cfg.dataloader.dataset.inputs) and not any((
            batch_rest,
            frame_rest,
            table_rest,
            meta_rest,
        )):
            pass

        case _:
            logger.error(msg := "invalid batch structure", batch=batch)

            raise AssertionError(msg)


def test_nuscenes_rrd() -> None:
    with initialize(version_base=None, config_path=CONFIG_PATH):
        cfg = compose(
            "visualize", overrides=["dataset=nuscenes_rrd", f"+data_dir={DATA_DIR}"]
        )

    dataloader = instantiate(cfg.dataloader)

    c = SimpleNamespace(
        B=cfg.dataloader.batch_size, S=cfg.dataloader.dataset.sample_builder.length
    )

    batch = next(iter(dataloader))
    match batch.to_dict():
        case {
            "frame": {
                "CAM_FRONT": Tensor(shape=[c.B, c.S, *_]),
                "CAM_FRONT_LEFT": Tensor(shape=[c.B, c.S, *_]),
                "CAM_FRONT_RIGHT": Tensor(shape=[c.B, c.S, *_]),
                **frame_rest,
            },
            "table": {
                "/world/ego_vehicle/CAM_FRONT/timestamp": Tensor(shape=[c.B, c.S, *_]),
                "/world/ego_vehicle/CAM_FRONT/_idx_": Tensor(shape=[c.B, c.S, *_]),
                "/world/ego_vehicle/CAM_FRONT_LEFT/timestamp": Tensor(
                    shape=[c.B, c.S, *_]
                ),
                "/world/ego_vehicle/CAM_FRONT_LEFT/_idx_": Tensor(shape=[c.B, c.S, *_]),
                "/world/ego_vehicle/CAM_FRONT_RIGHT/timestamp": Tensor(
                    shape=[c.B, c.S, *_]
                ),
                "/world/ego_vehicle/CAM_FRONT_RIGHT/_idx_": Tensor(
                    shape=[c.B, c.S, *_]
                ),
                "/world/ego_vehicle/LIDAR_TOP/timestamp": Tensor(shape=[c.B, c.S, *_]),
                "/world/ego_vehicle/LIDAR_TOP/Position3D": Tensor(shape=[c.B, c.S, *_]),
                **table_rest,
            },
            "meta": {
                "input_id": input_id,
                "sample_idx": Tensor(shape=[c.B]),
                **meta_rest,
            },
            **batch_rest,
        } if set(input_id).issubset(cfg.dataloader.dataset.inputs) and not any((
            batch_rest,
            frame_rest,
            table_rest,
            meta_rest,
        )):
            pass

        case _:
            logger.error(msg := "invalid batch structure", batch=batch)

            raise AssertionError(msg)


def test_yaak() -> None:
    with initialize(version_base=None, config_path=CONFIG_PATH):
        cfg = compose("visualize", overrides=["dataset=yaak", f"+data_dir={DATA_DIR}"])

    dataloader = instantiate(cfg.dataloader)

    c = SimpleNamespace(
        B=cfg.dataloader.batch_size, S=cfg.dataloader.dataset.sample_builder.length
    )

    batch = next(iter(dataloader))
    match batch.to_dict():
        case {
            "frame": {
                "cam_front_left": Tensor(shape=[c.B, c.S, *_]),
                "cam_left_backward": Tensor(shape=[c.B, c.S, *_]),
                "cam_right_backward": Tensor(shape=[c.B, c.S, *_]),
                **frame_rest,
            },
            "table": {
                "ImageMetadata.cam_front_left.frame_idx": Tensor(shape=[c.B, c.S]),
                "ImageMetadata.cam_front_left.time_stamp": Tensor(shape=[c.B, c.S]),
                "ImageMetadata.cam_left_backward.frame_idx": Tensor(shape=[c.B, c.S]),
                "ImageMetadata.cam_left_backward.time_stamp": Tensor(shape=[c.B, c.S]),
                "ImageMetadata.cam_right_backward.frame_idx": Tensor(shape=[c.B, c.S]),
                "ImageMetadata.cam_right_backward.time_stamp": Tensor(shape=[c.B, c.S]),
                "VehicleMotion.gear": Tensor(shape=[c.B, c.S]),
                "VehicleMotion.speed": Tensor(shape=[c.B, c.S]),
                "VehicleMotion.time_stamp": Tensor(shape=[c.B, c.S]),
                "/ai/safety_score.clip.end_timestamp": Tensor(shape=[c.B, c.S]),
                "/ai/safety_score.score": Tensor(shape=[c.B, c.S]),
                **table_rest,
            },
            "meta": {
                "input_id": input_id,
                "sample_idx": Tensor(shape=[c.B]),
                **meta_rest,
            },
        } if set(input_id).issubset(cfg.dataloader.dataset.inputs) and not any((
            frame_rest,
            table_rest,
            meta_rest,
        )):
            pass

        case _:
            logger.error(msg := "invalid batch structure", batch=batch)

            raise AssertionError(msg)
