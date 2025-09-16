from pathlib import Path
from types import SimpleNamespace
from typing import cast

import dill  # noqa: S403
import pytest
import torch
from pytest_lazy_fixtures import lf
from structlog import get_logger
from torch import Tensor

from rbyte import Dataset

logger = get_logger(__name__)


@pytest.mark.parametrize("dataset", [lf("yaak_dataset"), lf("yaak_dataset_pydantic")])
def test_yaak_dataset(dataset: Dataset) -> None:
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
            "meta": {"input_id": [*_], **meta_rest},
            **batch_rest,
        } if not any((batch_rest, data_rest, meta_rest)):
            waypoints_normalized = cast(
                Tensor, batch.data["waypoints/waypoints_normalized"]
            )

            assert waypoints_normalized.shape[2:] == (10, 2), "invalid waypoints shape"

            assert not (waypoints_normalized == 0.0).all(), "waypoints are all zero"

            atol_relative = 1
            relative_distances = torch.linalg.norm(
                torch.diff(waypoints_normalized, dim=2), dim=3, ord=2
            )

            # since we duplicate waypoints at the end of the ride
            relative_distances = torch.where(
                relative_distances != 0.0, relative_distances, 10.0
            )

            assert torch.allclose(
                relative_distances,
                torch.full_like(relative_distances, 10.0),
                atol=atol_relative,
            ), (
                f"Expected relative distances to be 10 +- {atol_relative}, "
                f"but max distance is {relative_distances.max().item()} "
                f"and min distance is {relative_distances.min().item()}"
            )

            waypoints_radius = torch.linalg.norm(waypoints_normalized, dim=3, ord=2)
            max_radius = 150.0
            assert torch.all(waypoints_radius <= max_radius).item(), (
                f"Expected all waypoints to be within radius {max_radius}, "
                f"but max radius is {waypoints_radius.max().item()}"
            )

            atol_origin = 10
            first_waypoints = waypoints_radius[..., 0]
            assert torch.allclose(
                first_waypoints, torch.zeros_like(first_waypoints), atol=atol_origin
            ), (
                f"Expected first waypoint to be near origin (0,0) "
                f"with tolerance {atol_origin}, "
                f"but found at least one waypoint at {first_waypoints.max().item()}"
            )

        case _:
            logger.error(msg := "invalid batch structure", batch=batch)

            raise AssertionError(msg)

    match (batch := dataset.get_batch(index, include_streams=False)).to_dict():
        case {
            "data": {
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
            "meta": {"input_id": [*_], **meta_rest},
            **batch_rest,
        } if not any((batch_rest, data_rest, meta_rest)):
            pass

        case _:
            logger.error(msg := "invalid batch structure", batch=batch)

            raise AssertionError(msg)


def test_carla_garage_dataset(carla_garage_dataset: Dataset) -> None:
    index = [0, 2]
    c = SimpleNamespace(B=len(index))

    match (batch := carla_garage_dataset.get_batch(index)).to_dict():
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
            "meta": {"input_id": _, **meta_rest},
            **batch_rest,
        } if not any((batch_rest, data_rest, meta_rest)):
            pass

        case _:
            logger.error(msg := "invalid batch structure", batch=batch)

            raise AssertionError(msg)


def test_mimicgen_dataset(mimicgen_dataset: Dataset) -> None:
    index = [0, 2]
    c = SimpleNamespace(B=len(index))

    match (batch := mimicgen_dataset.get_batch(index)).to_dict():
        case {
            "data": {
                "obs/agentview_image": Tensor(shape=[c.B, *_]),
                "_idx_": Tensor(shape=[c.B, *_]),
                "obs/robot0_eef_pos": Tensor(shape=[c.B, *_]),
                **data_rest,
            },
            "meta": {"input_id": _, **meta_rest},
            **batch_rest,
        } if not any((batch_rest, data_rest, meta_rest)):
            pass

        case _:
            logger.error(msg := "invalid batch structure", batch=batch)

            raise AssertionError(msg)


def test_nuscenes_dataset(nuscenes_dataset: Dataset) -> None:
    index = [0, 2]
    c = SimpleNamespace(B=len(index))

    match (batch := nuscenes_dataset.get_batch(index)).to_dict():
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
            "meta": {"input_id": _, **meta_rest},
            **batch_rest,
        } if not any((batch_rest, data_rest, meta_rest)):
            pass

        case _:
            logger.error(msg := "invalid batch structure", batch=batch)

            raise AssertionError(msg)


def test_zod_dataset(zod_dataset: Dataset) -> None:
    index = [2]
    c = SimpleNamespace(B=len(index))

    match (batch := zod_dataset.get_batch(index)).to_dict():
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
            "meta": {"input_id": [*_], **meta_rest},
            **batch_rest,
        } if not any((batch_rest, data_rest, meta_rest)):
            pass

        case _:
            logger.error(msg := "invalid batch structure", batch=batch)

            raise AssertionError(msg)


@pytest.mark.parametrize(
    "dataset",
    [
        lf("carla_garage_dataset"),
        lf("mimicgen_dataset"),
        lf("nuscenes_dataset"),
        lf("yaak_dataset"),
        lf("zod_dataset"),
    ],
)
def test_save_and_load(dataset: Dataset, tmp_path: Path) -> None:
    dataset.save(tmp_path)
    assert dataset == Dataset.load(tmp_path)


@pytest.mark.parametrize(
    "dataset",
    [
        lf("carla_garage_dataset"),
        lf("mimicgen_dataset"),
        lf("nuscenes_dataset"),
        lf("yaak_dataset"),
        lf("zod_dataset"),
    ],
)
def test_pickle(dataset: Dataset) -> None:
    assert dill.pickles(dataset, exact=True, safe=True)
