from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize
from hydra.utils import instantiate
from structlog import get_logger
from torch import Tensor

logger = get_logger(__name__)

CONFIG_PATH = "../config"
DATA_DIR = Path(__file__).resolve().parent / "data"


def test_yaak() -> None:
    with initialize(version_base=None, config_path=CONFIG_PATH):
        cfg = compose(
            "visualize", overrides=["dataset=yaak", f"+data_dir={DATA_DIR}/yaak"]
        )

    dataset = instantiate(cfg.dataset)
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
                "meta/VehicleMotion/gear": Tensor(shape=[c.B, *_]),
                "meta/VehicleMotion/speed": Tensor(shape=[c.B, *_]),
                "mcap//ai/safety_score/clip.end_timestamp": Tensor(shape=[c.B, *_]),
                "mcap//ai/safety_score/score": Tensor(shape=[c.B, *_]),
                **data_rest,
            },
            "meta": {"input_id": [*_], "sample_idx": Tensor(shape=[c.B]), **meta_rest},
            **batch_rest,
        } if not any((batch_rest, data_rest, meta_rest)):
            pass

        case _:
            logger.error(msg := "invalid batch structure", batch=batch)

            raise AssertionError(msg)

    match (
        batch := dataset.get_batch(
            index, keys={"data", ("data", "meta/VehicleMotion/speed"), "meta"}
        )
    ).to_dict():
        case {
            "data": {"meta/VehicleMotion/speed": Tensor(shape=[c.B, *_]), **data_rest},
            "meta": {"input_id": [*_], "sample_idx": Tensor(shape=[c.B]), **meta_rest},
            **batch_rest,
        } if not any((batch_rest, data_rest, meta_rest)):
            pass

        case _:
            logger.error(msg := "invalid batch structure", batch=batch)

            raise AssertionError(msg)

    match (
        batch := dataset.get_batch(index, keys={("data", "meta/VehicleMotion/speed")})
    ).to_dict():
        case {
            "data": {"meta/VehicleMotion/speed": Tensor(shape=[c.B, *_]), **data_rest},
            "meta": None,
            **batch_rest,
        } if not any((batch_rest, data_rest)):
            pass

        case _:
            logger.error(msg := "invalid batch structure", batch=batch)

            raise AssertionError(msg)

    match (batch := dataset.get_batch(index, keys={("meta", "input_id")})).to_dict():
        case {"data": None, "meta": {"input_id": [*_]}, **batch_rest} if not batch_rest:
            pass

        case _:
            logger.error(msg := "invalid batch structure", batch=batch)

            raise AssertionError(msg)
