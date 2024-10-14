from typing import Any, cast

import hydra
import more_itertools as mit
import numpy as np
import numpy.typing as npt
import rerun as rr
from omegaconf import DictConfig, OmegaConf
from pydantic import ConfigDict, NonNegativeInt
from structlog import get_logger
from structlog.contextvars import bound_contextvars
from tqdm import tqdm

from rbyte.config.base import BaseModel, HydraConfig
from rbyte.io.frame.base import FrameReader
from rbyte.viz.loggers.rerun_logger import FrameConfig

logger = get_logger(__name__)


class Config(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    frame_reader: HydraConfig[FrameReader]
    frame_config: FrameConfig
    application_id: str
    entity_path: str
    batch_size: NonNegativeInt = 1


@hydra.main(version_base=None)
def main(_config: DictConfig) -> None:
    config = Config.model_validate(OmegaConf.to_object(_config))
    frame_reader = config.frame_reader.instantiate()
    frame_config = config.frame_config

    rr.init(config.application_id, spawn=True)

    for frame_indexes in mit.chunked(
        tqdm(sorted(frame_reader.get_available_indexes())),
        config.batch_size,
        strict=False,
    ):
        tensor = frame_reader.read(frame_indexes)

        with bound_contextvars(frame_config=frame_config, shape=tensor.shape):
            match frame_config:
                case {rr.Image: image_format} | {rr.DepthImage: image_format}:
                    match (
                        image_format.pixel_format,
                        image_format.color_model,
                        tensor.shape,
                    ):
                        case None, color_model, shape:
                            match color_model, shape:
                                case (
                                    (rr.ColorModel.L, (batch, height, width, 1))
                                    | (rr.ColorModel.RGB, (batch, height, width, 3))
                                    | (rr.ColorModel.RGBA, (batch, height, width, 4))
                                ):
                                    pass

                                case (
                                    (rr.ColorModel.L, (batch, 1, height, width))
                                    | (rr.ColorModel.RGB, (batch, 3, height, width))
                                    | (rr.ColorModel.RGBA, (batch, 4, height, width))
                                ):
                                    tensor = tensor.permute(0, 2, 3, 1)

                                case _:
                                    logger.error("not implemented")

                                    raise NotImplementedError

                        case rr.PixelFormat.NV12, _, (batch, dim, width):
                            height = int(dim / 1.5)

                        case _:
                            logger.error("not implemented")

                            raise NotImplementedError

                    arr = cast(npt.NDArray[Any], tensor.cpu().numpy())  # pyright: ignore[reportUnknownMemberType]
                    image_format = rr.components.ImageFormat(
                        height=height,
                        width=width,
                        pixel_format=image_format.pixel_format,
                        color_model=image_format.color_model,
                        channel_datatype=rr.ChannelDatatype.from_np_dtype(arr.dtype),
                    )

                    components = [
                        mit.one(frame_config).indicator(),
                        rr.components.ImageFormatBatch([image_format] * batch),
                        rr.components.ImageBufferBatch(
                            arr.reshape(batch, -1).view(np.uint8)
                        ),
                    ]

                case _:
                    logger.error("not implemented")

                    raise NotImplementedError

        times = [rr.TimeSequenceColumn("frame_index", frame_indexes)]

        rr.send_columns(
            entity_path=config.entity_path,
            times=times,
            components=components,  # pyright: ignore[reportArgumentType]
            strict=True,
        )


if __name__ == "__main__":
    main()
