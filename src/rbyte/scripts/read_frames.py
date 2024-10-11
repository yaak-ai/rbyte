from typing import Any, cast

import hydra
import more_itertools as mit
import numpy as np
import numpy.typing as npt
import rerun as rr
from hydra.utils import instantiate
from omegaconf import DictConfig
from pydantic import TypeAdapter
from structlog import get_logger
from structlog.contextvars import bound_contextvars
from tqdm import tqdm

from rbyte.io.frame.base import FrameReader
from rbyte.viz.loggers.rerun_logger import FrameConfig

logger = get_logger(__name__)


@hydra.main(version_base=None)
def main(config: DictConfig) -> None:
    frame_reader = cast(FrameReader, instantiate(config.frame_reader))
    frame_config = cast(
        FrameConfig, TypeAdapter(FrameConfig).validate_python(config.frame_config)
    )

    rr.init(config.application_id, spawn=True)

    for frame_indexes in mit.chunked(
        tqdm(sorted(frame_reader.get_available_indexes())),
        config.batch_size,
        strict=False,
    ):
        with bound_contextvars(frame_config=frame_config):
            match frame_config:
                case {rr.Image: image_format} | {rr.DepthImage: image_format}:
                    arr = cast(
                        npt.NDArray[Any],
                        frame_reader.read(frame_indexes).cpu().numpy(),  # pyright: ignore[reportUnknownMemberType]
                    )
                    with bound_contextvars(image_format=image_format, shape=arr.shape):
                        match (
                            image_format.pixel_format,
                            image_format.color_model,
                            arr.shape,
                        ):
                            case None, rr.ColorModel(), (batch, height, width, _):
                                pass

                            case rr.PixelFormat.NV12, None, (batch, dim, width):
                                height = int(dim / 1.5)

                            case _:
                                logger.error("not implemented")

                                raise NotImplementedError

                        image_format = rr.components.ImageFormat(
                            height=height,
                            width=width,
                            pixel_format=image_format.pixel_format,
                            color_model=image_format.color_model,
                            channel_datatype=rr.ChannelDatatype.from_np_dtype(
                                arr.dtype
                            ),
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
