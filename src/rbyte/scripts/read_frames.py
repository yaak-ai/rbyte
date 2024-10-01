from collections.abc import Mapping
from typing import Annotated

import hydra
import more_itertools as mit
import rerun as rr
import torch
from omegaconf import DictConfig, OmegaConf
from pydantic import BeforeValidator, NonNegativeInt
from structlog import get_logger
from tqdm import tqdm

from rbyte.config import BaseModel, HydraConfig
from rbyte.io.frame.base import FrameReader

logger = get_logger(__name__)


TORCH_TO_RERUN_DTYPE: Mapping[torch.dtype, rr.ChannelDatatype] = {
    torch.uint8: rr.ChannelDatatype.U8,
    torch.uint16: rr.ChannelDatatype.U16,
    torch.uint32: rr.ChannelDatatype.U32,
    torch.uint64: rr.ChannelDatatype.U64,
    torch.float16: rr.ChannelDatatype.F16,
    torch.float32: rr.ChannelDatatype.F32,
    torch.float64: rr.ChannelDatatype.F64,
}


class Config(BaseModel):
    frame_reader: HydraConfig[FrameReader]
    batch_size: NonNegativeInt = 1
    application_id: str = "rbyte-read-frames"
    entity_path: str = "frames"
    pixel_format: (
        Annotated[rr.PixelFormat, BeforeValidator(rr.PixelFormat.auto)] | None
    ) = None
    color_model: (
        Annotated[rr.ColorModel, BeforeValidator(rr.ColorModel.auto)] | None
    ) = None


@hydra.main(version_base=None)
def main(_config: DictConfig) -> None:
    config = Config.model_validate(OmegaConf.to_object(_config))

    frame_reader = config.frame_reader.instantiate()

    rr.init(config.application_id, spawn=True)
    rr.log(config.entity_path, [rr.Image.indicator()], static=True, strict=True)

    for frame_indexes in mit.chunked(
        tqdm(sorted(frame_reader.get_available_indexes())),
        config.batch_size,
        strict=False,
    ):
        frames = frame_reader.read(frame_indexes)

        match (config.pixel_format, config.color_model, frames.shape):
            case None, rr.ColorModel.RGB, (_, height, width, 3) | (_, 3, height, width):
                image_format = rr.components.ImageFormat(
                    width=width,
                    height=height,
                    color_model=rr.ColorModel.RGB,
                    channel_datatype=TORCH_TO_RERUN_DTYPE[frames.dtype],
                )

            case rr.PixelFormat.NV12, None, (_, dim, width):
                image_format = rr.components.ImageFormat(
                    width=width, height=int(dim / 1.5), pixel_format=rr.PixelFormat.NV12
                )

            case _:
                raise NotImplementedError

        rr.send_columns(
            config.entity_path,
            times=[rr.TimeSequenceColumn("frame_index", frame_indexes)],
            components=[
                rr.components.ImageFormatBatch([image_format] * len(frame_indexes)),
                rr.components.ImageBufferBatch(
                    frames.flatten(1, -1).cpu().numpy()  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
                ),
            ],
        )


if __name__ == "__main__":
    main()
