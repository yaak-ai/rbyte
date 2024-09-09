from typing import cast

import hydra
import more_itertools as mit
import rerun as rr
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from structlog import get_logger
from tqdm import tqdm

from rbyte.io.frame.base import FrameReader

logger = get_logger(__name__)


@hydra.main(version_base=None)
def main(config: DictConfig) -> None:
    frame_reader = cast(FrameReader, instantiate(config.frame_reader))
    rr.init("rbyte", spawn=True)

    for frame_indexes in mit.chunked(
        tqdm(sorted(frame_reader.get_available_indexes())),
        config.batch_size,
        strict=False,
    ):
        frames = frame_reader.read(frame_indexes)
        match frames.shape, frames.dtype:
            case ((_, height, width, 3), torch.uint8):
                rr.log(
                    config.entity_path,
                    [
                        rr.components.ImageFormat(
                            height=height,
                            width=width,
                            color_model="RGB",
                            channel_datatype="U8",
                        ),
                        rr.Image.indicator(),
                    ],
                    static=True,
                    strict=True,
                )

                rr.send_columns(
                    config.entity_path,
                    times=[rr.TimeSequenceColumn("frame_index", frame_indexes)],
                    components=[
                        rr.components.ImageBufferBatch(
                            frames.flatten(start_dim=1, end_dim=-1).numpy()  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
                        )
                    ],
                    strict=True,
                )

            case _:
                raise NotImplementedError


if __name__ == "__main__":
    main()
