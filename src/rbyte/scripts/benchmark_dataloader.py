import multiprocessing as mp

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm


@hydra.main(version_base=None)
def main(config: DictConfig) -> None:
    dataloader = instantiate(config.dataloader)
    for _ in tqdm(dataloader, disable=False):
        pass


if __name__ == "__main__":
    mp.set_forkserver_preload(["rbyte"])

    main()
