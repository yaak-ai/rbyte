import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(version_base=None)
def main(config: DictConfig) -> None:
    with instantiate(config.tracker):
        instantiate(config.dataset)


if __name__ == "__main__":
    main()
