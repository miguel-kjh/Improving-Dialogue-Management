import random

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
import hydra

from service.TrainAndEvaluateService import TrainAndEvaluateService
import warnings

warnings.filterwarnings("ignore", ".*")


def reset_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    reset_seed(cfg['resources']['seed'])
    train_and_evaluate_service = TrainAndEvaluateService(cfg, create_states=False)
    train_and_evaluate_service.process()


if __name__ == "__main__":
    main()
