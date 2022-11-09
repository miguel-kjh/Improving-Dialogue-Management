import random

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
import hydra

from service.LoadDataService import LoadDataService
from service.StateTrackerService import StateTrackerService
from service.TrainAndEvaluateService import TrainAndEvaluateService
from service.MetricService import MetricService

import warnings

from service.TrainService import TrainService
from view.Logger import Logger

warnings.filterwarnings("ignore", ".*")


def reset_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class Main:
    def __init__(self, config: DictConfig):
        self.config = config

    def run(self):
        pass


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    reset_seed(cfg['resources']['seed'])

    Logger.print_title("Load Data")
    load_data_service = LoadDataService(cfg)
    df = load_data_service.run()
    actions = list(set(np.hstack(df[cfg['state']['action']])))

    Logger.print_title("State Tracker")
    dst = StateTrackerService(cfg)
    data_module = dst.run(df)

    Logger.print_title("Train and Evaluate")
    train_service = TrainService(cfg, actions)
    trainer = train_service.run(data_module)

    """train_and_evaluate_service = TrainAndEvaluateService(cfg)
    train_and_evaluate_service.run()

    metrics_service = MetricService(train_and_evaluate_service.get_path_results())
    metrics_service.run()"""


if __name__ == "__main__":
    main()
