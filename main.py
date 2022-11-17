import os
import random
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
import hydra

from models.datamodule.DataModule import DataModule
from service.EvaluateService import EvaluateService
from service.InputOutput.OutputCsvService import OutputCsvService
from service.InputOutput.OutputJpgService import OutputJpgService
from service.LoadDataService import LoadDataService
from service.StateTrackerService import StateTrackerService
import pytorch_lightning as pl

import warnings

from service.TrainService import TrainService
from view.Logger import Logger

warnings.filterwarnings("ignore", ".*")


def reset_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def create_folder(path: str, delete_if_exist: bool = True):
    if os.path.exists(path):
        if delete_if_exist:
            shutil.rmtree(path)
        else:
            return

    Logger.info("Creating folder: " + path)
    os.makedirs(path)


def get_current_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


class Main:
    def __init__(self, config: DictConfig):
        self._config = config
        self._result_folder = 'results'
        self._name_dataset = f'{self._config.dataset.name}_{self._config.dataset.domain}'
        self._name_model = f'{self._config.model.name}-{get_current_time()}'
        datasets_results_path = os.path.join(self._result_folder, self._name_dataset)
        self._folder = os.path.join(datasets_results_path, self._name_model)

        create_folder(datasets_results_path, delete_if_exist=False)
        create_folder(self._folder, delete_if_exist=True)

        self.output_csv_service = OutputCsvService()
        self.output_jpg_service = OutputJpgService()

    def _load_data(self) -> pd.DataFrame:
        Logger.print_title(f"Load Data: {self._config.dataset.name}")
        load_data_service = LoadDataService(self._config)
        return load_data_service.run()

    def _state_tracker(self, pd_df: pd.DataFrame) -> DataModule:
        Logger.print_title("State Tracker")
        dst = StateTrackerService(self._config)
        data_module = dst.run(pd_df)
        return data_module

    def _train(self, data_module: DataModule) -> pl.Trainer:
        Logger.print_title("Train")
        name_experiment = os.path.basename(self._folder)
        train_service = TrainService(self._config, data_module.classes, name_experiment)
        trainer = train_service.run(data_module)
        return trainer

    def _evaluate(self, data_module: DataModule, trainer) -> dict:
        Logger.print_title("Evaluate")
        evaluate_service = EvaluateService(data_module, data_module.classes)
        test_results, test, cm = evaluate_service.run(trainer)
        return {
            'test_results': test_results,
            'test': test,
            'cm': cm
        }

    def _save_results(self, results: dict):
        Logger.print_title("Save Results")
        for name, result in results.items():
            if isinstance(result, pd.DataFrame):
                self.output_csv_service.save(
                    result,
                    os.path.join(self._folder, f'{name}.csv')
                )
            else:
                if result:
                    self.output_jpg_service.save(
                        result,
                        os.path.join(self._folder, f'{name}.jpg')
                    )

    def _save_config(self):
        OmegaConf.save(self._config, os.path.join(self._folder, 'config.yaml'))

    def run(self):
        df = self._load_data()

        data_module = self._state_tracker(df)

        trainer = self._train(data_module)
        trainer.save_checkpoint(os.path.join(self._folder, 'model.ckpt'))

        results = self._evaluate(data_module, trainer)
        self._save_results(results)

        self._save_config()


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    reset_seed(cfg['resources']['seed'])

    main_program = Main(cfg)
    main_program.run()


if __name__ == "__main__":
    main()
