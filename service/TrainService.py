from typing import List
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from models.dialogue_policy.supervised_learning.DiaPolicy import DiaPolicy
from models.dialogue_policy.supervised_learning.Ted import Ted
from models.dialogue_policy.supervised_learning.Red import Red
from service.Pipeline import Pipeline
import pytorch_lightning as pl


class TrainService(Pipeline):

    def __init__(self, config: dict, actions: List[int]):
        super().__init__()
        self.config = config
        self.model_config = config['model']
        self.activate_wandb_logging = self.config['resources']['wandb']
        self.actions = list(range(len(actions)))
        self.name = self.config['dataset']['name']
        self.name_experiment = "dummy"  # TODO: change name to something more meaningful

    @staticmethod
    def get_model(model: str, config: dict, actions: List[int]) -> pl.LightningModule:
        models = {
            "SS": Ted,
            "RED": Red,
            "DIA": DiaPolicy
        }
        return models[model](config, actions)

    @staticmethod
    def __get_callbacks() -> list:
        callbacks = [
            EarlyStopping(
                monitor='val_f1',
                verbose=True,
                mode='max'
            )
        ]

        return callbacks

    def _fit(self, model: pl.LightningModule, data: pl.LightningDataModule) -> pl.Trainer:
        wandb_logger = WandbLogger(project=self.name, name=self.name_experiment) \
            if self.activate_wandb_logging else None

        trainer = pl.Trainer(
            gpus=self.config['resources']['gpus'],
            logger=wandb_logger,
            max_epochs=self.config['model']['epochs'],
            # callbacks=self.__get_callbacks()
        )

        trainer.fit(model, data)

        return trainer

    def run(self, data: object = None) -> object:
        assert isinstance(data, pl.LightningDataModule), "Data must be of type LightningDataModule"
        model = self.get_model(self.model_config['name'], self.model_config, self.actions)
        trainer = self._fit(model, data)
        return trainer
