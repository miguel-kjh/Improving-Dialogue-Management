from typing import List
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from models.dialogue_policy.supervised_learning.DiaSeqPolicy import DiaSeqPolicy
from models.dialogue_policy.supervised_learning.DiaMultiClassPolicy import DiaMultiClassPolicy
from models.dialogue_policy.supervised_learning.DiaMultiDensePolicy import DiaMultiDensePolicy
from models.dialogue_policy.supervised_learning.Ted import Ted
from models.dialogue_policy.supervised_learning.Red import Red
from service.Pipeline import Pipeline
import pytorch_lightning as pl


class TrainService(Pipeline):

    def __init__(self, config: dict, actions: List[int], name_experiment: str):
        super().__init__()
        self.config = config
        self.model_config = config['model']
        self.activate_wandb_logging = self.config['resources']['wandb']
        self.actions = list(range(len(actions)))
        self.name_experiment = name_experiment
        self._models = {
            "TED": Ted,
            "RED": Red,
            "MC": DiaMultiClassPolicy,
            "MD": DiaMultiDensePolicy,
            "SEQ": DiaSeqPolicy
        }

    def get_model(self, model: str, config: dict, actions: List[int]) -> pl.LightningModule:
        return self._models[model](config, actions)

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
