from abc import ABC
from typing import List

import torch

from models.dialogue_policy.supervised_learning.DiaMultiClass import DiaMultiClass
from models.dialogue_policy.supervised_learning.Policy import Policy


class DiaPolicy(Policy, ABC):

    def __init__(self, config: dict, actions: List[int]) -> None:
        super(DiaPolicy, self).__init__(config, len(actions))
        config['a_dim'] = self.n_actions
        self.save_hyperparameters(config)
        # TODO: add all models for dia.yaml: DiaMultiClass.py, DiaMultiDense.py, seq.py
        self.net = DiaMultiClass(self.hparams)

    def forward(self, s, a_target_gold, s_target_pos=None):
        return self.net(s, a_target_gold, s_target_pos)

    def training_step(self, batch, batch_idx):
        s, a_target_gold, s_target_pos = batch
        loss, pred = self(s, a_target_gold, s_target_pos)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        s, a_target_gold, s_target_pos = batch
        loss, _ = self(s, a_target_gold, s_target_pos)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        s, a_target_gold, s_target_pos = batch
        loss, _ = self(s, a_target_gold, s_target_pos)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return getattr(torch.optim, self.hparams.opt)(self.parameters(), lr=self.hparams.lr)
