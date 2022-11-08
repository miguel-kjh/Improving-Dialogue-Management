from typing import Any

import pytorch_lightning as pl
import torch
from torch import nn

from models.dialogue_policy.supervised_learning.DiaMultiClass import DiaMultiClass


class DiaPolicy(pl.LightningModule):

    def __init__(self, config: dict) -> None:
        super(DiaPolicy, self).__init__()
        self.save_hyperparameters(config)
        # TODO: add all models for dia.yaml: DiaMultiClass.py, DiaMultiDense.py, seq.py
        self.net = DiaMultiClass(self.hparams)

    def forward(self, s, a_target_gold, s_target_pos=None):
        return self.net(s, a_target_gold, s_target_pos)

    def training_step(self, batch, batch_idx):
        s, a_target_gold, s_target_pos = batch
        loss, _ = self.net(s, a_target_gold, s_target_pos)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        s, a_target_gold, s_target_pos = batch
        loss, _ = self.net(s, a_target_gold, s_target_pos)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        s, a_target_gold, s_target_pos = batch
        loss, _ = self.net(s, a_target_gold, s_target_pos)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return getattr(torch.optim, self.hparams.opt)(self.parameters(), lr=self.hparams.lr)
