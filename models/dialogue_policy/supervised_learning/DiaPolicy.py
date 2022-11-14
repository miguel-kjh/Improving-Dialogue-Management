from abc import ABC
from typing import List, Tuple

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

    def _transfrom_tensors_for_prediction(self, x, y) -> Tuple[torch.Tensor, torch.Tensor]:
        index_tensor = torch.arange(0, x.shape[1])
        index_tensor = index_tensor.repeat(x.shape[0], 1)
        pred = torch.where(x, index_tensor, torch.zeros_like(x))
        x_hat = torch.sort(y, dim=1)[0]
        y_hat = torch.sort(pred, dim=1)[0]
        return x_hat, y_hat

    def training_step(self, batch, batch_idx):
        s, a_target_gold, s_target_pos = batch
        loss, pred = self(s, a_target_gold, s_target_pos)
        self.log("train_loss", loss)
        pred, a_target_gold = self._transfrom_tensors_for_prediction(pred, a_target_gold)
        self.log_metrics('train', pred, a_target_gold, multiclass=True)
        return loss

    def validation_step(self, batch, batch_idx):
        s, a_target_gold, s_target_pos = batch
        loss, pred = self(s, a_target_gold, s_target_pos)
        self.log("val_loss", loss)
        pred, a_target_gold = self._transfrom_tensors_for_prediction(pred, a_target_gold)
        self.log_metrics('val', pred, a_target_gold, multiclass=True)
        return loss

    def test_step(self, batch, batch_idx):
        s, a_target_gold, s_target_pos = batch
        loss, pred = self(s, a_target_gold, s_target_pos)
        self.log("test_loss", loss)
        pred, a_target_gold = self._transfrom_tensors_for_prediction(pred, a_target_gold)
        self.log_metrics('test', pred, a_target_gold, multiclass=True)
        return loss

    def configure_optimizers(self):
        return getattr(torch.optim, self.hparams.opt)(self.parameters(), lr=self.hparams.lr)
