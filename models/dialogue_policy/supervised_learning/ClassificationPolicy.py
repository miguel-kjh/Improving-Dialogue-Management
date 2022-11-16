from abc import ABC
from typing import List, Tuple

import torch

from models.dialogue_policy.supervised_learning.Policy import Policy


class ClassificationPolicy(Policy, ABC):

    def __init__(self, config: dict, actions: List[int]) -> None:
        super(ClassificationPolicy, self).__init__(config, len(actions))
        config['a_dim'] = self.n_actions
        self.save_hyperparameters(config)
        self.net = None

    def forward(self, s, a_target_gold, s_target_pos=None):
        return self.net(s, a_target_gold, s_target_pos)

    def _update_test_log(self, s, actions, pred, log):
        for idx in range(s.size(0)):
            self.test_results['Index'] += [log[idx].item()]
            self.test_results['Inputs'] += [s[idx]]
            self.test_results['Labels'] += [actions[idx]]
            self.test_results['Predictions'] += [pred[idx]]
            self.test_results['IsCorrect'] += [all(actions[idx] == pred[idx])]

    def _transfrom_tensors_for_prediction(self, x, y) -> Tuple[torch.Tensor, torch.Tensor]:
        x_hat = x.type(torch.int64)
        y_hat = y.type(torch.int64)
        return x_hat, y_hat

    def training_step(self, batch, batch_idx):
        s, a_target_gold, s_target_pos, actions, _ = batch
        loss, pred = self(s, a_target_gold, s_target_pos)
        self.log("train_loss", loss)
        pred, actions = self._transfrom_tensors_for_prediction(pred, actions)
        self.log_metrics('train', pred, actions, multiclass=True)
        return loss

    def validation_step(self, batch, batch_idx):
        s, a_target_gold, s_target_pos, actions, _ = batch
        loss, pred = self(s, a_target_gold, s_target_pos)
        self.log("val_loss", loss)
        pred, actions = self._transfrom_tensors_for_prediction(pred, actions)
        self.log_metrics('val', pred, actions, multiclass=True)
        return loss

    def test_step(self, batch, batch_idx):
        s, a_target_gold, s_target_pos, actions, log = batch
        _, pred = self(s, a_target_gold, s_target_pos)
        pred, actions = self._transfrom_tensors_for_prediction(pred, actions)
        self.log_metrics('test', pred, actions, multiclass=True)
        self._update_test_log(s, actions, pred, log)

    def configure_optimizers(self):
        return getattr(torch.optim, self.hparams.opt)(self.parameters(), lr=self.hparams.lr)
