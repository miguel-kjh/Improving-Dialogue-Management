from typing import List

from models.dialogue_policy.supervised_learning.ClassificationPolicy import ClassificationPolicy
from models.dialogue_policy.supervised_learning.DiaSeq import DiaSeq


class DiaSeqPolicy(ClassificationPolicy):

    def __init__(self, config: dict, actions: List[int]) -> None:
        super().__init__(config, actions)
        self.net = DiaSeq(self.hparams)

    def forward(self, s, type='train', s_target_seq=None):
        return self.net(s, type, s_target_seq)

    def training_step(self, batch, batch_idx):
        s, s_target_seq, _ = batch
        loss, pred = self(s, s_target_seq=s_target_seq)
        self.log("train_loss", loss)
        pred, s_target_seq = self._transfrom_tensors_for_prediction(pred, s_target_seq)
        self.log_metrics('train', pred, s_target_seq, multiclass=True)
        return loss

    def validation_step(self, batch, batch_idx):
        s, s_target_seq, _ = batch
        loss, pred = self(s, type='val', s_target_seq=s_target_seq)
        self.log("val_loss", loss)
        pred, s_target_seq = self._transfrom_tensors_for_prediction(pred, s_target_seq)
        self.log_metrics('val', pred, s_target_seq, multiclass=True)
        return loss

    def test_step(self, batch, batch_idx):
        s, s_target_seq, _ = batch
        _, pred = self(s, type='test', s_target_seq=s_target_seq)
        pred, a_target_gold = self._transfrom_tensors_for_prediction(pred, s_target_seq)
        self.log_metrics('test', pred, s_target_seq, multiclass=True)
