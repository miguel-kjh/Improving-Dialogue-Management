from typing import List

from models.dialogue_policy.supervised_learning.ClassificationPolicy import ClassificationPolicy
from models.dialogue_policy.supervised_learning.DiaSeq import DiaSeq


class DiaSeqPolicy(ClassificationPolicy):

    def __init__(self, config: dict, actions: List[int], embedding_size) -> None:
        config['embed_size'] = embedding_size
        config['max_len'] = len(actions)
        super().__init__(config, actions, embedding_size)
        self.net = DiaSeq(self.hparams)

    def forward(self, s, type='train', s_target_seq=None):
        return self.net(s, type, s_target_seq)

    def training_step(self, batch, batch_idx):
        s, s_target_seq, _, actions, _ = batch
        loss, pred = self(s, s_target_seq=s_target_seq)
        self.log("train_loss", loss)
        pred, s_target_seq = self._transfrom_tensors_for_prediction(pred, actions)
        self.log_metrics('train', pred, actions, multiclass=True)
        return loss

    def validation_step(self, batch, batch_idx):
        s, s_target_seq, _, actions, _ = batch
        loss, pred = self(s, type='val', s_target_seq=s_target_seq)
        self.log("val_loss", loss)
        pred, s_target_seq = self._transfrom_tensors_for_prediction(pred, actions)
        self.log_metrics('val', pred, actions, multiclass=True)
        return loss

    def test_step(self, batch, batch_idx):
        s, s_target_seq, _, actions, log = batch
        _, pred = self(s, type='test', s_target_seq=s_target_seq)
        pred, a_target_gold = self._transfrom_tensors_for_prediction(pred, actions)
        self.log_metrics('test', pred, actions, multiclass=True)
        self._update_test_log(s, actions, pred, log)
