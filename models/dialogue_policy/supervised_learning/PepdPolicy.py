from typing import List

from models.dialogue_policy.supervised_learning.ClassificationPolicy import ClassificationPolicy
from models.dialogue_policy.supervised_learning.Pepd import Pepd


class PedpPolicy(ClassificationPolicy):
    def __init__(self, config: dict, actions: List[int], embedding_size) -> None:
        config['s_dim'] = embedding_size
        config['embed_size'] = embedding_size
        config['max_len'] = len(actions)
        super().__init__(config, actions, embedding_size)
        self.net = Pepd(self.hparams)

    def training_step(self, batch, batch_idx):
        state, next_state, s_pos, a_target_gold, last_pos, a_target_full, actions, _ = batch
        loss, pred = self.net(
            state,
            a_target_gold,
            self.hparams.beta,
            next_state,
            s_pos,
            a_target_full,
            last_pos
        )
        self.log("train_loss", loss)
        pred, s_target_seq = self._transfrom_tensors_for_prediction(pred, actions)
        self.log_metrics('train', pred, actions, multiclass=True)
        return loss

    def validation_step(self, batch, batch_idx):
        state, next_state, s_pos, a_target_gold, last_pos, a_target_full, actions, _ = batch
        loss, pred = self.net(
            state,
            a_target_gold,
            self.hparams.beta,
            next_state,
            s_pos,
            a_target_full,
            last_pos
        )
        self.log("val_loss", loss)
        pred, s_target_seq = self._transfrom_tensors_for_prediction(pred, actions)
        self.log_metrics('val', pred, actions, multiclass=True)
        return loss

    def test_step(self, batch, batch_idx):
        state, next_state, s_pos, a_target_gold, last_pos, a_target_full, actions, idx = batch
        _, pred = self.net(
            state,
            a_target_gold,
            self.hparams.beta,
            next_state,
            s_pos,
            a_target_full,
            last_pos
        )
        pred, a_target_gold = self._transfrom_tensors_for_prediction(pred, actions)
        self.log_metrics('test', pred, actions, multiclass=True)
        self._update_test_log(state, actions, pred, idx)
