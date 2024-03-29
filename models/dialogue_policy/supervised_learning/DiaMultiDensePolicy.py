from typing import List

from models.dialogue_policy.supervised_learning.ClassificationPolicy import ClassificationPolicy
from models.dialogue_policy.supervised_learning.DiaMultiDense import DiaMultiDense


class DiaMultiDensePolicy(ClassificationPolicy):

    def __init__(self, config: dict, actions: List[int], embedding_size) -> None:
        super().__init__(config, actions, embedding_size)
        self.net = DiaMultiDense(self.hparams)
