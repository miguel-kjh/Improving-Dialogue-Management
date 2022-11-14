from typing import List

from models.dialogue_policy.supervised_learning.ClassificationPolicy import ClassificationPolicy
from models.dialogue_policy.supervised_learning.DiaMultiClass import DiaMultiClass


class DiaMultiClassPolicy(ClassificationPolicy):

    def __init__(self, config: dict, actions: List[int]) -> None:
        super().__init__(config, actions)
        self.net = DiaMultiClass(self.hparams)