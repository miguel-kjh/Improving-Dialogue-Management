from typing import List

from models.dialogue_policy.supervised_learning.ClassificationPolicy import ClassificationPolicy
from models.dialogue_policy.supervised_learning.Pepd import Pepd


class PedpPolicy(ClassificationPolicy):
    def __init__(self, config: dict, actions: List[int], embedding_size) -> None:
        config['embed_size'] = embedding_size
        config['max_len'] = len(actions)
        super().__init__(config, actions, embedding_size)
        self.net = Pepd(self.hparams)
