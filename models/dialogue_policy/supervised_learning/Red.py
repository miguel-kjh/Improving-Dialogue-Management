from abc import ABC

from torch import nn
from typing import List

from models.dialogue_policy.supervised_learning.EmbeddingPolicy import EmbeddingPolicy


class Red(EmbeddingPolicy, ABC):

    def __init__(self, config: dict, actions: List[int], embedding_size: int):
        super().__init__(config, actions, embedding_size)

        self.model = nn.LSTM(input_size=self.hparams.hidden_layers_sizes_pre_dial[-1],
                             hidden_size=self.hparams.hidden_size,
                             num_layers=self.hparams.num_layers,
                             dropout=self.hparams.dropout_dialogue,
                             batch_first=True)

    def _make_a_transformation(self, x):
        x = self.pre_dial(x)
        x, _ = self.model(x)
        return x[:, -1], None