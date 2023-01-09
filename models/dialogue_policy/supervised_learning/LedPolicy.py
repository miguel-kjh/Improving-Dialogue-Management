from abc import ABC

from torch import nn as nn

from models.dialogue_policy.supervised_learning.EmbeddingPolicy import EmbeddingPolicy
from utils.ted_utils import create_embedding_layer


class Led(EmbeddingPolicy, ABC):

    def __init__(self, config: dict, n_actions: int, embedding_size: int):
        super().__init__(config, n_actions, embedding_size)

        self.model = nn.LSTM(input_size=self.num_features,
                             hidden_size=self.hparams.hidden_size,
                             num_layers=self.hparams.num_layers,
                             dropout=self.hparams.dropout_dialogue,
                             batch_first=True)

        self.dense_transformer = create_embedding_layer(
            self.hparams.hidden_size,
            self.hparams.dropout_dialogue,
            self.hparams.regularization_constant
        )

    def _make_a_transformation(self, x):
        x = self.pre_dial(x)
        x, _ = self.model(x)
        return x[:, -1], None