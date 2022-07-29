import math
from abc import ABC

import torch
from torch import nn as nn

from models.dialogue_policy.supervised_learning.EmbeddingPolicy import EmbeddingPolicy
from models.dialogue_policy.supervised_learning.PositionalEncoding import PositionalEncoding


def get_tgt_mask(size) -> torch.tensor:
    # Generates a squeare matrix where the each row allows one word more to be seen
    mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
    mask = mask.float()
    mask = mask.masked_fill(mask == 0, float('-inf'))  # Convert zeros to -inf
    mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0

    # EX for size=5:
    # [[0., -inf, -inf, -inf, -inf],
    #  [0.,   0., -inf, -inf, -inf],
    #  [0.,   0.,   0., -inf, -inf],
    #  [0.,   0.,   0.,   0., -inf],
    #  [0.,   0.,   0.,   0.,   0.]]

    return mask


class Ted(EmbeddingPolicy, ABC):

    def __init__(self, config: dict, n_actions: int):
        super().__init__(config, n_actions)

        self.pos_encoder = PositionalEncoding(
            self.hparams.encoding_dimension,
            dropout=self.hparams.dropout_attention
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hparams.encoding_dimension,
            nhead=self.hparams.heads,
            dropout=self.hparams.dropout_attention,
            dim_feedforward=self.hparams.transformer_size,
            activation=self.hparams.activation
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.hparams.num_layers,
            norm=nn.LayerNorm(self.hparams.encoding_dimension, eps=self.hparams.regularization_constant)
        )

    def _make_a_transformation(self, x):
        mask = get_tgt_mask(x.size(0)).to(self.device) if self.hparams.unidirectional_encoder else None
        x = self.pre_dial(x)
        x = x * math.sqrt(self.num_features)
        x = self.pos_encoder(x)
        x = self.transformer(x, mask=mask)
        x = x.mean(dim=1)
        return x, mask