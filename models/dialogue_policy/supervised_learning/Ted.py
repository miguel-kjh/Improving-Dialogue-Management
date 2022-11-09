import math
from abc import ABC
from typing import List

from torch import nn

from models.dialogue_policy.supervised_learning.EmbeddingPolicy import EmbeddingPolicy
from models.dialogue_policy.supervised_learning.PositionalEncoding import PositionalEncoding
from models.dialogue_policy.supervised_learning.TedPolicy import get_tgt_mask
from utils.ted_utils import create_ffn_layer


class Ted(EmbeddingPolicy, ABC):

    def __init__(self, config: dict, actions: List[int]):
        super().__init__(config, actions)

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
            norm=nn.LayerNorm(
                self.hparams.encoding_dimension,
                eps=self.hparams.regularization_constant
            )
        )

    def _make_a_transformation(self, x):
        mask = get_tgt_mask(x.size(0)).to(self.device) \
            if self.hparams.unidirectional_encoder \
            else None
        x = self.pre_dial(x)
        x = x * math.sqrt(x.size(0))
        x = self.pos_encoder(x)
        x = self.transformer(x, mask=mask)
        x = x.mean(dim=1)
        return x, mask
