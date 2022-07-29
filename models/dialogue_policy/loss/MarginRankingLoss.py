import torch
from torch import nn


class MarginRankingLoss(nn.Module):
    """
        This class implements the Margin Ranking Loss function.
        Adapted from: https://github.com/ChristophAlt/pytorch-starspace
    """

    def __init__(self, margin=1., aggregate=torch.mean):
        super(MarginRankingLoss, self).__init__()
        self.margin = margin
        self.aggregate = aggregate

    def forward(self, positive_similarity, negative_similarity):
        return self.aggregate(
            torch.clamp(
                self.margin - positive_similarity + negative_similarity,
                min=0
            )
        )
