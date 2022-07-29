import torch
from torch import nn


class InnerProductSimilarity(nn.Module):
    """
    This class implements the Inner Product Similarity function.
    Adapted from: https://github.com/ChristophAlt/pytorch-starspace
    """
    def __init__(self):
        super(InnerProductSimilarity, self).__init__()

    def forward(self, a, b):
        # a => B x [n_a x] dim, b => B x [n_b x] dim

        if a.dim() == 2:
            a = a.unsqueeze(1)  # B x n_a x dim

        if b.dim() == 2:
            b = b.unsqueeze(1)  # B x n_b x dim

        return torch.bmm(a, b.transpose(2, 1))  # B x n_a x n_b