import torch
from torch import nn
from torch.nn import functional as F


class ScaleCosineLoss(nn.Module):
    """
        This class implements the Dot Product Similarity function.
    """

    def __init__(self, aggregate=torch.mean):
        super(ScaleCosineLoss, self).__init__()
        self.aggregate = aggregate

    def forward(self, positive_similarity, negative_similarity):

        softmax_logits = torch.concat(
            [positive_similarity, negative_similarity],
            dim=-1
        )
        # create label_ids for softmax
        softmax_label_ids = torch.zeros_like(softmax_logits[..., 0]).type(torch.LongTensor)
        softmax_loss = F.cross_entropy(
            softmax_logits, softmax_label_ids
        )
        return self.aggregate(softmax_loss)