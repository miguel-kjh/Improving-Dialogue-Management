from torch import nn
from torch.nn import functional as F


class CosineSimilarity(nn.Module):

    def __init__(self):
        super(CosineSimilarity, self).__init__()

    def forward(self, a, b):
        # a => B x [n_a x] dim, b => B x [n_b x] dim

        if a.dim() == 2:
            a = a.unsqueeze(1)  # B x n_a x dim

        if b.dim() == 2:
            b = b.unsqueeze(1)  # B x n_b x dim

        return F.cosine_similarity(a, b, dim=2)  # B x n_a x n_b
