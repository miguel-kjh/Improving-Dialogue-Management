import torch
from torch import nn

from models.dialogue_policy.loss.InnerProductSimilarity import InnerProductSimilarity
from models.dialogue_policy.loss.MarginRankingLoss import MarginRankingLoss
from models.transformation.NegativeSampling import NegativeSampling


class StarSpace(nn.Module):
    def __init__(self, d_embed, n_input, n_output, max_norm=10, aggregate=torch.sum):
        super(StarSpace, self).__init__()

        self.n_input = n_input
        self.n_output = n_output
        self.aggregate = aggregate

        self.input_embedding = nn.Sequential(
            nn.LazyLinear(d_embed),
            nn.LayerNorm(d_embed)
        )
        self.output_embedding = nn.Sequential(
            nn.LazyLinear(d_embed),
            nn.LayerNorm(d_embed)
        )

    def forward(self, input=None, output=None):
        input_repr, output_repr = None, None

        if input is not None:
            if input.dim() == 1:
                input = input.unsqueeze(-1)

            input_emb = self.input_embedding(input)  # B x L_i x dim
            input_repr = self.aggregate(input_emb, dim=1)  # B x dim

        if output is not None:
            if output.dim() == 1:
                output = output.unsqueeze(-1)  # B x L_o

            output_emb = self.output_embedding(output)  # B x L_o x dim
            output_repr = self.aggregate(output_emb, dim=1)  # B x dim

        return input_repr, output_repr


if __name__ == '__main__':


    # transformer output
    lhs = torch.tensor(
        [
            [1, 2, 3, 4, 5],
            [1, 2, 3, 2, 3],
            [1, 2, 3, 4, 22]
        ]
        , dtype=torch.float32
    )
    rhs = torch.tensor([[0, 1, 1], [0, 0, 0], [0, 1, 0]], dtype=torch.float32)

    n_lhs = lhs.size(1)
    n_rhs = rhs.size(1)
    batch_size = 3

    model = StarSpace(2, 23, n_rhs)
    criterion = MarginRankingLoss(margin=1., aggregate=torch.mean)
    lhs_repr, pos_rhs_repr = model(lhs, rhs)
    print(lhs_repr)
    print(pos_rhs_repr)
    sim = InnerProductSimilarity()
    positive_similarity = sim(lhs_repr, pos_rhs_repr)

    n_negative = 1
    n_samples = batch_size * n_negative
    neg_sampling = NegativeSampling(n_output=batch_size, n_negative=n_negative)
    neg_rhs = neg_sampling.sample(n_samples)
    if lhs.is_cuda:
        neg_rhs = neg_rhs.cuda()
    _, neg_rhs_repr = model(output=neg_rhs)  # (B * n_negative) x dim
    neg_rhs_repr = neg_rhs_repr.view(3, n_negative, -1)  # B x n_negative x dim
    negative_similarity = sim(lhs_repr, neg_rhs_repr).squeeze(1)

    loss = criterion(positive_similarity, negative_similarity)

    # test
    test_candidate_rhs = torch.tensor([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ])
    n_hrs = test_candidate_rhs.size(0)
    test_rhs = torch.tensor([
        [1, 2, 3, 4, 5],
        [1, 2, 3, 2, 3],
        [1, 2, 3, 4, 22]
    ])
    test_lhs_repr, test_candidate_rhs_repr = model(lhs,
                                                   test_candidate_rhs.view(
                                                       batch_size * n_hrs))  # B x dim, (B * n_output) x dim
    test_candidate_rhs_repr = test_candidate_rhs_repr.view(batch_size, n_hrs, -1)  # B x n_output x dim
    similarity = sim(test_lhs_repr, test_candidate_rhs_repr).squeeze(1)  # B x n_output
    print(similarity)
    res = torch.max(similarity, dim=-1)[1].data

    print("SIM +", positive_similarity)
    print("SIM -", negative_similarity)
    print("LOSS", loss)
    print("RES", res)
