import torch
from torch import nn

from models.dialogue_policy.loss.InnerProductSimilarity import InnerProductSimilarity
from models.dialogue_policy.loss.MarginRankingLoss import MarginRankingLoss
from models.transformation.NegativeSampling import NegativeSampling


class StarSpace(nn.Module):
    def __init__(self, d_embed, aggregate=torch.sum):
        super(StarSpace, self).__init__()

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

            input_repr = self.input_embedding(input)  # B x L_i x dim
            # input_repr = self.aggregate(input_emb, dim=1)  # B x dim

        if output is not None:
            if output.dim() == 1:
                output = output.unsqueeze(-1)  # B x L_o

            output_repr = self.output_embedding(output)  # B x L_o x dim
            # output_repr = self.aggregate(output_emb, dim=1)  # B x dim

        return input_repr, output_repr


if __name__ == '__main__':
    # transformer output
    lhs = torch.tensor(
        [
            [1, 2, 3, 4, 5],
            [1, 2, 3, 2, 3],
            [1, 2, 3, 4, 22],
            [1, 3, -4, 5, 6]
        ]
        , dtype=torch.float32
    )
    rhs = torch.tensor([
        [0, 1, 1],
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 0]
    ],
        dtype=torch.float32
    )

    n_lhs = lhs.size(1)
    n_rhs = rhs.size(1)
    print("lhs:", lhs.size())
    print("rhs:", rhs.size())
    d_emb = 20
    batch_size = 4

    model = StarSpace(d_emb)
    criterion = MarginRankingLoss(margin=1., aggregate=torch.mean)
    lhs_repr, pos_rhs_repr = model(lhs, rhs)
    sim = InnerProductSimilarity()
    positive_similarity = sim(lhs_repr, pos_rhs_repr)

    n_negative = 1
    n_samples = batch_size * n_negative
    neg_sampling = NegativeSampling(n_output=batch_size, n_negative=n_negative)
    neg_rhs = neg_sampling.sample(n_samples)
    # get negative samples of rhs
    neg_rhs_emb = torch.index_select(rhs, 0, neg_rhs)
    _, neg_rhs_repr = model(output=neg_rhs_emb)  # (B * n_negative) x dim
    neg_rhs_repr = neg_rhs_repr.view(batch_size, n_negative, -1)  # B x n_negative x dim
    negative_similarity = sim(lhs_repr, neg_rhs_repr).squeeze(1)

    loss = criterion(positive_similarity, negative_similarity)

    print("SIM +", positive_similarity)
    print("SIM -", negative_similarity)
    print("LOSS", loss)

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
    ], dtype=torch.float32)
    n_hrs = test_candidate_rhs.size(0)
    test_rhs = torch.tensor([
        [1, 2, 3, 4, 5],
        [1, 2, 3, 2, 3],
        [1, 2, 3, 4, 22]
    ], dtype=torch.float32)
    test_lhs_repr, test_candidate_rhs_repr = model(
        test_rhs,
        test_candidate_rhs
    )
    # B x n_output x dim
    print("test_candidate_rhs_repr", test_candidate_rhs_repr.size())
    # test_candidate_rhs_repr = test_candidate_rhs_repr.view(batch_size, n_hrs, 20)
    test_candidate_rhs_repr = test_candidate_rhs_repr.T
    test_candidate_rhs_repr = test_candidate_rhs_repr.unsqueeze(2)
    test_candidate_rhs_repr = test_candidate_rhs_repr.expand(d_emb, n_hrs, batch_size)
    test_candidate_rhs_repr = test_candidate_rhs_repr.T
    print("test_candidate_rhs_repr", test_candidate_rhs_repr.size())
    similarity = sim(test_lhs_repr, test_candidate_rhs_repr).squeeze(1)  # B x n_output
    print(similarity)
    res = torch.max(similarity, dim=-1)[1].data
    print("RES", res)
    print("RES", test_candidate_rhs.index_select(0, res))
