import abc
import math
from copy import copy
from typing import Optional, List, Dict
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

from models.dialogue_policy.loss.CosineSimilarity import CosineSimilarity
from models.dialogue_policy.loss.InnerProductSimilarity import InnerProductSimilarity
from models.dialogue_policy.loss.MarginRankingLoss import MarginRankingLoss
from models.dialogue_policy.loss.ScaleCosineLoss import ScaleCosineLoss
from models.dialogue_policy.supervised_learning.Policy import Policy
from models.dialogue_policy.supervised_learning.PositionalEncoding import PositionalEncoding
from models.dialogue_policy.supervised_learning.TedPolicy import get_tgt_mask
from models.transformation.NegativeSampling import NegativeSampling
from utils.ted_utils import get_metrics, create_ffn_layer, create_embedding_layer
from models.dialogue_policy.supervised_learning.StarSpace import StarSpace


class EmbeddingPolicy(Policy):

    def __init__(self, config: dict, actions: List[int]):
        super(EmbeddingPolicy, self).__init__(config, len(actions))

        self.actions_one_hot = F.one_hot(torch.tensor(actions, device=self.device), self.n_actions).float()

        self.pre_dial = create_ffn_layer(
            self.hparams.hidden_layers_sizes_pre_dial,
            self.hparams.regularization_constant
        )

        self.dense_transformer = create_embedding_layer(
            self.hparams.embedding_space,
            self.hparams.dropout_dialogue,
            self.hparams.regularization_constant
        )

        self.start_space = StarSpace(self.hparams.embedding_space)

        self.similarity = CosineSimilarity() if self.hparams.similarity == 'cosine' else InnerProductSimilarity()

        self.criterion = ScaleCosineLoss() if self.hparams.loss == 'dot' else MarginRankingLoss()

        self.test_results = {
            'Index': [],
            'Inputs': [],
            'Embeddings': [],
            'Labels': [],
            'Predictions': [],
            'IsCorrect': [],
            'Ranking': [],
        }

        self.actions_results = {
            'Actions': actions,
            'Inputs': copy(self.actions_one_hot.tolist()),
            'Embeddings': [],
        }

    def get_test_results(self) -> Dict[str, List]:

        if len(self.test_results['Vectors']) == 0:
            return {}

        return self.test_results

    @abc.abstractmethod
    def _make_a_transformation(self, x):
        pass

    def forward(self, x, y):
        x, mask = self._make_a_transformation(x)
        x, y = self.start_space(x, y)
        return x, y, mask

    def _train_prediction(self, batch):
        x, y, _, _ = batch
        batch_size = x.size(0)
        y = y.type(torch.int64)
        y = F.one_hot(y, self.n_actions).float()
        x_repr, y_repr, _ = self(x, y)
        positive_similarity = self.similarity(x_repr, y_repr)

        n_samples = batch_size * self.hparams.num_neg
        neg_sampling = NegativeSampling(
            n_output=self.n_actions,
            n_negative=self.hparams.num_neg
        )
        neg_y = neg_sampling.sample(n_samples)
        neg_y = neg_y.to(self.device)
        neg_y_emb = F.one_hot(neg_y, self.n_actions).float()
        _, neg_y_repr = self.start_space(output=neg_y_emb)  # (B * n_negative) x dim
        neg_y_repr = neg_y_repr.view(batch_size, self.hparams.num_neg, -1)  # B x n_negative x dim
        negative_similarity = self.similarity(x_repr, neg_y_repr).squeeze(1)

        return self.criterion(positive_similarity, negative_similarity)

    def training_step(self, batch, batch_idx):
        loss = self._train_prediction(batch)
        y_hat, _ = self.predict_step(batch, batch_idx)
        _, y, y_set, _ = batch
        self.log('loss', loss, prog_bar=True)
        self.log_metrics('train', y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._train_prediction(batch)
        y_hat, _ = self.predict_step(batch, batch_idx)
        _, y, y_set, _ = batch
        self.log('val_loss', loss, prog_bar=True)
        self.log_metrics('val', y_hat, y)

    def test_step(self, batch, batch_idx):

        y_hat, ranking = self.predict_step(batch, batch_idx, is_test=True)
        _, y, y_set, _ = batch

        self.log_metrics('test', y_hat, y)

        for index in range(0, len(y)):
            self.test_results['Inputs'].append(batch[0][index].cpu().numpy().tolist())
            self.test_results['Labels'].append(batch[1][index].cpu().numpy().tolist())
            self.test_results['Predictions'].append(y_hat[index].cpu().numpy().tolist())
            self.test_results['IsCorrect'].append(y_hat[index].cpu().numpy() == y[index].cpu().numpy())
            self.test_results['Ranking'].append(ranking[index].cpu().numpy().tolist())

    def predict_step(self, batch, batch_idx: int, dataloader_idx: Optional[int] = None, is_test: bool = False):
        x, _, _, idx = batch
        batch_size = x.size(0)
        x_repr, y_repr, _ = self(x, self.actions_one_hot.to(self.device))
        n_actions = y_repr.size(0)

        y_repr = y_repr.T
        y_repr = y_repr.unsqueeze(2)
        y_repr = y_repr.expand(
            self.hparams.embedding_space,
            n_actions,
            batch_size
        )
        y_repr = y_repr.T

        similarity = self.similarity(x_repr, y_repr).squeeze(1)  # B x n_output
        res = torch.max(similarity, dim=-1)[1].data
        ranking = torch.argsort(similarity, dim=-1, descending=True)
        if is_test:
            for index in range(0, len(x)):
                self.test_results['Index'].append(idx[index].item())
                self.test_results['Embeddings'].append(x_repr[index].cpu().numpy().tolist())

            if not self.actions_results['Embeddings']:
                self.actions_results['Embeddings'] = y_repr.cpu().numpy().tolist()[0]
        return res, ranking

    def configure_optimizers(self):
        return getattr(torch.optim, self.hparams.opt)(self.parameters(), lr=self.hparams.lr)
