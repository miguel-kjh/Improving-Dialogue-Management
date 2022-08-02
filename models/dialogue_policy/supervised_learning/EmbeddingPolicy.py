import abc
from copy import copy
from typing import Optional, List, Dict
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn

from utils.ted_utils import get_metrics, create_ffn_layer, create_embedding_layer
from models.dialogue_policy.loss.DotProductLoss import SingleLabelDotProductLoss


class EmbeddingPolicy(pl.LightningModule):

    def __init__(self, config: dict, n_actions: int):
        super(EmbeddingPolicy, self).__init__()
        self.save_hyperparameters(config)

        self.num_features = self.hparams.hidden_layers_sizes_pre_dial[-1][-1] \
            if self.hparams.hidden_layers_sizes_pre_dial else self.hparams.n_features

        self.pre_dial = create_ffn_layer(
            self.hparams.hidden_layers_sizes_pre_dial,
            self.hparams.regularization_constant
        )

        self.dense_transformer = create_embedding_layer(
            self.num_features,
            self.hparams.embedding_space,
            self.hparams.dropout_dialogue,
            self.hparams.regularization_constant
        )

        self.dense_action = nn.Sequential(
            create_ffn_layer(self.hparams.hidden_layers_sizes_bot, self.hparams.regularization_constant),
            create_embedding_layer(n_actions, self.hparams.embedding_space,
                                   self.hparams.dropout_action, self.hparams.regularization_constant)
        )

        self.similarity_function = SingleLabelDotProductLoss(self.hparams.num_neg, self.device)

        def get_one_hot_action(index: int):
            action = np.zeros(n_actions)
            action[len(action) - index - 1] = 1
            return action.astype(np.float32)

        self.actions_one_hot = [get_one_hot_action(index) for index in range(0, n_actions)]
        self.labels = torch.tensor([[index] for index in range(0, n_actions)], device=self.device)

        self.final_confusion_matrix = torch.zeros(n_actions, n_actions, device=self.device)

        self.train_acc, self.train_precision, self.train_recall, self.train_f1, self.train_mAP, \
        self.valid_acc, self.valid_precision, self.valid_recall, self.valid_f1, self.valid_mAP, \
        self.test_acc, self.test_precision, self.test_recall, self.test_f1, self.test_mAP, \
        self.conf_matrix = get_metrics(n_actions)

        self.test_results = {
            'Inputs': [],
            'Embeddings': [],
            'Labels': [],
            'Predictions': [],
            'IsCorrect': [],
            'Ranking': [],
            'Ranking_similarity': [],
            'Distance_between_true_label_and_prediction': []
        }

        self.actions_results = {
            'Actions': [index for index in range(0, n_actions)],
            'Inputs': copy(self.actions_one_hot),
            'Embeddings': [],
        }

    def get_test_results(self) -> Dict[str, List]:

        if len(self.test_results['Vectors']) == 0:
            return {}

        return self.test_results

    @abc.abstractmethod
    def _make_a_transformation(self, x):
        return torch.tensor([])

    def forward(self, x, y):
        x, mask = self._make_a_transformation(x)
        x = self.dense_transformer(x)
        y = self.dense_action(y)
        return x, y, mask

    def _train_prediction(self, batch):
        x, y, _ = batch
        y_to_dense = torch.tensor([self.actions_one_hot[i] for i in y], device=self.device)
        all_label = torch.tensor(self.actions_one_hot, device=self.device)
        x_hat, y_hat, mask = self(x, y_to_dense)  # forward
        all_label_embedding = self.dense_action(all_label)
        return self.similarity_function(
            x_hat,
            y_hat,
            y.reshape(-1, 1),
            all_label_embedding,
            self.labels,
            mask
        )

    def training_step(self, batch, batch_idx):
        loss = self._train_prediction(batch)
        y_hat, _ = self.predict_step(batch, batch_idx)
        _, y, y_set = batch
        self.log('loss', loss, prog_bar=True)
        self.log('acc', self.train_acc(y_hat, y), prog_bar=True)
        self.log('acc_sets', self.train_mAP(y_hat, y_set), prog_bar=True)
        self.log('precision', self.train_precision(y_hat, y), prog_bar=True)
        self.log('recall', self.train_recall(y_hat, y), prog_bar=True)
        self.log('f1', self.train_f1(y_hat, y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._train_prediction(batch)
        y_hat, _ = self.predict_step(batch, batch_idx)
        _, y, y_set = batch
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.valid_acc(y_hat, y), prog_bar=True)
        self.log('val_sets', self.valid_mAP(y_hat, y_set), prog_bar=True)
        self.log('val_precision', self.valid_precision(y_hat, y), prog_bar=True)
        self.log('val_recall', self.valid_recall(y_hat, y), prog_bar=True)
        self.log('val_f1', self.valid_f1(y_hat, y), prog_bar=True)

    def test_step(self, batch, batch_idx):

        y_hat, ranking = self.predict_step(batch, batch_idx, is_test=True)
        _, y, y_set = batch

        self.log('test_acc', self.test_acc(y_hat, y))
        self.log('test_sets', self.test_mAP(y_hat, y_set))
        self.log('test_precision', self.test_precision(y_hat, y))
        self.log('test_recall', self.test_recall(y_hat, y))
        self.log('test_f1', self.test_f1(y_hat, y))

        for index in range(0, len(y)):
            self.test_results['Inputs'].append(batch[0][index].cpu().numpy()[0].tolist())
            self.test_results['Labels'].append(batch[1][index].cpu().numpy().tolist())
            self.test_results['Predictions'].append(y_hat[index].cpu().numpy().tolist())
            self.test_results['IsCorrect'].append(y_hat[index].cpu().numpy() == y[index].cpu().numpy())
            ranking_list_similarity = ranking[index].cpu().numpy()
            # transform the ranking to the original labels
            ranking_list = [(idx, ranking_value) for idx, ranking_value in enumerate(ranking_list_similarity)]
            ranking_list = sorted(ranking_list, key=lambda x: x[1], reverse=True)
            self.test_results['Ranking'].append(
                [idx for idx, _ in ranking_list]
            )
            self.test_results['Ranking_similarity'].append(
                [value for _, value in ranking_list]
            )
            prediction_similarity = self.test_results['Ranking_similarity'][-1][0]
            label_similarity = self.test_results['Ranking_similarity'][-1][
                self.test_results['Ranking'][-1].index(self.test_results['Labels'][-1])
            ]
            self.test_results['Distance_between_true_label_and_prediction'].append(
                abs(prediction_similarity - label_similarity)

            )

    def predict_step(self, batch, batch_idx: int, dataloader_idx: Optional[int] = None, is_test: bool = False):
        x, _, _ = batch
        y = torch.tensor([self.actions_one_hot] * x.size(0), device=self.device)
        x, _ = self._make_a_transformation(x)
        x = self.dense_transformer(x)
        y = self.dense_action(y)

        ranking = torch.tensor(
            [self.similarity_function.sim(x, y[:, j, :]).cpu().detach().numpy() for j in range(y.size(1))],
            device=self.device
        ).T
        actions = ranking.argmax(dim=1)
        if is_test:
            for index in range(0, len(x)):
                self.test_results['Embeddings'].append(x[index].cpu().numpy().tolist())

            if not self.actions_results['Embeddings']:
                self.actions_results['Embeddings'] = y.cpu().numpy().tolist()[0]
        return actions, ranking

    def configure_optimizers(self):
        return getattr(torch.optim, self.hparams.opt)(self.parameters(), lr=self.hparams.lr)
