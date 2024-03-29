from abc import ABC

import pytorch_lightning as pl
import torch
import torchmetrics


class Policy(pl.LightningModule, ABC):

    @staticmethod
    def _create_metrics(n_actions: int) -> dict:

        metrics = {
            'accuracy': torchmetrics.classification.accuracy.Accuracy(
                num_classes=n_actions,
                average='macro'
            ),
            'precision': torchmetrics.classification.precision_recall.Precision(
                num_classes=n_actions,
                average='macro'
            ),
            'recall': torchmetrics.classification.precision_recall.Recall(
                num_classes=n_actions,
                average='macro'
            ),
            'f1': torchmetrics.classification.f_beta.F1(
                num_classes=n_actions,
                average='macro'
            ),
        }
        return metrics

    @staticmethod
    def _create_metrics_multiclass(n_actions: int, mdmc_average='global', average='macro') -> dict:

        metrics = {
            'accuracy': torchmetrics.classification.accuracy.Accuracy(
                num_classes=n_actions,
                mdmc_average=mdmc_average,
                average=average
            ),
            'precision': torchmetrics.classification.precision_recall.Precision(
                num_classes=n_actions,
                average=average,
                mdmc_average=mdmc_average
            ),
            'recall': torchmetrics.classification.precision_recall.Recall(
                num_classes=n_actions,
                average=average,
                mdmc_average=mdmc_average
            ),
            'f1': torchmetrics.classification.f_beta.F1(
                num_classes=n_actions,
                average=average,
                mdmc_average=mdmc_average
            ),
        }
        return metrics

    def __init__(self, config: dict, n_actions: int, embedding_size: int) -> None:
        super(Policy, self).__init__()
        self.save_hyperparameters(config)
        self.n_actions = n_actions
        self.embedding_size = embedding_size
        self.test_results = {
            'Dialogue_ID': [],
            'Index': [],
            'Inputs': [],
            'Labels': [],
            'Predictions': [],
            'IsCorrect': [],
        }

    def log_metrics(self, name: str, y_hat: torch.Tensor, y: torch.Tensor, multiclass: bool = False) -> None:

        if multiclass:
            metrics = self._create_metrics_multiclass(2)
        else:
            metrics = self._create_metrics(self.n_actions)

        y_hat = y_hat.cpu().detach()
        y = y.cpu().detach()

        for metric_name, metric in metrics.items():
            self.log(f'{name}_{metric_name}', metric(y_hat, y), prog_bar=True)
