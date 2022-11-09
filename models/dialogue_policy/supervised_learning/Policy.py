from abc import ABC

import pytorch_lightning as pl
import torch
import torchmetrics


class Policy(pl.LightningModule, ABC):

    @staticmethod
    def _create_metrics(n_actions: int):
        metrics = {
            'accuracy':  torchmetrics.classification.accuracy.Accuracy(),
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

    def __init__(self, config: dict, n_actions: int) -> None:
        super(Policy, self).__init__()
        self.save_hyperparameters(config)
        self.n_actions = n_actions

    def log_metrics(self, name: str, y_hat: torch.Tensor, y: torch.Tensor):
        metrics = self._create_metrics(self.n_actions)
        for metric_name, metric in metrics.items():
            self.log(f'{name}_{metric_name}', metric(y_hat, y), prog_bar=True)


