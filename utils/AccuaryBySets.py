import torch
from torchmetrics import Metric


class AccuracyBySets(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape[0] == target.shape[0], f"Bab dimension {preds.shape[0]} - {target.shape[0]}"

        for y, y_hat in zip(preds, target):
            self.correct += torch.sum(torch.tensor(int(y in y_hat)))
        self.total += len(preds)

    def compute(self):
        return self.correct.float() / self.total