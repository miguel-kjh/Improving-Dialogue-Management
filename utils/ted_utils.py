from operator import itemgetter
from typing import Union, Tuple, List

import torch
import torchmetrics
import torch.nn as nn
from torch import Tensor

import numpy as np

from torchmetrics import F1Score

from utils.AccuaryBySets import AccuracyBySets


def _f1_score(y_pred: Tensor, y_true: Tensor) -> Tensor:
    f1 = F1Score(num_classes=10)
    return f1(y_pred, y_true)


def create_embedding_layer(initial_dimensions: int, final_dimensions: int, dropout: int, eps: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(initial_dimensions, final_dimensions),
        nn.LayerNorm(final_dimensions, eps=eps),
        nn.Dropout(dropout)
    )


def create_ffn_layer(
        layer_sizes: List[List[int]],
        eps: float,
        use_bias: bool = True,
        activation=nn.ReLU()
) -> nn.Sequential:
    ffn = []

    for initial_dimension, final_dimension in layer_sizes:
        ffn.append(
            nn.Linear(initial_dimension, final_dimension, bias=use_bias)
        )
        ffn.append(
            activation
        )
        ffn.append(
            nn.LayerNorm(final_dimension, eps=eps)
        )

    return nn.Sequential(*ffn)


def _scale_loss(log_likelihood: Tensor) -> Tensor:
    """Creates scaling loss coefficient depending on the prediction probability.
    Arguments:
        log_likelihood: a tensor, log-likelihood of prediction
    Returns:
        Scaling tensor.
    """
    p = torch.exp(log_likelihood)

    return torch.pow((1 - p) / 0.5, 4).detach() if torch.max(p) > 0.5 else torch.ones_like(p)


def binary_vector2names(
        vectors: List[np.array],
        intentions: List[str],
        entities: List[str],
        actions: List[str]
) -> dict:
    names = {
        "intentions": [],
        "entities": [],
        "Prev_actions": []
    }

    for binary_vector in vectors:
        intentions_len = len(intentions)
        entities_len = len(entities)

        intent_index = np.where(binary_vector[:intentions_len] == 1)[0]
        entity_index = np.where(binary_vector[intentions_len:intentions_len + entities_len] == 1)[0]
        action_index = np.where(binary_vector[intentions_len + entities_len:] == 1)[0]

        names["intentions"].append(itemgetter(*intent_index)(intentions) if len(intent_index) > 0 else [])
        names["entities"].append(itemgetter(*entity_index)(entities) if len(entity_index) > 0 else [])
        names["Prev_actions"].append(itemgetter(*action_index)(actions) if len(action_index) > 0 else [])

    return names


def get_metrics(n_actions: int) -> Tuple:
    def metrics():
        acc = torchmetrics.classification.accuracy.Accuracy()
        precision = torchmetrics.classification.precision_recall.Precision(
            num_classes=n_actions,
            average='macro'
        )
        recall = torchmetrics.classification.precision_recall.Recall(
            num_classes=n_actions,
            average='macro'
        )

        acc_sets = AccuracyBySets()

        f1 = torchmetrics.classification.f_beta.F1(
            num_classes=n_actions,
            average='macro'
        )

        return acc, precision, recall, f1, acc_sets

    # metrics
    train_acc, train_precision, train_recall, train_f1, train_acc_sets = metrics()
    valid_acc, valid_precision, valid_recall, valid_f1, valid_acc_sets = metrics()
    test_acc, test_precision, test_recall, test_f1, test_acc_sets = metrics()

    conf_matrix = torchmetrics.classification.confusion_matrix.ConfusionMatrix(
        num_classes=n_actions
    )

    return train_acc, train_precision, train_recall, train_f1, train_acc_sets, \
           valid_acc, valid_precision, valid_recall, valid_f1, valid_acc_sets, \
           test_acc, test_precision, test_recall, test_f1, test_acc_sets, \
           conf_matrix


def random_indices(
        batch_size: Union[Tensor, int], n: Union[Tensor, int], n_max: Union[Tensor, int]
) -> Tensor:
    """Creates `batch_size * n` random indices that run from `0` to `n_max`.
    Args:
        batch_size: Number of items in each batch
        n: Number of random indices in each example
        n_max: Maximum index (excluded)
    Returns:
        A uniformly distributed integer tensor of indices
    """
    if n_max - 1 <= 0:
        return torch.zeros(batch_size, n, dtype=torch.int64)
    return torch.round(torch.distributions.uniform.Uniform(0, n_max - 1).sample([batch_size, n]))


def batch_flatten(x: Tensor) -> Tensor:
    """Flattens all but last dimension of `x` so it becomes 2D.
    Args:
        x: Any tensor with at least 2 dimensions
    Returns:
        The reshaped tensor, where all but the last dimension
        are flattened into the first dimension
    """
    return torch.reshape(x, (-1, x.shape[-1]))


def get_candidate_values(
        x: Tensor,  # (batch_size, ...)
        candidate_ids: Tensor,  # (batch_size, num_candidates)
) -> Tensor:
    """Gathers candidate values according to IDs.
    Args:
        x: Any tensor with at least one dimension
        candidate_ids: Indicator for which candidates to gather
    Returns:
        A tensor of shape `(batch_size, 1, num_candidates, x.size(-1))`, where
        for each batch example, we generate a list of `num_candidates` vectors, and
        each candidate is chosen from `x` according to the candidate id. For example:
        ```
        x = [[0 1 2],
                [3 4 5],
                [6 7 8]]
        candidate_ids = [[0, 1], [0, 0], [2, 0]]
        gives
        [
            [[0 1 2],
             [3 4 5]],
            [[0 1 2],
             [0 1 2]],
            [[6 7 8],
             [0 1 2]]
        ]
        ```
    """

    return torch.Tensor(
        [x[id.long()].cpu().detach().numpy() for id in candidate_ids]
    )


def th_random_choice(a, n_samples=1, replace=True, p=None):
    """
    Parameters
    -----------
    a : 1-D array-like
        If a th.Tensor, a random sample is generated from its elements.
        If an int, the random sample is generated as if a was th.range(n)
    n_samples : int, optional
        Number of samples to draw. Default is None, in which case a
        single value is returned.
    replace : boolean, optional
        Whether the sample is with or without replacement
    p : 1-D array-like, optional
        The probabilities associated with each entry in a.
        If not given the sample assumes a uniform distribution over all
        entries in a.
    Returns
    --------
    samples : 1-D ndarray, shape (size,)
        The generated random samples
    """
    if isinstance(a, int):
        a = torch.arange(0, a)

    if p is None:
        if replace:
            idx = torch.floor(torch.rand(n_samples) * a.size(0)).long()
        else:
            idx = torch.randperm(len(a))[:n_samples]
    else:
        if abs(1.0 - sum(p)) > 1e-3:
            raise ValueError('p must sum to 1.0')
        if not replace:
            raise ValueError('replace must equal true if probabilities given')
        idx_vec = torch.cat([torch.zeros(round(p[i] * 1000)) + i for i in range(len(p))])
        idx = (torch.floor(torch.rand(n_samples) * 999)).long()
        idx = idx_vec[idx].long()
    selection = a[idx]
    if n_samples == 1:
        selection = selection[0]
    return selection


if __name__ == '__main__':
    """candidate_ids = Tensor([[0, 1], [0, 0], [2, 0]])
    r = get_candidate_values(
        torch.tensor(
            [[0, 1, 2],
             [3, 4, 5],
             [6, 7, 8]]
        ),
        torch.tensor(
            candidate_ids
        )
    )
    print(r)
    print(candidate_ids.size())
    print(r.size())"""

    print(_scale_loss(Tensor([-0.6, 93])))
