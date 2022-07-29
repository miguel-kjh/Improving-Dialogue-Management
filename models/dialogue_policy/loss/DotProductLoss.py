from typing import Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from utils.ted_utils import get_candidate_values, batch_flatten, random_indices, _scale_loss


class DotProductLoss(nn.Module):
    """Abstract dot-product loss layer class.
    Idea based on StarSpace paper: http://arxiv.org/abs/1709.03856
    Implements similarity methods
    * `sim` (computes a similarity between vectors)
    * `get_similarities_and_confidences_from_embeddings` (calls `sim` and also computes
        confidence values)
    """

    def __init__(
            self,
            num_candidates: int,
            device,
            scale_loss: bool = False,
            constrain_similarities: bool = True,
    ):
        super(DotProductLoss, self).__init__()
        self.num_neg = num_candidates
        self.scale_loss = scale_loss
        self.constrain_similarities = constrain_similarities
        self.device = device

    def sim(
            self, a: torch.Tensor, b: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        a = a.to(self.device)
        b = b.to(self.device)

        sim = F.cosine_similarity(
            a.to(self.device),
            b.to(self.device),
            dim=-1
        )

        return sim

    def get_similarities_and_confidences_from_embeddings(
            self,
            input_embeddings: torch.Tensor,
            label_embeddings: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes similary between input and label embeddings and model's confidence.
        First compute the similarity from embeddings and then apply an activation
        function if needed to get the confidence.
        Args:
            input_embeddings: Embeddings of input.
            label_embeddings: Embeddings of labels.
            mask: Mask (should contain 1s for inputs and 0s for padding). Note, that
                `len(mask.shape) == len(a.shape) - 1` should hold.
        Returns:
            similarity between input and label embeddings and model's prediction
            confidence for each label.
        """
        similarities = self.sim(input_embeddings, label_embeddings, mask)
        confidences = F.softmax(similarities)
        return similarities, confidences

    def apply_mask_and_scaling(
            self, loss: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Scales the loss and applies the mask if necessary.
        Args:
            loss: The loss tensor
            mask: (Optional) A mask to multiply with the loss
        Returns:
            The scaled loss, potentially averaged over the sequence
            dimension.
        """
        """if mask is not None:
            loss *= mask"""

        loss *= _scale_loss(-loss)

        if len(loss.shape) == 2:
            # average over the sequence
            if mask is not None:
                loss = torch.mean(loss, dim=-1) / torch.mean(mask, dim=-1)
            else:
                loss = torch.mean(loss, dim=-1)

        return loss


class SingleLabelDotProductLoss(DotProductLoss):
    """Single-label dot-product loss layer.
    This loss layer assumes that only one output (label) is correct for any given input.
    """

    def __init__(
            self,
            num_candidates: int,
            device,
            scale_loss: bool = False,
            constrain_similarities: bool = True,
            mu_pos: float = 0.8,
            mu_neg: float = -0.2,
            use_max_sim_neg: bool = True,
            neg_lambda: float = 0.5,
            same_sampling: bool = False
    ) -> None:
        """Declares instance variables with default values.
        Args:
            num_candidates: Positive integer, the number of incorrect labels;
                the algorithm will minimize their similarity to the input.
            mu_pos: Indicates how similar the algorithm should
                try to make embedding vectors for correct labels;
                should be 0.0 < ... < 1.0 for `cosine` similarity type.
            mu_neg: Maximum negative similarity for incorrect labels,
                should be -1.0 < ... < 1.0 for `cosine` similarity type.
            use_max_sim_neg: If `True` the algorithm only minimizes
                maximum similarity over incorrect intent labels,
                used only if `loss_type` is set to `margin`.
            neg_lambda: The scale of how important it is to minimize
                the maximum similarity between embeddings of different labels,
                used only if `loss_type` is set to `margin`.
            scale_loss: If `True` scale loss inverse proportionally to
                the confidence of the correct prediction.
            same_sampling: If `True` sample same negative labels
                for the whole batch.
            constrain_similarities: If `True` and loss_type is `cross_entropy`, a
                sigmoid loss term is added to the total loss to ensure that similarity
                values are approximately bounded.
        """
        super().__init__(
            num_candidates,
            device,
            scale_loss=scale_loss,
            constrain_similarities=constrain_similarities
        )
        self.mu_pos = mu_pos
        self.mu_neg = mu_neg
        self.use_max_sim_neg = use_max_sim_neg
        self.neg_lambda = neg_lambda
        self.same_sampling = same_sampling

    def _get_bad_mask(
            self, labels: torch.Tensor, target_labels: torch.Tensor, idxs: torch.Tensor
    ) -> torch.Tensor:
        """Calculate bad mask for given indices.
        Checks that input features are different for positive negative samples.
        """
        pos_labels = torch.unsqueeze(target_labels, dim=-2)
        neg_labels = get_candidate_values(labels, idxs)


        return torch.all(
            torch.eq(
                neg_labels.to(self.device),
                pos_labels.to(self.device)
            ), -1)

    def _get_negs(
            self, embeds: torch.Tensor, labels: torch.Tensor, target_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embeds_flat = batch_flatten(embeds)
        labels_flat = batch_flatten(labels)
        target_labels_flat = batch_flatten(target_labels)

        total_candidates = embeds_flat.size(0)  # torch.shape(embeds_flat)[0]
        target_size = target_labels_flat.size(0)  # torch.shape(target_labels_flat)[0]

        neg_ids = random_indices(
            target_size, self.num_neg, total_candidates
        )

        neg_embeds = get_candidate_values(embeds_flat, neg_ids)
        bad_negs = self._get_bad_mask(labels_flat, target_labels_flat, neg_ids)

        # check if inputs have sequence dimension
        if len(target_labels.shape) == 3:
            # tensors were flattened for sampling, reshape back
            # add sequence dimension if it was present in the inputs
            target_shape = target_labels.size()
            neg_embeds = torch.reshape(
                neg_embeds, (target_shape[0], target_shape[1], -1, embeds.shape[-1])
            )
            bad_negs = torch.reshape(bad_negs, (target_shape[0], target_shape[1], -1))

        return neg_embeds, bad_negs

    def _sample_negatives(
            self,
            inputs_embed: Tensor,
            labels_embed: Tensor,
            labels: Tensor,
            all_labels_embed: Tensor,
            all_labels: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Sample negative examples."""

        pos_inputs_embed = torch.unsqueeze(inputs_embed, dim=-2)
        pos_labels_embed = torch.unsqueeze(labels_embed, dim=-2)

        # sample negative inputs
        neg_inputs_embed, inputs_bad_negs = self._get_negs(inputs_embed, labels, labels)
        # sample negative labels
        neg_labels_embed, labels_bad_negs = self._get_negs(
            all_labels_embed, all_labels, labels
        )
        return (
            pos_inputs_embed,
            pos_labels_embed,
            neg_inputs_embed,
            neg_labels_embed,
            inputs_bad_negs,
            labels_bad_negs,
        )

    def _train_sim(
            self,
            pos_inputs_embed: Tensor,
            pos_labels_embed: Tensor,
            neg_inputs_embed: Tensor,
            neg_labels_embed: Tensor,
            inputs_bad_negs: Tensor,
            labels_bad_negs: Tensor,
            mask: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Define similarity."""

        # calculate similarity with several
        # embedded actions for the loss
        neg_inf = -1e9

        sim_pos = self.sim(pos_inputs_embed, pos_labels_embed, mask)
        sim_neg_il = (
                self.sim(pos_inputs_embed, neg_labels_embed, mask)
                + neg_inf * labels_bad_negs
        )
        sim_neg_ll = (
                self.sim(pos_labels_embed, neg_labels_embed, mask)
                + neg_inf * labels_bad_negs
        )
        sim_neg_ii = (
                self.sim(pos_inputs_embed, neg_inputs_embed, mask)
                + neg_inf * inputs_bad_negs
        )
        sim_neg_li = (
                self.sim(pos_labels_embed, neg_inputs_embed, mask)
                + neg_inf * inputs_bad_negs
        )

        # output similarities between user input and bot actions
        # and similarities between bot actions and similarities between user inputs
        return sim_pos, sim_neg_il, sim_neg_ll, sim_neg_ii, sim_neg_li

    def _compute_softmax_loss(
            self,
            sim_pos: Tensor,
            sim_neg_il: Tensor,
            sim_neg_ll: Tensor,
            sim_neg_ii: Tensor,
            sim_neg_li: Tensor,
    ) -> Tensor:
        # Similarity terms between input and label should be optimized relative
        # to each other and hence use them as logits for softmax term
        softmax_logits = torch.concat(
            [sim_pos, sim_neg_il, sim_neg_li, sim_neg_ii, sim_neg_ll],
            dim=-1
        )
        # create label_ids for softmax
        softmax_label_ids = torch.zeros_like(softmax_logits[..., 0], dtype=torch.int32).type(torch.LongTensor)
        softmax_loss = F.cross_entropy(
            softmax_logits, softmax_label_ids
        )
        return softmax_loss

    def _loss_cross_entropy(
            self,
            sim_pos: Tensor,
            sim_neg_il: Tensor,
            sim_neg_ll: Tensor,
            sim_neg_ii: Tensor,
            sim_neg_li: Tensor,
            mask: Optional[Tensor],
    ) -> Tensor:
        """Defines cross entropy loss."""
        loss = self._compute_softmax_loss(
            sim_pos, sim_neg_il, sim_neg_ll, sim_neg_ii, sim_neg_li
        )

        loss = self.apply_mask_and_scaling(loss, mask)

        # average the loss over the batch
        return torch.mean(loss)

    def forward(
            self,
            inputs_embed: Tensor,
            labels_embed: Tensor,
            labels: Tensor,
            all_labels_embed: Tensor,
            all_labels: Tensor,
            mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Calculate loss and accuracy.
        Args:
            inputs_embed: Embedding tensor for the batch inputs;
                shape `(batch_size, ..., num_features)`
            labels_embed: Embedding tensor for the batch labels;
                shape `(batch_size, ..., num_features)`
            labels: Tensor representing batch labels; shape `(batch_size, ..., 1)`
            all_labels_embed: Embedding tensor for the all labels;
                shape `(num_labels, num_features)`
            all_labels: Tensor representing all labels; shape `(num_labels, 1)`
            mask: Optional mask, contains `1` for inputs and `0` for padding;
                shape `(batch_size, 1)`
        Returns:
            loss: Total loss.
            accuracy: Training accuracy.
        """

        (
            pos_inputs_embed,
            pos_labels_embed,
            neg_inputs_embed,
            neg_labels_embed,
            inputs_bad_negs,
            labels_bad_negs,
        ) = self._sample_negatives(
            inputs_embed, labels_embed, labels, all_labels_embed, all_labels
        )

        # calculate similarities
        sim_pos, sim_neg_il, sim_neg_ll, sim_neg_ii, sim_neg_li = self._train_sim(
            pos_inputs_embed,
            pos_labels_embed,
            neg_inputs_embed,
            neg_labels_embed,
            inputs_bad_negs,
            labels_bad_negs,
            mask,
        )

        #accuracy = self._calc_accuracy(sim_pos, sim_neg_il)

        loss = self._loss_cross_entropy(
            sim_pos, sim_neg_il, sim_neg_ll, sim_neg_ii, sim_neg_li, mask
        )

        return loss
