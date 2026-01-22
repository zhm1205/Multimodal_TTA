import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses, miners

class FocalLoss(nn.Module):
    """Binary focal loss with logits.

    Args:
        alpha (float): Weighting factor for the class balance.
        gamma (float): Focusing parameter for modulating factor (1 - p_t).
        reduction (str): 'none' | 'mean' | 'sum'.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        probas = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - probas) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss

class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.loss_func = losses.TripletMarginLoss(margin=margin)
        self.miner = miners.TripletMarginMiner(margin=margin, type_of_triplets="all")

    def forward(self, embeddings, labels):
        mined_triplets = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, mined_triplets)
        return loss