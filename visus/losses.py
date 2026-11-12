import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """1 - soft Dice coefficient over the whole batch. Takes probabilities."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs, targets = probs.reshape(-1), targets.reshape(-1)
        inter = (probs * targets).sum()
        return 1.0 - (2.0 * inter + self.smooth) / (probs.sum() + targets.sum() + self.smooth)


class BCEDiceLoss(nn.Module):
    """Binary cross-entropy (on logits) + Dice. BCE gives stable per-pixel gradients,
    Dice counters the class imbalance (occlusions cover ~5-15% of a face image)."""

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.bce_weight, self.dice_weight = bce_weight, dice_weight
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        return self.bce_weight * bce + self.dice_weight * self.dice(torch.sigmoid(logits), targets)
