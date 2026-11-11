import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """1 - Dice coefficient, computed over the whole batch."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = predictions.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (predictions * targets).sum()
        dice_coeff = (2.0 * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )
        return 1.0 - dice_coeff
