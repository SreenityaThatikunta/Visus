"""
Segmentation head: repeatedly refines vision features with cross-modal fusion.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .fusion import CrossModalAttentionFusion


class EnhancedSegmentationHead(nn.Module):
    """
    Three-stage refinement: each stage convolves, then fuses with text features
    via CrossModalAttentionFusion, before a final 1x1 conv produces a mask.

    Channel flow: 1 -> 128 -> CMAF(64) -> 128 -> CMAF(64) -> 64 -> CMAF(64) -> 1

    If text_features is None the fusion blocks fall back to their internal
    vision_conv (text guidance is skipped gracefully).

    Attribute names (cross_modal_fusionN, conv4) match the released checkpoints.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 128,
        text_feature_dim: int = 3,
        fusion_hidden_dim: int = 64,
    ):
        super().__init__()

        # Stage 1
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.cross_modal_fusion1 = CrossModalAttentionFusion(
            hidden_channels, text_feature_dim, fusion_hidden_dim
        )

        # Stage 2
        self.conv2 = nn.Conv2d(fusion_hidden_dim, hidden_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.cross_modal_fusion2 = CrossModalAttentionFusion(
            hidden_channels, text_feature_dim, fusion_hidden_dim
        )

        # Stage 3
        self.conv3 = nn.Conv2d(fusion_hidden_dim, hidden_channels // 2, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_channels // 2)
        self.cross_modal_fusion3 = CrossModalAttentionFusion(
            hidden_channels // 2, text_feature_dim, fusion_hidden_dim
        )

        # Output
        self.conv4 = nn.Conv2d(fusion_hidden_dim, 1, 1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def _fuse(block: CrossModalAttentionFusion, x: torch.Tensor, text: torch.Tensor | None):
        if text is not None:
            return block(x, text)
        return block.vision_conv(x)

    def forward(self, vision_features: torch.Tensor, text_features: torch.Tensor | None) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(vision_features)))
        x = self._fuse(self.cross_modal_fusion1, x, text_features)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self._fuse(self.cross_modal_fusion2, x, text_features)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self._fuse(self.cross_modal_fusion3, x, text_features)

        return self.sigmoid(self.conv4(x))
