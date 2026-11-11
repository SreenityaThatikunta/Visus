"""
Cross-modal attention fusion: conditions spatial vision features on text features.
"""
import torch
import torch.nn as nn


class CrossModalAttentionFusion(nn.Module):
    """
    Projects text features into a shared space, then uses them as
    key/value in multi-head cross-attention over spatial vision features.
    """

    def __init__(self, vision_channels: int = 1, text_feature_dim: int = 3, hidden_dim: int = 64):
        super().__init__()

        self.text_proj = nn.Sequential(
            nn.Linear(text_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True,
        )
        self.vision_conv = nn.Sequential(
            nn.Conv2d(vision_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
        )

    def forward(self, vision_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        B, _, H, W = vision_features.shape

        text_feat = self.text_proj(text_features).unsqueeze(1)          # (B, 1, hidden_dim)
        vision_feat = self.vision_conv(vision_features)                  # (B, hidden_dim, H, W)
        vision_flat = vision_feat.view(B, vision_feat.shape[1], -1).transpose(1, 2)  # (B, H*W, hidden_dim)

        attended, _ = self.cross_attn(query=vision_flat, key=text_feat, value=text_feat)
        attended = attended.transpose(1, 2).view(B, -1, H, W)           # (B, hidden_dim, H, W)

        fused = torch.cat([vision_feat, attended], dim=1)               # (B, 2*hidden_dim, H, W)
        return self.fusion_conv(fused)                                   # (B, hidden_dim, H, W)
