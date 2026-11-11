"""
Top-level model: wires CLIP + attention extractors + segmentation head.
"""
from __future__ import annotations

import clip
import torch
import torch.nn as nn

from ..losses import DiceLoss
from .attention import AttentionExtractor, TextAttentionExtractor
from .segmentation import EnhancedSegmentationHead

NUM_UNFROZEN_VISION_BLOCKS = 3


class EnhancedCLIPAttentionFinetuner(nn.Module):
    """
    End-to-end model that:
      1. Extracts a spatial CLS-attention map from CLIP's visual encoder.
      2. Extracts aggregated attention features from CLIP's text encoder.
      3. Feeds both into a cross-modal segmentation head.

    `forward` returns per-pixel probabilities in `binary_masks`; threshold them
    (see `visus.inference.Predictor`) to get a hard mask.
    """

    def __init__(
        self,
        clip_model_name: str = "ViT-B/32",
        freeze_clip: bool = True,
        text_layers_to_use: int = 3,
        device: str = "cpu",
    ):
        super().__init__()

        self.clip_model, self.preprocess = clip.load(clip_model_name, device=device)

        if freeze_clip:
            for p in self.clip_model.parameters():
                p.requires_grad = False
            vision_blocks = self.clip_model.visual.transformer.resblocks
            for p in vision_blocks[-NUM_UNFROZEN_VISION_BLOCKS:].parameters():
                p.requires_grad = True

        self.vision_attn_extractor = AttentionExtractor(self.clip_model, device=device)
        self.text_attn_extractor = TextAttentionExtractor(
            self.clip_model, device=device, num_layers_to_extract=text_layers_to_use
        )

        self.seg_head = EnhancedSegmentationHead(
            in_channels=1,
            hidden_channels=128,
            text_feature_dim=text_layers_to_use,
            fusion_hidden_dim=64,
        )

        self.dice_loss = DiceLoss()

    def forward(
        self,
        images: torch.Tensor,
        texts: torch.Tensor,
        ground_truth: torch.Tensor | None = None,
    ) -> dict:
        h, w = images.shape[2], images.shape[3]

        vision_attn = self.vision_attn_extractor.get_attention_map(images, texts, target_size=(h, w))
        text_features = self.text_attn_extractor.get_text_features(texts)

        binary_masks = self.seg_head(vision_attn, text_features)

        loss = self.dice_loss(binary_masks, ground_truth) if ground_truth is not None else None

        return {
            "attention_maps": vision_attn,
            "text_features": text_features,
            "binary_masks": binary_masks,
            "loss": loss,
        }

    def cleanup(self):
        """Remove the forward hooks registered on CLIP."""
        self.vision_attn_extractor.cleanup()
        self.text_attn_extractor.cleanup()


def build_model_and_optimizer(
    clip_model_name: str = "ViT-B/32",
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    text_layers_to_use: int = 3,
    device: str = "cpu",
) -> tuple[EnhancedCLIPAttentionFinetuner, torch.optim.Optimizer]:
    model = EnhancedCLIPAttentionFinetuner(
        clip_model_name=clip_model_name,
        text_layers_to_use=text_layers_to_use,
        device=device,
    ).float()

    trainable_params = [
        *model.clip_model.visual.transformer.resblocks[-NUM_UNFROZEN_VISION_BLOCKS:].parameters(),
        *model.seg_head.parameters(),
    ]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)

    return model, optimizer
