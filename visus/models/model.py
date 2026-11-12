"""
Top-level model: wires CLIP + attention/text extractors + segmentation head.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import AttentionExtractor, TextAttentionExtractor, TextEmbedder
from .segmentation import EnhancedSegmentationHead


@dataclass
class ModelConfig:
    """Everything needed to rebuild a model; stored inside every checkpoint."""

    clip_model_name: str = "ViT-B/32"
    trainable_vision_blocks: int = 3      # last N CLIP vision blocks that fine-tune (0 = frozen CLIP)
    text_mode: str = "embedding"          # "embedding" (CLIP text embedding) | "attention" (legacy)
    text_layers_to_use: int = 3           # only for text_mode="attention"
    head_rgb: bool = True                 # also feed the RGB image to the head
    head_scale: float = 0.5               # head runs at this fraction of 224 px; logits are upsampled
    attn_layernorm: bool = True           # compute attention on LayerNorm-ed inputs (as CLIP does)
    predicts: str = "occlusion"           # what the sigmoid output means: "occlusion" | "background"

    @classmethod
    def legacy(cls) -> "ModelConfig":
        """The configuration of the released v1 checkpoint (`visus_epoch50.pt`)."""
        return cls(trainable_vision_blocks=0, text_mode="attention", head_rgb=False,
                   attn_layernorm=False, predicts="background", head_scale=1.0)

    def to_dict(self) -> dict:
        return asdict(self)


class EnhancedCLIPAttentionFinetuner(nn.Module):
    """
    1. CLS->patch attention map from CLIP's last vision block (7x7 for ViT-B/32, upsampled).
    2. A text feature for the prompt (CLIP text embedding in v2).
    3. A cross-modal segmentation head that turns (attention map [+ image], text) into a mask.

    `forward` returns `logits` and `probs`, where probs is P(pixel is occluded by the
    prompted object). For legacy checkpoints that predict background, probs is flipped
    so callers always get occlusion probabilities.
    """

    def __init__(self, config: ModelConfig | None = None, device: str = "cpu"):
        super().__init__()
        self.config = config or ModelConfig()
        self.device_name = device

        self.clip_model, self.preprocess = clip.load(self.config.clip_model_name, device=device)
        self.clip_model = self.clip_model.float()
        for p in self.clip_model.parameters():
            p.requires_grad = False
        n = self.config.trainable_vision_blocks
        if n > 0:
            for p in self.clip_model.visual.transformer.resblocks[-n:].parameters():
                p.requires_grad = True

        self.vision_attn_extractor = AttentionExtractor(
            self.clip_model, device=device,
            use_layernorm=self.config.attn_layernorm, trainable=n > 0,
        )
        if self.config.text_mode == "embedding":
            self.text_encoder = TextEmbedder(self.clip_model, device=device)
            text_dim = self.clip_model.text_projection.shape[1]
        elif self.config.text_mode == "attention":
            self.text_encoder = TextAttentionExtractor(
                self.clip_model, device=device, num_layers_to_extract=self.config.text_layers_to_use
            )
            text_dim = self.config.text_layers_to_use
        else:
            raise ValueError(f"unknown text_mode {self.config.text_mode!r}")

        self.seg_head = EnhancedSegmentationHead(
            in_channels=4 if self.config.head_rgb else 1,
            hidden_channels=128,
            text_feature_dim=text_dim,
            fusion_hidden_dim=64,
        )

    def trainable_vision_blocks(self) -> nn.ModuleList:
        n = self.config.trainable_vision_blocks
        blocks = self.clip_model.visual.transformer.resblocks
        return blocks[-n:] if n > 0 else nn.ModuleList()

    def forward(self, images: torch.Tensor, texts: torch.Tensor) -> dict:
        h, w = images.shape[2], images.shape[3]
        attn = self.vision_attn_extractor.get_attention_map(images, target_size=(h, w))
        text_features = self.text_encoder.get_text_features(texts)

        head_in = torch.cat([attn, images], dim=1) if self.config.head_rgb else attn
        s = self.config.head_scale
        if s != 1.0:   # the full-resolution head needs ~14 GB at batch 64; half resolution ~4 GB
            head_in = F.interpolate(head_in, scale_factor=s, mode="bilinear", align_corners=False)
        logits = self.seg_head(head_in, text_features)
        if logits.shape[-2:] != (h, w):
            logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        if self.config.predicts == "background":
            logits = -logits                     # sigmoid(-x) = 1 - sigmoid(x)
        return {"attention_maps": attn, "text_features": text_features,
                "logits": logits, "probs": torch.sigmoid(logits)}

    def cleanup(self):
        """Remove the forward hooks registered on CLIP."""
        self.vision_attn_extractor.cleanup()
        self.text_encoder.cleanup()


def build_model_and_optimizer(
    config: ModelConfig | None = None,
    learning_rate: float = 3e-4,
    clip_learning_rate: float = 1e-5,
    weight_decay: float = 0.01,
    device: str = "cpu",
) -> tuple[EnhancedCLIPAttentionFinetuner, torch.optim.Optimizer]:
    """Head trains at `learning_rate`; the unfrozen CLIP blocks at the smaller `clip_learning_rate`."""
    model = EnhancedCLIPAttentionFinetuner(config, device=device).to(device)
    groups = [{"params": list(model.seg_head.parameters()), "lr": learning_rate}]
    clip_params = [p for p in model.trainable_vision_blocks().parameters() if p.requires_grad]
    if clip_params:
        groups.append({"params": clip_params, "lr": clip_learning_rate})
    optimizer = torch.optim.AdamW(groups, weight_decay=weight_decay)
    return model, optimizer
