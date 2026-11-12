"""
Save / load model weights.

Only the trainable parts are stored: the segmentation head and the fine-tuned CLIP
vision blocks. The frozen CLIP backbone is re-downloaded by `clip.load`.

Formats:
  * v2 (current) — keys `model_config`, `vision_blocks_state_dict`, `seg_head_state_dict`,
                   plus training metadata (epoch, val metrics, chosen threshold).
  * v1 (legacy)  — key `vision_last_layer_state_dict` (one block, loaded into
                   resblocks[-3]); the format of the released `visus_epoch50.pt`.
"""
from __future__ import annotations

import os

import torch

from .models.model import EnhancedCLIPAttentionFinetuner, ModelConfig


def save_checkpoint(model: EnhancedCLIPAttentionFinetuner, path: str, **metadata) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {
        "format": 2,
        "model_config": model.config.to_dict(),
        "vision_blocks_state_dict": model.trainable_vision_blocks().state_dict(),
        "seg_head_state_dict": model.seg_head.state_dict(),
        **metadata,
    }
    torch.save(state, path)
    return path


def read_checkpoint(path: str, device: str = "cpu") -> tuple[ModelConfig, dict]:
    state = torch.load(path, map_location=device, weights_only=False)
    if "model_config" in state:
        return ModelConfig(**state["model_config"]), state
    if "vision_last_layer_state_dict" in state:
        return ModelConfig.legacy(), state
    raise KeyError(f"{path} is not a Visus checkpoint")


def load_model(path: str, device: str = "cpu") -> tuple[EnhancedCLIPAttentionFinetuner, dict]:
    """Build the right model for a checkpoint and load its weights. Returns (model, state)."""
    config, state = read_checkpoint(path, device)
    model = EnhancedCLIPAttentionFinetuner(config, device=device).to(device)
    resblocks = model.clip_model.visual.transformer.resblocks
    if "vision_blocks_state_dict" in state:
        if state["vision_blocks_state_dict"]:
            model.trainable_vision_blocks().load_state_dict(state["vision_blocks_state_dict"])
    else:  # legacy: a single block saved from resblocks[-3]
        resblocks[-3].load_state_dict(state["vision_last_layer_state_dict"])
    model.seg_head.load_state_dict(state["seg_head_state_dict"])
    print(f"Loaded {path} (epoch {state.get('epoch', '?')}, "
          f"{'legacy v1' if 'model_config' not in state else 'v2'})")
    return model, state
