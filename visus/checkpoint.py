"""
Save / load model weights.

Only the trainable parts are stored: the segmentation head and the
fine-tuned CLIP vision blocks. The frozen CLIP backbone is re-downloaded
by `clip.load`.

Two formats are supported when loading:
  * current  — key `vision_blocks_state_dict` (last 3 vision blocks)
  * legacy   — key `vision_last_layer_state_dict` (one block, loaded into
               resblocks[-3]); this is the format of the released
               `visus_epoch50.pt` and the checkpoints behind the reported results.
"""
from __future__ import annotations

import os
from datetime import datetime

import torch

from .models.model import NUM_UNFROZEN_VISION_BLOCKS


def save_model_weights(model, epoch: int, save_dir: str = "checkpoints") -> str:
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    blocks = model.clip_model.visual.transformer.resblocks[-NUM_UNFROZEN_VISION_BLOCKS:]
    state = {
        "epoch": epoch,
        "vision_blocks_state_dict": blocks.state_dict(),
        "seg_head_state_dict": model.seg_head.state_dict(),
        "timestamp": timestamp,
    }

    filepath = os.path.join(save_dir, f"visus_epoch_{epoch:03d}_{timestamp}.pt")
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")
    return filepath


def load_model_weights(model, checkpoint_path: str, device: str = "cpu") -> int:
    """Load weights into `model` in place. Returns the saved epoch."""
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    resblocks = model.clip_model.visual.transformer.resblocks

    if "vision_blocks_state_dict" in state:
        resblocks[-NUM_UNFROZEN_VISION_BLOCKS:].load_state_dict(state["vision_blocks_state_dict"])
    elif "vision_last_layer_state_dict" in state:
        resblocks[-NUM_UNFROZEN_VISION_BLOCKS].load_state_dict(state["vision_last_layer_state_dict"])
    else:
        raise KeyError(f"No vision-block weights found in {checkpoint_path}")

    model.seg_head.load_state_dict(state["seg_head_state_dict"])
    epoch = state.get("epoch", -1)
    print(f"Loaded checkpoint from epoch {epoch} ({checkpoint_path})")
    return epoch
