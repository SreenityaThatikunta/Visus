from dataclasses import dataclass, field
from typing import Optional

import torch


def get_device(preferred: Optional[str] = None) -> str:
    """Return `preferred` if given, otherwise the best available device (cuda > mps > cpu)."""
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class TrainingConfig:
    # Paths
    images_dir: str = "data/celeba_occluded/images"
    masks_dir: str = "data/celeba_occluded/masks"
    output_dir: str = "runs/default"

    # Model
    clip_model_name: str = "ViT-B/32"
    freeze_clip: bool = True
    text_layers_to_use: int = 3

    # Training
    num_epochs: int = 50
    batch_size: int = 24
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    train_split: float = 0.8
    num_workers: int = 4
    max_samples: Optional[int] = None
    seed: int = 42

    # Logging
    log_interval: int = 20         # batches between loss prints
    save_interval: int = 2         # epochs between checkpoint saves
    vis_interval: int = 4          # epochs between visualisation grids
    mask_save_interval: int = 10   # epochs between individual mask saves

    # Device
    device: str = field(default_factory=get_device)

    @property
    def checkpoint_dir(self) -> str:
        return f"{self.output_dir}/checkpoints"

    @property
    def train_sample_dir(self) -> str:
        return f"{self.output_dir}/samples/train"

    @property
    def val_sample_dir(self) -> str:
        return f"{self.output_dir}/samples/val"

    @property
    def losses_dir(self) -> str:
        return f"{self.output_dir}/losses"
