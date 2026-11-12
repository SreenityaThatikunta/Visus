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
    # Data: a folder built by scripts/data/kaggle_generate_occlusions.py
    data_dir: str = "data/visus_occlusion_v1"
    output_dir: str = "runs/default"
    max_samples: Optional[int] = None
    negative_prob: float = 0.25        # share of training prompts naming an absent object

    # Optimisation
    num_epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 3e-4        # segmentation head
    clip_learning_rate: float = 1e-5   # unfrozen CLIP vision blocks
    weight_decay: float = 0.01
    warmup_epochs: float = 1.0
    grad_clip: float = 1.0
    amp: bool = True                   # mixed precision on CUDA
    num_workers: int = 4
    seed: int = 42

    # Logging
    log_interval: int = 50             # batches between loss prints
    sample_interval: int = 2           # epochs between sample-prediction grids

    device: str = field(default_factory=get_device)

    @property
    def checkpoint_dir(self) -> str:
        return f"{self.output_dir}/checkpoints"

    @property
    def sample_dir(self) -> str:
        return f"{self.output_dir}/samples"
