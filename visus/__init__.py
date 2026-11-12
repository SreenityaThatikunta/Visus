"""Visus: text-guided segmentation of facial occlusions with CLIP cross-modal attention."""

from .config import TrainingConfig, get_device
from .models import EnhancedCLIPAttentionFinetuner, ModelConfig, build_model_and_optimizer

__version__ = "2.0.0"

__all__ = [
    "TrainingConfig",
    "get_device",
    "EnhancedCLIPAttentionFinetuner",
    "ModelConfig",
    "build_model_and_optimizer",
]
