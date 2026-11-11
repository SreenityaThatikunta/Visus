"""
Inference: load a checkpoint and segment occlusions in images given a text prompt.
"""
from __future__ import annotations

from pathlib import Path

import clip
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from .checkpoint import load_model_weights  # noqa: E402
from .config import get_device  # noqa: E402
from .models import EnhancedCLIPAttentionFinetuner  # noqa: E402

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
PROMPTS = ("glasses", "mask")

# Threshold used to produce the reported mIoU numbers.
DEFAULT_THRESHOLD = 0.9


class Predictor:
    """Wraps the model for single-image prediction and saving outputs."""

    def __init__(
        self,
        checkpoint_path: str | None = None,
        clip_model_name: str = "ViT-B/32",
        device: str | None = None,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        self.device = get_device(device)
        self.threshold = threshold

        self.model = EnhancedCLIPAttentionFinetuner(
            clip_model_name=clip_model_name, device=self.device
        ).float().to(self.device)

        if checkpoint_path:
            load_model_weights(self.model, checkpoint_path, self.device)

        self.model.eval()

    @torch.no_grad()
    def predict(self, image: str | Path | Image.Image, prompt: str) -> dict:
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = image.convert("RGB")

        pixels = self.model.preprocess(image).unsqueeze(0).to(self.device)
        tokens = clip.tokenize([prompt]).to(self.device)

        outputs = self.model(pixels, tokens)
        prob = outputs["binary_masks"][0, 0].cpu().numpy()

        return {
            "image": image,
            "prompt": prompt,
            "attention_map": outputs["attention_maps"][0, 0].cpu().numpy(),
            "probability": prob,
            "mask": (prob > self.threshold).astype(np.uint8),
        }

    @staticmethod
    def save(result: dict, out_dir: str | Path, stem: str, with_figure: bool = True) -> Path:
        """Write `<stem>_mask_binary.png` (and a 3-panel figure) to `out_dir`."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        mask_path = out_dir / f"{stem}_mask_binary.png"
        cv2.imwrite(str(mask_path), result["mask"] * 255)

        if with_figure:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            panels = [
                (np.array(result["image"].resize((224, 224))), None, "Input"),
                (result["attention_map"], "inferno", "CLIP attention"),
                (result["mask"], "gray", f'Mask ("{result["prompt"]}")'),
            ]
            for ax, (img, cmap, title) in zip(axes, panels):
                ax.imshow(img, cmap=cmap)
                ax.set_title(title)
                ax.axis("off")
            plt.tight_layout()
            fig.savefig(out_dir / f"{stem}_viz.png", dpi=120, bbox_inches="tight")
            plt.close(fig)

        return mask_path

    def cleanup(self):
        self.model.cleanup()


def list_images(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def prompts_from_filename(path: Path) -> list[str]:
    """Infer which occlusion prompt(s) apply from a filename like `0001_glasses.jpg`."""
    name = path.stem.lower()
    return [p for p in PROMPTS if p in name]
