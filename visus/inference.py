"""
Inference: load a checkpoint and segment the occlusion named by a text prompt.

Masks are written with the occluded region in WHITE (255) and the rest black, for both
v2 checkpoints and the legacy v1 release (whose raw output is flipped on load).
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

from .checkpoint import load_model  # noqa: E402
from .config import get_device  # noqa: E402
from .data import image_to_tensor  # noqa: E402

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
PROMPT_WORDS = {"glasses": "glasses", "mask": "face mask"}

# Legacy v1 thresholded P(background) > 0.9, i.e. P(occluded) > 0.1.
LEGACY_THRESHOLD = 0.1


class Predictor:
    def __init__(self, checkpoint_path: str, device: str | None = None, threshold: float | None = None):
        self.device = get_device(device)
        self.model, state = load_model(checkpoint_path, self.device)
        self.model.eval()
        if threshold is None:
            threshold = state.get("threshold", LEGACY_THRESHOLD if "model_config" not in state else 0.5)
        self.threshold = threshold

    @torch.no_grad()
    def predict(self, image: str | Path | Image.Image, prompt: str) -> dict:
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = image.convert("RGB")
        pixels = image_to_tensor(image).unsqueeze(0).to(self.device)
        tokens = clip.tokenize([prompt]).to(self.device)
        out = self.model(pixels, tokens)

        prob = out["probs"][0, 0].float().cpu().numpy()
        w, h = image.size
        prob_full = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        return {
            "image": image,
            "prompt": prompt,
            "attention_map": out["attention_maps"][0, 0].float().cpu().numpy(),
            "probability": prob_full,
            "mask": (prob_full > self.threshold).astype(np.uint8),   # 1 = occluded
        }

    @staticmethod
    def save(result: dict, out_dir: str | Path, stem: str, with_figure: bool = True) -> Path:
        """Write `<stem>_mask_binary.png` (occluded = white) and optionally a 3-panel figure."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        mask_path = out_dir / f"{stem}_mask_binary.png"
        cv2.imwrite(str(mask_path), result["mask"] * 255)

        if with_figure:
            img = np.array(result["image"])
            overlay = img.copy()
            m = result["mask"].astype(bool)
            overlay[m] = (0.45 * overlay[m] + 0.55 * np.array([255, 40, 40])).astype(np.uint8)
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            panels = [(img, None, "Input"), (result["attention_map"], "inferno", "CLIP attention"),
                      (overlay, None, f'Mask: "{result["prompt"]}"')]
            for ax, (im, cmap, title) in zip(axes, panels):
                ax.imshow(im, cmap=cmap)
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
    """`0001_glasses.jpg` -> ["glasses"], `0001_mask.jpg` -> ["face mask"]."""
    name = path.stem.lower()
    return [prompt for key, prompt in PROMPT_WORDS.items() if key in name]
