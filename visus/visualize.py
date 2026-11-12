"""
Plots: sample-prediction grids during training and the training history.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from .data import CLIP_MEAN, CLIP_STD  # noqa: E402


def denormalize(images: torch.Tensor) -> np.ndarray:
    """(B,3,H,W) CLIP-normalised -> (B,H,W,3) uint8."""
    mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
    x = (images.detach().cpu().float() * std + mean).clamp(0, 1)
    return (x.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)


def save_prediction_grid(batch: dict, outputs: dict, path: str, threshold: float = 0.5, n: int = 8) -> None:
    """Rows: image + prompt, CLIP attention, predicted probability, thresholded mask, ground truth."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = min(n, len(batch["ids"]))
    imgs = denormalize(batch["images"][:n])
    attn = outputs["attention_maps"][:n, 0].detach().float().cpu().numpy()
    prob = outputs["probs"][:n, 0].detach().float().cpu().numpy()
    gt = batch["masks"][:n, 0].cpu().numpy()
    texts = [f'"{t}"' for t in batch.get("prompts", batch["kinds"])[:n]]

    fig, axes = plt.subplots(5, n, figsize=(2 * n, 10.5))
    axes = np.atleast_2d(axes).reshape(5, n)
    rows = [("input", None), ("CLIP attention", "inferno"), ("P(occluded)", "viridis"),
            (f"mask > {threshold:g}", "gray"), ("ground truth", "gray")]
    for j in range(n):
        panels = [imgs[j], attn[j], prob[j], (prob[j] > threshold).astype(float), gt[j]]
        for i, (img, (title, cmap)) in enumerate(zip(panels, rows)):
            ax = axes[i, j]
            ax.imshow(img, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
            ax.set_xticks([]), ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(title, fontsize=9)
        axes[0, j].set_title(texts[j], fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def plot_history(history: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not history:
        return
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history], marker="o", label="train")
    ax[0].plot(ep, [h["val_loss"] for h in history], marker="o", label="val")
    ax[0].set(xlabel="epoch", ylabel="BCE + Dice loss", title="Loss")
    ax[0].legend(), ax[0].grid(alpha=0.3)
    for key, label in (("iou_glasses", "IoU glasses"), ("iou_mask", "IoU face mask"),
                       ("clean_rate", "clean rate (absent object)")):
        ax[1].plot(ep, [h["val"][key] for h in history], marker="o", label=label)
    ax[1].set(xlabel="epoch", ylim=(0, 1), title="Validation (best threshold per epoch)")
    ax[1].legend(), ax[1].grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
