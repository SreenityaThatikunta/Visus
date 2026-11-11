"""
Visualisation utilities: save individual masks and comparison grids.
"""
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def save_masks_as_images(
    predicted_masks: torch.Tensor,
    ground_truth_masks: torch.Tensor,
    filenames: list[str],
    epoch: int,
    save_dir: str = "output_masks",
    every_n_epochs: int = 10,
    max_samples: int = 8,
) -> None:
    if epoch % every_n_epochs != 0:
        return

    epoch_dir = os.path.join(save_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    pred_np = predicted_masks.detach().cpu().numpy()
    gt_np   = ground_truth_masks.detach().cpu().numpy()
    n = min(len(pred_np), max_samples)

    for i in range(n):
        stem = Path(filenames[i]).stem
        pred_img = (pred_np[i, 0] * 255).astype(np.uint8)
        gt_img   = (gt_np[i, 0]   * 255).astype(np.uint8)

        cv2.imwrite(os.path.join(epoch_dir, f"{stem}_pred.png"), pred_img)
        cv2.imwrite(os.path.join(epoch_dir, f"{stem}_gt.png"),   gt_img)
        cv2.imwrite(
            os.path.join(epoch_dir, f"{stem}_comparison.png"),
            np.hstack([gt_img, pred_img]),
        )

    print(f"Saved {n} masks → {epoch_dir}")


def create_mask_visualization_grid(
    predicted_masks: torch.Tensor,
    ground_truth_masks: torch.Tensor,
    filenames: list[str],
    epoch: int,
    save_dir: str = "visualizations",
    every_n_epochs: int = 4,
    max_samples: int = 16,
) -> None:
    if epoch % every_n_epochs != 0:
        return

    os.makedirs(save_dir, exist_ok=True)

    pred_np = predicted_masks.detach().cpu().numpy()
    gt_np   = ground_truth_masks.detach().cpu().numpy()
    n = min(len(pred_np), max_samples)

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    if n == 1:
        axes = axes.reshape(-1, 1)

    for i in range(n):
        stem = Path(filenames[i]).stem
        axes[0, i].imshow(gt_np[i, 0],   cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"GT: {stem}",   fontsize=8)
        axes[0, i].axis("off")
        axes[1, i].imshow(pred_np[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_title(f"Pred: {stem}", fontsize=8)
        axes[1, i].axis("off")

    plt.suptitle(f"Mask comparison — epoch {epoch}", fontsize=12)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"grid_epoch_{epoch}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved grid → {save_path}")


def plot_loss_curve(
    train_losses: list[float],
    val_losses: list[float] | None = None,
    save_path: str = "loss_curve.png",
) -> None:
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_losses, "b-", linewidth=2, marker="o", label="Train loss")
    if val_losses:
        plt.plot(epochs, val_losses, "r-", linewidth=2, marker="s", label="Val loss")

    plt.title("Training & validation loss", fontsize=16, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Dice loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
