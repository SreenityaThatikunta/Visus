"""
Trainer: owns the epoch loop, logging, and periodic output saving.
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .visualize import (
    save_masks_as_images,
    create_mask_visualization_grid,
    plot_loss_curve,
)
from .checkpoint import save_model_weights


class Trainer:
    def __init__(self, model, optimizer, config: TrainingConfig):
        self.model     = model
        self.optimizer = optimizer
        self.cfg       = config
        self.device    = config.device

    # ------------------------------------------------------------------
    # Single step
    # ------------------------------------------------------------------

    def _train_step(self, images, texts, ground_truth) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(images, texts, ground_truth)
        outputs["loss"].backward()
        self.optimizer.step()
        return outputs["loss"].item()

    # ------------------------------------------------------------------
    # Epoch loops
    # ------------------------------------------------------------------

    def _run_train_epoch(self, loader: DataLoader, epoch: int) -> float:
        total_loss = 0.0
        for i, batch in enumerate(loader):
            images       = batch["images"].to(self.device)
            texts        = batch["texts"].to(self.device)
            ground_truth = batch["masks"].to(self.device)

            loss = self._train_step(images, texts, ground_truth)
            total_loss += loss

            if (i + 1) % self.cfg.log_interval == 0:
                print(f"  [epoch {epoch}] batch {i+1}/{len(loader)}  loss={loss:.4f}")

        return total_loss / len(loader)

    @torch.no_grad()
    def _run_val_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        for batch in loader:
            images       = batch["images"].to(self.device)
            texts        = batch["texts"].to(self.device)
            ground_truth = batch["masks"].to(self.device)
            outputs = self.model(images, texts, ground_truth)
            total_loss += outputs["loss"].item()
        return total_loss / len(loader)

    # ------------------------------------------------------------------
    # Periodic output saving
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _save_sample_outputs(self, batch: dict, epoch: int, tag: str, out_dir: str):
        self.model.eval()
        images = batch["images"].to(self.device)
        texts  = batch["texts"].to(self.device)
        outputs = self.model(images, texts)
        pred_masks = outputs["binary_masks"].cpu()
        gt_masks   = batch["masks"]
        filenames  = batch["filenames"]

        save_masks_as_images(
            pred_masks, gt_masks, filenames, epoch,
            save_dir=out_dir,
            every_n_epochs=self.cfg.mask_save_interval,
        )
        create_mask_visualization_grid(
            pred_masks, gt_masks, filenames, epoch,
            save_dir=f"{out_dir}/grids",
            every_n_epochs=self.cfg.vis_interval,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        train_sample_batch: dict,
        val_sample_batch:   dict,
    ):
        train_losses, val_losses = [], []

        for epoch in range(1, self.cfg.num_epochs + 1):
            print(f"\n{'='*20} Epoch {epoch}/{self.cfg.num_epochs} {'='*20}")

            avg_train = self._run_train_epoch(train_loader, epoch)
            train_losses.append(avg_train)
            print(f"→ Train loss: {avg_train:.4f}")

            avg_val = self._run_val_epoch(val_loader)
            val_losses.append(avg_val)
            print(f"→ Val   loss: {avg_val:.4f}")

            if epoch % self.cfg.save_interval == 0 or epoch == 1:
                save_model_weights(self.model, epoch, self.cfg.checkpoint_dir)
                self._save_sample_outputs(
                    train_sample_batch, epoch, "train", self.cfg.train_sample_dir
                )
                self._save_sample_outputs(
                    val_sample_batch, epoch, "val", self.cfg.val_sample_dir
                )
                plot_loss_curve(
                    train_losses, val_losses,
                    save_path=f"{self.cfg.losses_dir}/loss_curve_epoch_{epoch}.png",
                )

        print("\nTraining complete.")
        save_model_weights(self.model, self.cfg.num_epochs, self.cfg.checkpoint_dir)
        plot_loss_curve(
            train_losses, val_losses,
            save_path=f"{self.cfg.losses_dir}/loss_curve_final.png",
        )
