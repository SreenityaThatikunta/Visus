"""
Training loop: mixed precision, warmup + cosine schedule, validation every epoch,
best-checkpoint selection on validation (not just "the last epoch").
"""
from __future__ import annotations

import json
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint
from .config import TrainingConfig
from .evaluation import evaluate_model
from .losses import BCEDiceLoss
from .visualize import plot_history, save_prediction_grid


class Trainer:
    def __init__(self, model, optimizer, config: TrainingConfig):
        self.model = model
        self.optimizer = optimizer
        self.cfg = config
        self.device = config.device
        self.amp = config.amp and self.device == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.criterion = BCEDiceLoss()
        self.history: list[dict] = []
        self.best_score = -1.0

    def _make_scheduler(self, steps_per_epoch: int):
        total = max(1, self.cfg.num_epochs * steps_per_epoch)
        warmup = max(1, int(self.cfg.warmup_epochs * steps_per_epoch))

        def factor(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(1, total - warmup)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, factor)

    def _step(self, batch) -> float:
        self.model.train()
        images = batch["images"].to(self.device, non_blocking=True)
        texts = batch["texts"].to(self.device, non_blocking=True)
        masks = batch["masks"].to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            out = self.model(images, texts)
        loss = self.criterion(out["logits"], masks)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        params = [p for g in self.optimizer.param_groups for p in g["params"]]
        torch.nn.utils.clip_grad_norm_(params, self.cfg.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return float(loss.detach())

    @torch.no_grad()
    def _val_loss(self, loader: DataLoader) -> float:
        self.model.eval()
        total, n = 0.0, 0
        for batch in loader:
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                out = self.model(batch["images"].to(self.device), batch["texts"].to(self.device))
            total += float(self.criterion(out["logits"], batch["masks"].to(self.device))) * len(batch["ids"])
            n += len(batch["ids"])
        return total / max(1, n)

    @torch.no_grad()
    def _save_samples(self, batch: dict, epoch: int, threshold: float):
        self.model.eval()
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            out = self.model(batch["images"].to(self.device), batch["texts"].to(self.device))
        save_prediction_grid(batch, out, f"{self.cfg.sample_dir}/epoch_{epoch:03d}.png", threshold)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, sample_batch: dict | None = None):
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        scheduler = self._make_scheduler(len(train_loader))
        t0 = time.time()

        for epoch in range(1, self.cfg.num_epochs + 1):
            print(f"\n=== Epoch {epoch}/{self.cfg.num_epochs} ===", flush=True)
            running, count = 0.0, 0
            for i, batch in enumerate(train_loader, 1):
                running += self._step(batch)
                count += 1
                scheduler.step()
                if i % self.cfg.log_interval == 0:
                    print(f"  batch {i}/{len(train_loader)}  loss {running / count:.4f}  "
                          f"lr {scheduler.get_last_lr()[0]:.2e}  {time.time() - t0:.0f}s", flush=True)
            train_loss = running / max(1, count)

            val_loss = self._val_loss(val_loader)
            metrics = evaluate_model(self.model, val_loader, self.device, amp=self.amp)
            best = metrics.best("overall")
            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val": best,
                   "val_at_0.5": metrics.summary(0.5), "minutes": (time.time() - t0) / 60}
            self.history.append(row)
            print(f"→ train {train_loss:.4f} | val {val_loss:.4f} | thr {best['threshold']:.1f}: "
                  f"IoU glasses {best['iou_glasses']:.3f}  mask {best['iou_mask']:.3f}  "
                  f"clean {best['clean_rate']:.3f}  overall {best['overall']:.3f}", flush=True)

            meta = {"epoch": epoch, "threshold": best["threshold"], "val": best,
                    "train_config": {k: v for k, v in vars(self.cfg).items()}}
            save_checkpoint(self.model, f"{self.cfg.checkpoint_dir}/last.pt", **meta)
            if best["overall"] > self.best_score:
                self.best_score = best["overall"]
                save_checkpoint(self.model, f"{self.cfg.checkpoint_dir}/best.pt", **meta)
                print(f"  new best ({self.best_score:.4f}) → best.pt", flush=True)

            if sample_batch is not None and (epoch % self.cfg.sample_interval == 0 or epoch == 1):
                self._save_samples(sample_batch, epoch, best["threshold"])
            with open(f"{self.cfg.output_dir}/history.json", "w") as fh:
                json.dump(self.history, fh, indent=2)
            plot_history(self.history, f"{self.cfg.output_dir}/training_curves.png")

        print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Best val overall {self.best_score:.4f}")
        return self.history
