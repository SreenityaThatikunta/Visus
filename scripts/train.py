"""
Train Visus on an occluded-face dataset.

    python scripts/train.py --images data/celeba_occluded/images \
                            --masks  data/celeba_occluded/masks --epochs 50
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from visus.config import TrainingConfig, get_device  # noqa: E402
from visus.data import create_dataloaders  # noqa: E402
from visus.models import build_model_and_optimizer  # noqa: E402
from visus.trainer import Trainer  # noqa: E402


def parse_args() -> TrainingConfig:
    cfg = TrainingConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", default=cfg.images_dir, help="directory of occluded face images")
    p.add_argument("--masks", default=cfg.masks_dir, help="directory of ground-truth masks")
    p.add_argument("--output", default=cfg.output_dir, help="where checkpoints, samples and plots go")
    p.add_argument("--epochs", type=int, default=cfg.num_epochs)
    p.add_argument("--batch-size", type=int, default=cfg.batch_size)
    p.add_argument("--lr", type=float, default=cfg.learning_rate)
    p.add_argument("--max-samples", type=int, default=None, help="cap dataset size (for quick runs)")
    p.add_argument("--workers", type=int, default=cfg.num_workers)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--device", default=None, help="cuda / mps / cpu (auto-detected if omitted)")
    a = p.parse_args()

    cfg.images_dir, cfg.masks_dir, cfg.output_dir = a.images, a.masks, a.output
    cfg.num_epochs, cfg.batch_size, cfg.learning_rate = a.epochs, a.batch_size, a.lr
    cfg.max_samples, cfg.num_workers, cfg.seed = a.max_samples, a.workers, a.seed
    cfg.device = get_device(a.device)
    return cfg


def main():
    cfg = parse_args()
    torch.manual_seed(cfg.seed)
    print(f"Device: {cfg.device}")

    for d in (cfg.losses_dir, cfg.checkpoint_dir, cfg.train_sample_dir, cfg.val_sample_dir):
        os.makedirs(d, exist_ok=True)

    train_loader, val_loader = create_dataloaders(
        images_dir=cfg.images_dir,
        masks_dir=cfg.masks_dir,
        batch_size=cfg.batch_size,
        train_split=cfg.train_split,
        clip_model_name=cfg.clip_model_name,
        num_workers=cfg.num_workers,
        max_samples=cfg.max_samples,
        seed=cfg.seed,
    )
    train_sample_batch = next(iter(train_loader))
    val_sample_batch = next(iter(val_loader))

    model, optimizer = build_model_and_optimizer(
        clip_model_name=cfg.clip_model_name,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        text_layers_to_use=cfg.text_layers_to_use,
        device=cfg.device,
    )
    model.to(cfg.device)

    Trainer(model, optimizer, cfg).fit(train_loader, val_loader, train_sample_batch, val_sample_batch)
    model.cleanup()


if __name__ == "__main__":
    main()
