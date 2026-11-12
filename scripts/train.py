"""
Train Visus on the synthetic occlusion dataset (see scripts/data/kaggle_generate_occlusions.py).

    python scripts/train.py --data data/visus_occlusion_v1 --epochs 20

Writes runs/<name>/checkpoints/{best,last}.pt, history.json, training_curves.png and
sample-prediction grids. `best.pt` is chosen on the validation split, and stores the
threshold that scored best there.

Ablations (all default to the v2 model):
    --trainable-blocks 0      frozen CLIP, only the head trains
    --text-mode attention     legacy 3-number text features instead of the CLIP text embedding
    --no-rgb                  head sees only the CLIP attention map, not the image
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from visus.config import TrainingConfig, get_device  # noqa: E402
from visus.data import OcclusionDataset, make_loader  # noqa: E402
from visus.models import ModelConfig, build_model_and_optimizer  # noqa: E402
from visus.trainer import Trainer  # noqa: E402


def parse_args():
    cfg, mcfg = TrainingConfig(), ModelConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=cfg.data_dir, help="dataset folder with images/, masks/, meta.csv")
    p.add_argument("--output", default=cfg.output_dir)
    p.add_argument("--epochs", type=int, default=cfg.num_epochs)
    p.add_argument("--batch-size", type=int, default=cfg.batch_size)
    p.add_argument("--lr", type=float, default=cfg.learning_rate, help="head learning rate")
    p.add_argument("--clip-lr", type=float, default=cfg.clip_learning_rate, help="learning rate of unfrozen CLIP blocks")
    p.add_argument("--negative-prob", type=float, default=cfg.negative_prob)
    p.add_argument("--max-samples", type=int, default=None, help="cap images per split (quick runs)")
    p.add_argument("--workers", type=int, default=cfg.num_workers)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--device", default=None, help="cuda / mps / cpu (auto-detected if omitted)")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--trainable-blocks", type=int, default=mcfg.trainable_vision_blocks)
    p.add_argument("--text-mode", choices=("embedding", "attention"), default=mcfg.text_mode)
    p.add_argument("--no-rgb", action="store_true")
    a = p.parse_args()

    cfg.data_dir, cfg.output_dir = a.data, a.output
    cfg.num_epochs, cfg.batch_size = a.epochs, a.batch_size
    cfg.learning_rate, cfg.clip_learning_rate = a.lr, a.clip_lr
    cfg.negative_prob, cfg.max_samples = a.negative_prob, a.max_samples
    cfg.num_workers, cfg.seed, cfg.amp = a.workers, a.seed, not a.no_amp
    cfg.device = get_device(a.device)
    mcfg.trainable_vision_blocks, mcfg.text_mode, mcfg.head_rgb = a.trainable_blocks, a.text_mode, not a.no_rgb
    return cfg, mcfg


def main():
    cfg, mcfg = parse_args()
    random.seed(cfg.seed), np.random.seed(cfg.seed), torch.manual_seed(cfg.seed)
    print(f"Device: {cfg.device} | model: {mcfg}")

    train_ds = OcclusionDataset(cfg.data_dir, "train", max_samples=cfg.max_samples, negative_prob=cfg.negative_prob)
    val_ds = OcclusionDataset(cfg.data_dir, "val", max_samples=cfg.max_samples)
    print(f"train images: {len(train_ds)} | val pairs: {len(val_ds)}")
    train_loader = make_loader(train_ds, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, seed=cfg.seed)
    val_loader = make_loader(val_ds, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, seed=cfg.seed)
    sample_batch = make_loader(val_ds, 8, shuffle=True, num_workers=0, seed=cfg.seed).__iter__().__next__()

    model, optimizer = build_model_and_optimizer(
        mcfg, learning_rate=cfg.learning_rate, clip_learning_rate=cfg.clip_learning_rate,
        weight_decay=cfg.weight_decay, device=cfg.device,
    )
    n_train = sum(p.numel() for g in optimizer.param_groups for p in g["params"])
    print(f"trainable parameters: {n_train:,}")

    Trainer(model, optimizer, cfg).fit(train_loader, val_loader, sample_batch)
    model.cleanup()


if __name__ == "__main__":
    main()
