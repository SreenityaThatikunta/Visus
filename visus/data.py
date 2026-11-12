"""
Datasets.

`OcclusionDataset` reads the dataset built by `scripts/data/kaggle_generate_occlusions.py`:

    <root>/images/<id>.jpg
    <root>/masks/<id>_glasses.png     occluded = white (255), else black
    <root>/masks/<id>_mask.png
    <root>/meta.csv                   id, split (train/val/test), has_glasses, has_mask, ...

Both mask files exist for every image (one is empty when that object is absent), so a
prompt can ask for an object that isn't there and the target is then all zeros.

Training samples one prompt per image: usually an object that is present, sometimes
(`negative_prob`) the absent one. Val/test enumerate every (image, object) pair so the
numbers are deterministic.

`PairDataset` wraps an explicit list of (image, prompt, ground truth) files, used for
the older LFW / CelebA test sets whose masks mark the occlusion in black.
"""
from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import clip
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2 as T
import torchvision.transforms.functional as TF

IMAGE_SIZE = 224
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

KINDS = ("glasses", "mask")
EVAL_PROMPTS = {"glasses": "glasses", "mask": "face mask"}
TRAIN_PROMPTS = {
    "glasses": ["glasses", "sunglasses", "eyeglasses", "eyewear", "spectacles", "shades",
                "a pair of glasses"],
    "mask": ["mask", "face mask", "surgical mask", "medical mask", "cloth mask",
             "mouth and nose covering", "a face mask"],
}

# A label smaller than this (fraction of the image) usually means a tiny or half-hidden
# face; those few images are skipped.
MIN_LABEL_FRACTION = {"glasses": 0.01, "mask": 0.02}


def image_to_tensor(image: Image.Image, size: int = IMAGE_SIZE) -> torch.Tensor:
    """Squash-resize to size x size (the same geometry is applied to masks) and CLIP-normalise."""
    image = image.convert("RGB").resize((size, size), Image.BICUBIC)
    return TF.normalize(TF.to_tensor(image), CLIP_MEAN, CLIP_STD)


def mask_to_tensor(mask: Image.Image, occluded_is_white: bool = True, size: int = IMAGE_SIZE) -> torch.Tensor:
    m = TF.to_tensor(mask.convert("L").resize((size, size), Image.NEAREST))
    return (m > 0.5).float() if occluded_is_white else (m <= 0.5).float()


@dataclass
class Sample:
    id: str
    kinds_present: tuple[str, ...]


class OcclusionDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        augment: bool | None = None,
        max_samples: Optional[int] = None,
        negative_prob: float = 0.25,
        image_size: int = IMAGE_SIZE,
    ):
        self.root = Path(root)
        self.split = split
        self.train = split == "train"
        self.augment = self.train if augment is None else augment
        self.negative_prob = negative_prob
        self.image_size = image_size

        samples = []
        area = 512 * 512
        with open(self.root / "meta.csv", newline="") as fh:
            for r in csv.DictReader(fh):
                if r["split"] != split:
                    continue
                present = tuple(k for k in KINDS if r[f"has_{k}"] == "1")
                if not present:
                    continue
                if any(int(r[f"{k}_px"]) < MIN_LABEL_FRACTION[k] * area
                       for k in present):
                    continue
                samples.append(Sample(r["id"], present))
        samples.sort(key=lambda s: s.id)
        if max_samples:
            samples = samples[:max_samples]
        self.samples = samples
        # eval: every (image, object) pair, present or not
        self.items = [(s, None) for s in samples] if self.train else [(s, k) for s in samples for k in KINDS]

        self.geom = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.9, 1.1), antialias=True),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
        ])
        self.color = T.ColorJitter(0.3, 0.3, 0.3, 0.03)

    def __len__(self) -> int:
        return len(self.items)

    def _pick_kind(self, s: Sample) -> str:
        absent = [k for k in KINDS if k not in s.kinds_present]
        if absent and random.random() < self.negative_prob:
            return random.choice(absent)
        return random.choice(s.kinds_present)

    def __getitem__(self, idx: int) -> dict:
        s, kind = self.items[idx]
        if kind is None:
            kind = self._pick_kind(s)
        image = Image.open(self.root / "images" / f"{s.id}.jpg").convert("RGB")
        mask = Image.open(self.root / "masks" / f"{s.id}_{kind}.png").convert("L")

        if self.augment:
            img_t = tv_tensors.Image(TF.pil_to_tensor(image))
            msk_t = tv_tensors.Mask(TF.pil_to_tensor(mask))
            img_t, msk_t = self.geom(img_t, msk_t)
            img_t = self.color(img_t)
            pixels = TF.normalize(img_t.float() / 255.0, CLIP_MEAN, CLIP_STD)
            target = (msk_t.float() / 255.0 > 0.5).float()
            prompt = random.choice(TRAIN_PROMPTS[kind])
        else:
            pixels = image_to_tensor(image, self.image_size)
            target = mask_to_tensor(mask, True, self.image_size)
            prompt = EVAL_PROMPTS[kind]

        return {"image": pixels, "mask": target, "text": prompt, "kind": kind,
                "present": kind in s.kinds_present, "id": s.id}


class PairDataset(Dataset):
    """Explicit (image path, kind, ground-truth path) triples; used for the legacy test sets."""

    def __init__(self, pairs: list[tuple[str, str, str]], occluded_is_white: bool):
        self.pairs = pairs
        self.occluded_is_white = occluded_is_white

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        img_path, kind, gt_path = self.pairs[idx]
        return {
            "image": image_to_tensor(Image.open(img_path)),
            "mask": mask_to_tensor(Image.open(gt_path), self.occluded_is_white),
            "text": EVAL_PROMPTS[kind], "kind": kind, "present": True, "id": Path(img_path).stem,
        }


def collate(batch: list[dict]) -> dict:
    return {
        "images": torch.stack([b["image"] for b in batch]),
        "texts": clip.tokenize([b["text"] for b in batch]),
        "prompts": [b["text"] for b in batch],
        "masks": torch.stack([b["mask"] for b in batch]),
        "kinds": [b["kind"] for b in batch],
        "present": torch.tensor([b["present"] for b in batch]),
        "ids": [b["id"] for b in batch],
    }


def _seed_worker(worker_id: int) -> None:
    # torch gives each worker a distinct seed; reuse it for Python's `random` (prompt choice)
    random.seed(torch.initial_seed() % 2**32)


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int = 4, seed: int = 42) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        collate_fn=collate, pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed), drop_last=shuffle,
        persistent_workers=num_workers > 0, worker_init_fn=_seed_worker,
    )
