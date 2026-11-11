"""
Metrics and prediction <-> ground-truth matching for the CelebA and LFW test sets.

Predictions are files named `<image stem>_<prompt>_mask_binary.png`, as written by
`scripts/predict.py`.

Mask convention (inherited from the dataset): occluded pixels are BLACK (0) and the
rest of the image is WHITE (255), in both ground truth and predictions. The occlusion
covers only ~7% of an image, so IoU of the white background is high even for an
empty prediction. The headline metric here is therefore IoU of the occlusion region;
background IoU and their mean (two-class mIoU) are reported alongside it.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

MASK_SIZE = (224, 224)
BINARY_THRESHOLD = 128
SUFFIX = "_mask_binary"


def load_binary_mask(path: str | Path, size: tuple[int, int] | None = MASK_SIZE) -> np.ndarray:
    """Return 1 for white (background) pixels and 0 for black (occluded) pixels."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load: {path}")
    if size is not None and img.shape[::-1] != size:
        img = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
    return (img >= BINARY_THRESHOLD).astype(np.uint8)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def dice_loss(pred: np.ndarray, gt: np.ndarray, smooth: float = 1.0) -> float:
    pred = pred.astype(np.float32).ravel()
    gt = gt.astype(np.float32).ravel()
    inter = (pred * gt).sum()
    return float(1.0 - (2.0 * inter + smooth) / (pred.sum() + gt.sum() + smooth))


# ---------------------------------------------------------------------------
# Ground-truth matching
# ---------------------------------------------------------------------------

def _parse_prediction(name: str) -> tuple[str, str] | None:
    """'<stem>_<prompt>_mask_binary' -> (stem, prompt)."""
    name = name.lower()
    for prompt in ("glasses", "mask"):
        tail = f"_{prompt}{SUFFIX}"
        if name.endswith(tail):
            return name[: -len(tail)], prompt
    return None


def build_gt_index(gt_dir: str | Path, dataset: str) -> dict:
    """
    celeba: `<gt_dir>/{glasses,mask}/<id>.jpg`      -> key (id, prompt)
    lfw:    `<gt_dir>/{glasses,mask}/<Name_0001_prompt>.jpg` -> key lowercase stem
    """
    index = {}
    # os.walk with followlinks so symlinked dataset folders work
    files = [Path(root) / n for root, _, names in os.walk(gt_dir, followlinks=True) for n in names]
    for f in files:
        if f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        if dataset == "celeba":
            index[(f.stem, f.parent.name)] = f
        else:
            index[f.stem.lower()] = f
    return index


def match_gt(pred_path: Path, gt_index: dict, dataset: str) -> tuple[Path, str] | None:
    parsed = _parse_prediction(pred_path.stem)
    if parsed is None:
        return None
    stem, prompt = parsed
    if dataset == "celeba":
        m = re.match(r"(\d+)", stem)
        key = (m.group(1), prompt) if m else None
    else:
        key = stem
    gt = gt_index.get(key)
    return (gt, prompt) if gt is not None else None


def _per_prompt():
    return {"glasses": [], "mask": []}


@dataclass
class EvalResult:
    ious: dict[str, list[float]] = field(default_factory=_per_prompt)      # occlusion IoU
    bg_ious: dict[str, list[float]] = field(default_factory=_per_prompt)   # background IoU
    dice: dict[str, list[float]] = field(default_factory=_per_prompt)      # occlusion Dice loss

    @staticmethod
    def _mean(d: dict, prompt: str | None) -> float:
        vals = d[prompt] if prompt else d["glasses"] + d["mask"]
        return float(np.mean(vals)) if vals else float("nan")

    def miou(self, prompt: str | None = None) -> float:
        """Mean IoU of the occluded region."""
        return self._mean(self.ious, prompt)

    def background_iou(self, prompt: str | None = None) -> float:
        return self._mean(self.bg_ious, prompt)

    def two_class_miou(self, prompt: str | None = None) -> float:
        return (self.miou(prompt) + self.background_iou(prompt)) / 2

    def mean_dice(self, prompt: str | None = None) -> float:
        return self._mean(self.dice, prompt)

    @property
    def count(self) -> int:
        return len(self.ious["glasses"]) + len(self.ious["mask"])


def evaluate_predictions(pred_dir: str | Path, gt_dir: str | Path, dataset: str) -> EvalResult:
    gt_index = build_gt_index(gt_dir, dataset)
    result = EvalResult()
    for pred_path in sorted(Path(pred_dir).rglob(f"*{SUFFIX}.png")):
        match = match_gt(pred_path, gt_index, dataset)
        if match is None:
            continue
        gt_path, prompt = match
        pred_bg = load_binary_mask(pred_path)
        gt_bg = load_binary_mask(gt_path)
        pred_occ, gt_occ = 1 - pred_bg, 1 - gt_bg
        result.ious[prompt].append(iou(pred_occ, gt_occ))
        result.bg_ious[prompt].append(iou(pred_bg, gt_bg))
        result.dice[prompt].append(dice_loss(pred_occ, gt_occ))
    return result
