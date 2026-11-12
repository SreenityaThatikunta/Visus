"""
Metrics.

Convention everywhere in Visus v2: a predicted mask is 1/white where the prompted object
occludes the face. The old LFW / CelebA test masks are the opposite (occlusion black);
callers say which convention a ground-truth folder uses.

Reported numbers
  * IoU (present)   mean per-image IoU of the occluded region, over (image, prompt) pairs
                    where the object is there. This is the headline number. An empty
                    prediction scores 0, so it cannot be gamed by predicting nothing.
  * clean rate      over pairs where the prompted object is NOT in the image (a mask-only
                    face asked for "glasses"): share of predictions that are empty
                    (< 0.5% of the image). Tests that the model listens to the prompt.
  * overall         mean over all pairs, counting an absent pair as 1 if clean else 0.
                    (An empty mask scores ~0.4 here, so it is reported but not optimised.)
  * score           sqrt(IoU x clean rate); picks the threshold and the best checkpoint.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .data import KINDS

DEFAULT_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
CLEAN_FRACTION = 0.005


# ---------------------------------------------------------------------------
# In-memory evaluation of a model on a DataLoader
# ---------------------------------------------------------------------------

@dataclass
class Metrics:
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS
    # per threshold -> kind -> list of IoUs on present pairs / clean flags on absent pairs
    iou: dict = field(default_factory=dict)
    clean: dict = field(default_factory=dict)

    def __post_init__(self):
        for t in self.thresholds:
            self.iou[t] = {k: [] for k in KINDS}
            self.clean[t] = {k: [] for k in KINDS}

    def update(self, probs: torch.Tensor, targets: torch.Tensor, kinds: list[str], present: torch.Tensor):
        probs, targets = probs.detach().float().cpu(), targets.detach().cpu() > 0.5
        hw = probs.shape[-1] * probs.shape[-2]
        for t in self.thresholds:
            pred = probs > t
            inter = (pred & targets).flatten(1).sum(1).float()
            union = (pred | targets).flatten(1).sum(1).float()
            area = pred.flatten(1).sum(1).float()
            for i, kind in enumerate(kinds):
                if bool(present[i]):
                    self.iou[t][kind].append(float(inter[i] / union[i]) if union[i] > 0 else 1.0)
                else:
                    self.clean[t][kind].append(float(area[i] / hw < CLEAN_FRACTION))

    @staticmethod
    def _mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    def summary(self, t: float) -> dict:
        iou = {k: self._mean(v) for k, v in self.iou[t].items()}
        clean = {k: self._mean(v) for k, v in self.clean[t].items()}
        pres = [x for v in self.iou[t].values() for x in v]
        absn = [x for v in self.clean[t].values() for x in v]
        return {
            "threshold": t,
            "iou_glasses": iou["glasses"], "iou_mask": iou["mask"],
            "iou": self._mean([iou[k] for k in KINDS if not np.isnan(iou[k])]),
            "clean_rate": self._mean(absn),
            "overall": self._mean(pres + absn),
            "n_present": len(pres), "n_absent": len(absn),
        }

    @staticmethod
    def score(row: dict) -> float:
        """sqrt(IoU x clean rate): 0 for an always-empty model AND for one that ignores the prompt.
        Used to pick the threshold and the best checkpoint."""
        iou, clean = row["iou"], row["clean_rate"]
        if np.isnan(clean):
            return iou
        return float(np.sqrt(max(iou, 0) * max(clean, 0)))

    def best(self, key: str = "score") -> dict:
        if key == "score":
            rows = [dict(self.summary(t)) for t in self.thresholds]
            for r in rows:
                r["score"] = self.score(r)
            return max(rows, key=lambda r: r["score"])
        rows = [self.summary(t) for t in self.thresholds]
        rows = [r for r in rows if not np.isnan(r[key])] or rows
        return max(rows, key=lambda r: r[key])


@torch.no_grad()
def evaluate_model(model, loader, device: str, thresholds=DEFAULT_THRESHOLDS, amp: bool = False) -> Metrics:
    model.eval()
    m = Metrics(tuple(thresholds))
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and device == "cuda"):
            out = model(batch["images"].to(device), batch["texts"].to(device))
        m.update(out["probs"], batch["masks"], batch["kinds"], batch["present"])
    return m


# ---------------------------------------------------------------------------
# Legacy test sets (LFW / CelebA subsets from the original project; occlusion = black)
# ---------------------------------------------------------------------------

def legacy_lfw_pairs(images_dir: str | Path, gt_dir: str | Path) -> list[tuple[str, str, str]]:
    """images: <Name_0001>_<kind>.jpg ; gt: <gt>/<kind>/<same name>.jpg"""
    pairs = []
    for img in sorted(Path(images_dir).glob("*.jpg")):
        for kind in KINDS:
            if img.stem.lower().endswith(f"_{kind}"):
                gt = Path(gt_dir) / kind / img.name
                if gt.exists():
                    pairs.append((str(img), kind, str(gt)))
    return pairs


def legacy_celeba_pairs(images_dir: str | Path, gt_dir: str | Path) -> list[tuple[str, str, str]]:
    """images: <id>_<kind>_pasted.jpg ; gt: <id>_<kind>_pasted_<kind>.jpg (filtered_masks_output)"""
    pairs = []
    for img in sorted(Path(images_dir).glob("*_pasted.jpg")):
        m = re.match(r"(\d+)_(glasses|mask)_pasted$", img.stem)
        if not m:
            continue
        gt = Path(gt_dir) / f"{img.stem}_{m.group(2)}.jpg"
        if gt.exists():
            pairs.append((str(img), m.group(2), str(gt)))
    return pairs


# ---------------------------------------------------------------------------
# File-based evaluation (scripts/evaluate.py): predictions written by scripts/predict.py
# ---------------------------------------------------------------------------

import cv2  # noqa: E402

MASK_SIZE = (224, 224)
SUFFIX = "_mask_binary"


def load_occlusion_mask(path: str | Path, occluded_is_white: bool, size=MASK_SIZE) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load: {path}")
    if size is not None and img.shape[::-1] != size:
        img = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
    white = img >= 128
    return (white if occluded_is_white else ~white).astype(np.uint8)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    union = np.logical_or(pred, gt).sum()
    return 1.0 if union == 0 else float(np.logical_and(pred, gt).sum() / union)


def _parse_prediction(name: str) -> tuple[str, str] | None:
    """'<stem>_<prompt>_mask_binary' -> (stem, kind)."""
    name = name.lower()
    for prompt, kind in (("glasses", "glasses"), ("face-mask", "mask"), ("mask", "mask")):
        tail = f"_{prompt}{SUFFIX}"
        if name.endswith(tail):
            return name[: -len(tail)], kind
    return None


def build_gt_index(gt_dir: str | Path, dataset: str) -> dict:
    """
    lfw:    <gt>/{glasses,mask}/<Name_0001_kind>.jpg          -> key: lowercase stem
    celeba: <gt>/<id>_<kind>_pasted_<kind>.jpg (or <gt>/<kind>/<id>.jpg) -> key: (id, kind)
    """
    index = {}
    files = [Path(r) / n for r, _, ns in os.walk(gt_dir, followlinks=True) for n in ns]
    for f in files:
        if f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        if dataset == "celeba":
            m = re.match(r"(\d+)_(glasses|mask)_pasted_\2$", f.stem)
            if m:
                index[(m.group(1), m.group(2))] = f
            elif f.parent.name in KINDS:
                index[(f.stem, f.parent.name)] = f
        else:
            index[f.stem.lower()] = f
    return index


@dataclass
class FileEvalResult:
    ious: dict = field(default_factory=lambda: {k: [] for k in KINDS})

    def miou(self, kind: str | None = None) -> float:
        vals = self.ious[kind] if kind else self.ious["glasses"] + self.ious["mask"]
        return float(np.mean(vals)) if vals else float("nan")

    @property
    def count(self) -> int:
        return sum(len(v) for v in self.ious.values())


def evaluate_predictions(pred_dir, gt_dir, dataset: str, gt_occluded_is_white: bool = False) -> FileEvalResult:
    """Predictions are occlusion = white. Old LFW/CelebA ground truth is occlusion = black."""
    index = build_gt_index(gt_dir, dataset)
    result = FileEvalResult()
    for pred_path in sorted(Path(pred_dir).rglob(f"*{SUFFIX}.png")):
        parsed = _parse_prediction(pred_path.stem)
        if parsed is None:
            continue
        stem, kind = parsed
        if dataset == "celeba":
            m = re.match(r"(\d+)", stem)
            key = (m.group(1), kind) if m else None
        else:
            key = stem
        gt = index.get(key)
        if gt is None:
            continue
        pred = load_occlusion_mask(pred_path, occluded_is_white=True)
        result.ious[kind].append(iou(pred, load_occlusion_mask(gt, gt_occluded_is_white)))
    return result
