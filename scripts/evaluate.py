"""
Evaluate a checkpoint.

Score a checkpoint directly (recommended):
    python scripts/evaluate.py --checkpoint runs/v2/checkpoints/best.pt \
        --data data/visus_occlusion_v1 --split test \
        --lfw-images data/legacy/lfw/images --lfw-gt data/legacy/lfw/gt \
        --celeba-images data/legacy/celeba/images --celeba-gt data/legacy/celeba/gt \
        --json results.json

  --data/--split        the synthetic test split: IoU (object present), clean rate
                        (object absent: is the prediction empty?) and overall
  --lfw-* / --celeba-*  the original project's LFW / CelebA test subsets
                        (ground truth: occlusion black, coarse landmark polygons)

Every table also shows the "empty mask" baseline (predict nothing).
The threshold is the one stored in the checkpoint (chosen on validation) unless --threshold.

Score prediction files written by scripts/predict.py:
    python scripts/evaluate.py --pred outputs/lfw --gt data/legacy/lfw/gt --dataset lfw
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from visus.checkpoint import load_model  # noqa: E402
from visus.config import get_device  # noqa: E402
from visus.data import OcclusionDataset, PairDataset, make_loader  # noqa: E402
from visus.evaluation import (Metrics, evaluate_model, evaluate_predictions,  # noqa: E402
                              legacy_celeba_pairs, legacy_lfw_pairs)


def empty_baseline(loader) -> dict:
    m = Metrics((0.5,))
    for b in loader:
        m.update(torch.zeros_like(b["masks"]), b["masks"], b["kinds"], b["present"])
    return m.summary(0.5)


def fmt(name, r):
    def f(x):
        return "   -  " if x != x else f"{x:6.3f}"
    return (f"{name:<22} {f(r['iou_glasses'])} {f(r['iou_mask'])} {f(r['iou'])} "
            f"{f(r['clean_rate'])} {f(r['overall'])}   n={r['n_present']}+{r['n_absent']}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--data", help="synthetic dataset folder")
    p.add_argument("--split", default="test")
    p.add_argument("--lfw-images"), p.add_argument("--lfw-gt")
    p.add_argument("--celeba-images"), p.add_argument("--celeba-gt")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default=None)
    p.add_argument("--json", help="also write results here")
    # file mode
    p.add_argument("--pred"), p.add_argument("--gt"), p.add_argument("--dataset", choices=("lfw", "celeba"))
    p.add_argument("--gt-white", action="store_true", help="ground truth marks occlusion in white")
    a = p.parse_args()

    if a.pred:
        r = evaluate_predictions(a.pred, a.gt, a.dataset, gt_occluded_is_white=a.gt_white)
        print(f"occlusion IoU  glasses {r.miou('glasses'):.4f}  mask {r.miou('mask'):.4f}  "
              f"all {r.miou():.4f}  (n={r.count})")
        return

    device = get_device(a.device)
    model, state = load_model(a.checkpoint, device)
    thr = a.threshold if a.threshold is not None else state.get("threshold", 0.1 if "model_config" not in state else 0.5)
    print(f"threshold {thr}")

    sets = {}
    if a.data:
        sets[f"synthetic {a.split}"] = OcclusionDataset(a.data, a.split)
    if a.lfw_images:
        sets["LFW (legacy)"] = PairDataset(legacy_lfw_pairs(a.lfw_images, a.lfw_gt), occluded_is_white=False)
    if a.celeba_images:
        sets["CelebA (legacy)"] = PairDataset(legacy_celeba_pairs(a.celeba_images, a.celeba_gt), occluded_is_white=False)

    results = {"checkpoint": a.checkpoint, "threshold": thr, "sets": {}}
    print(f"\n{'':<22} {'IoU-gl':>6} {'IoU-mk':>6} {'IoU':>6} {'clean':>6} {'overall':>6}")
    for name, ds in sets.items():
        loader = make_loader(ds, a.batch_size, shuffle=False, num_workers=a.workers)
        m = evaluate_model(model, loader, device, thresholds=(thr,), amp=device == "cuda")
        r, base = m.summary(thr), empty_baseline(loader)
        print(f"-- {name} ({len(ds)} pairs)")
        print(fmt("  model", r))
        print(fmt("  empty-mask baseline", base))
        results["sets"][name] = {"model": r, "empty_baseline": base}

    if a.json:
        with open(a.json, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwrote {a.json}")
    model.cleanup()


if __name__ == "__main__":
    main()
