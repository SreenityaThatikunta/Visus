"""
Per-sample analysis and figures for a trained checkpoint (used by the report and the README).

    python scripts/analyze.py --checkpoint checkpoints/visus_v2.pt --data data/visus_occlusion_v1 \
        --lfw-images data/legacy/lfw/images --lfw-gt data/legacy/lfw/gt --output runs/analysis

Writes analysis.json with
  * IoU broken down by glasses style, lens type, mask type, mask texture (all textured masks in
    the test split use the 5 textures held out of training) and "alone" vs "with the other object";
  * prompt robustness: IoU when the object is named with other words, some never used in training;
  * unrelated prompts ("hat", "a cat", ...): how often the prediction is empty;
and figures hero.png, test_examples.png, failures.png, lfw_examples.png.
"""
import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import Subset  # noqa: E402

import visus.data as vdata  # noqa: E402
from visus.checkpoint import load_model  # noqa: E402
from visus.config import get_device  # noqa: E402
from visus.data import OcclusionDataset, PairDataset, make_loader  # noqa: E402
from visus.evaluation import CLEAN_FRACTION, legacy_lfw_pairs  # noqa: E402
from visus.visualize import denormalize  # noqa: E402

# Prompts for the robustness test. Those marked * never appear in training (visus.data.TRAIN_PROMPTS).
PROMPTS = {
    "glasses": ["glasses", "sunglasses", "eyewear", "reading glasses*", "aviator sunglasses*", "eyeglass frames*"],
    "mask": ["face mask", "mask", "surgical mask", "N95 respirator*", "covid mask*", "face covering*"],
}
UNRELATED = ["hat", "beard", "a cat", "a red car"]
PRED_COLOR, GT_COLOR = (255, 60, 60), (40, 200, 120)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    for b in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            out = model(b["images"].to(device), b["texts"].to(device))
        yield b, {k: out[k].float().cpu() for k in ("probs", "attention_maps")}


def per_pair(model, ds, device, thr, workers):
    rows = []
    for b, out in predict(model, make_loader(ds, 64, shuffle=False, num_workers=workers), device):
        pred, gt = out["probs"][:, 0] > thr, b["masks"][:, 0] > 0.5
        inter = (pred & gt).flatten(1).sum(1).float()
        union = (pred | gt).flatten(1).sum(1).float()
        area = pred.flatten(1).float().mean(1)
        for i, id_ in enumerate(b["ids"]):
            rows.append({"id": id_, "kind": b["kinds"][i], "present": bool(b["present"][i]),
                         "iou": float(inter[i] / union[i]) if union[i] > 0 else 1.0,
                         "area": float(area[i]), "empty": float(area[i]) < CLEAN_FRACTION})
    return rows


def summarize(rows):
    pres = [r["iou"] for r in rows if r["present"]]
    absn = [r["empty"] for r in rows if not r["present"]]
    return {"iou": float(np.mean(pres)) if pres else None, "n_present": len(pres),
            "clean_rate": float(np.mean(absn)) if absn else None, "n_absent": len(absn)}


def breakdown(rows, meta):
    groups = defaultdict(list)
    for r in rows:
        if not r["present"]:
            continue
        m = meta[r["id"]]
        if r["kind"] == "glasses":
            groups[f"glasses style: {m['glasses_style']}"].append(r["iou"])
            groups[f"lens: {m['lens_type']}"].append(r["iou"])
        else:
            groups[f"mask type: {m['mask_type']}"].append(r["iou"])
            tex = ("held-out texture" if m["mask_pattern"] else
                   "solid colour" if m["mask_color"] else "default look")
            groups[f"mask texture: {tex}"].append(r["iou"])
        both = m["has_glasses"] == "1" and m["has_mask"] == "1"
        groups[f"{r['kind']}: {'with the other object' if both else 'alone'}"].append(r["iou"])
    return {k: {"iou": float(np.mean(v)), "n": len(v)} for k, v in sorted(groups.items())}


def with_prompt(ds, kinds, prompt):
    """Point visus.data's evaluation prompt for `kinds` at `prompt` (dataset workers are forked
    after this, so they see it too). Returns the old values."""
    old = {k: vdata.EVAL_PROMPTS[k] for k in kinds}
    for k in kinds:
        vdata.EVAL_PROMPTS[k] = prompt
    return old


def prompt_robustness(model, ds, device, thr, workers, n=1000):
    res = {}
    rng = random.Random(0)
    for kind, prompts in PROMPTS.items():
        idx = [i for i, (s, k) in enumerate(ds.items) if k == kind and k in s.kinds_present]
        sub = Subset(ds, sorted(rng.sample(idx, min(n, len(idx)))))
        for p in prompts:
            old = with_prompt(ds, [kind], p.rstrip("*"))
            rows = per_pair(model, sub, device, thr, workers)
            vdata.EVAL_PROMPTS.update(old)
            res[f"{kind}: {p}"] = {"iou": float(np.mean([r["iou"] for r in rows])), "n": len(rows),
                                   "unseen_in_training": p.endswith("*")}
            print(f"  {kind:8s} {p:22s} IoU {res[f'{kind}: {p}']['iou']:.3f}", flush=True)
    sub = Subset(ds, sorted(rng.sample(range(len(ds)), min(n, len(ds)))))
    for p in UNRELATED:
        old = with_prompt(ds, list(vdata.KINDS), p)
        rows = per_pair(model, sub, device, thr, workers)
        vdata.EVAL_PROMPTS.update(old)
        res[f"unrelated: {p}"] = {"empty_rate": float(np.mean([r["empty"] for r in rows])),
                                  "mean_area": float(np.mean([r["area"] for r in rows])), "n": len(rows)}
        print(f"  unrelated {p:20s} empty {res[f'unrelated: {p}']['empty_rate']:.3f}", flush=True)
    return res


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def overlay(img, m, color, a=0.55):
    out = img.astype(np.float32).copy()
    m = np.asarray(m).astype(bool)
    out[m] = (1 - a) * out[m] + a * np.array(color, np.float32)
    return out.astype(np.uint8)


def collect(model, ds, idxs, device):
    items = []
    for b, out in predict(model, make_loader(Subset(ds, idxs), 64, shuffle=False, num_workers=0), device):
        imgs = denormalize(b["images"])
        for i in range(len(b["ids"])):
            items.append({"img": imgs[i], "attn": out["attention_maps"][i, 0].numpy(),
                          "prob": out["probs"][i, 0].numpy(), "gt": b["masks"][i, 0].numpy(),
                          "prompt": b["prompts"][i], "present": bool(b["present"][i]), "id": b["ids"][i]})
    return items


def _clean(ax):
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def grid(items, path, thr, captions=None):
    n = len(items)
    fig, ax = plt.subplots(4, n, figsize=(1.9 * n, 8.2))
    ax = np.array(ax).reshape(4, n)
    labels = ["input", "CLIP attention", "prediction", "ground truth"]
    for j, it in enumerate(items):
        panels = [it["img"], it["attn"], overlay(it["img"], it["prob"] > thr, PRED_COLOR),
                  overlay(it["img"], it["gt"] > 0.5, GT_COLOR)]
        for i, p in enumerate(panels):
            ax[i, j].imshow(p, cmap="inferno" if i == 1 else None)
            _clean(ax[i, j])
            if j == 0:
                ax[i, j].set_ylabel(labels[i], fontsize=10)
        ax[0, j].set_title(f'"{it["prompt"]}"' + ("" if it["present"] else "\n(not in image)"), fontsize=9)
        if captions:
            ax[2, j].set_xlabel(captions[j], fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path, flush=True)


def caption(it, thr):
    pred = it["prob"] > thr
    if not it["present"]:
        return "empty ✓" if pred.mean() < CLEAN_FRACTION else f"{pred.mean():.1%} of image ✗"
    gt = it["gt"] > 0.5
    union = (pred | gt).sum()
    return f"IoU {(pred & gt).sum() / union:.2f}" if union else "IoU 1.00"


def hero(model, ds, meta, device, thr, path, n=5):
    """Faces wearing both objects: the prompt alone decides which one is segmented."""
    index = {(s.id, k): i for i, (s, k) in enumerate(ds.items)}
    both = [s.id for s in ds.samples if len(s.kinds_present) == 2
            and meta[s.id]["glasses_style"] and meta[s.id]["mask_type"]]
    ids = random.Random(3).sample(both, n)
    items = collect(model, ds, [index[(i, k)] for i in ids for k in vdata.KINDS], device)
    fig, ax = plt.subplots(3, n, figsize=(2.3 * n, 7.2))
    for j in range(n):
        g, m = items[2 * j], items[2 * j + 1]
        panels = [(g["img"], "input"),
                  (overlay(g["img"], g["prob"] > thr, PRED_COLOR), f'"{g["prompt"]}"'),
                  (overlay(m["img"], m["prob"] > thr, (60, 140, 255)), f'"{m["prompt"]}"')]
        for i, (p, label) in enumerate(panels):
            ax[i, j].imshow(p)
            _clean(ax[i, j])
            if j == 0:
                ax[i, j].set_ylabel(label, fontsize=11)
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--lfw-images"), p.add_argument("--lfw-gt")
    p.add_argument("--output", default="runs/analysis")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default=None)
    a = p.parse_args()
    os.makedirs(a.output, exist_ok=True)

    device = get_device(a.device)
    model, state = load_model(a.checkpoint, device)
    thr = state.get("threshold", 0.5)
    with open(os.path.join(a.data, "meta.csv"), newline="") as fh:
        meta = {r["id"]: r for r in csv.DictReader(fh)}
    ds = OcclusionDataset(a.data, "test")
    print(f"threshold {thr} | test pairs {len(ds)}", flush=True)

    rows = per_pair(model, ds, device, thr, a.workers)
    result = {"checkpoint": a.checkpoint, "threshold": thr, "test": summarize(rows),
              "breakdown": breakdown(rows, meta)}
    for k, v in result["breakdown"].items():
        print(f"  {k:40s} IoU {v['iou']:.3f}  n={v['n']}", flush=True)
    print("prompt robustness", flush=True)
    result["prompts"] = prompt_robustness(model, ds, device, thr, a.workers)
    with open(os.path.join(a.output, "analysis.json"), "w") as fh:
        json.dump(result, fh, indent=2)

    hero(model, ds, meta, device, thr, os.path.join(a.output, "hero.png"))
    rng = random.Random(1)
    present = [i for i, r in enumerate(rows) if r["present"]]
    absent = [i for i, r in enumerate(rows) if not r["present"]]
    picks = rng.sample(present, 6) + rng.sample(absent, 2)
    items = collect(model, ds, picks, device)
    grid(items, os.path.join(a.output, "test_examples.png"), thr, [caption(it, thr) for it in items])

    worst = sorted(present, key=lambda i: rows[i]["iou"])[:6]
    worst += sorted(absent, key=lambda i: -rows[i]["area"])[:2]
    items = collect(model, ds, worst, device)
    grid(items, os.path.join(a.output, "failures.png"), thr, [caption(it, thr) for it in items])

    if a.lfw_images:
        lfw = PairDataset(legacy_lfw_pairs(a.lfw_images, a.lfw_gt), occluded_is_white=False)
        picks = random.Random(2).sample(range(len(lfw)), min(8, len(lfw)))
        items = collect(model, lfw, picks, device)
        grid(items, os.path.join(a.output, "lfw_examples.png"), thr, [caption(it, thr) for it in items])
    model.cleanup()


if __name__ == "__main__":
    main()
