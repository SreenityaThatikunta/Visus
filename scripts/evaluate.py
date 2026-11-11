"""
Compute occlusion IoU, background IoU and Dice loss for predicted masks.

Single run:
    python scripts/evaluate.py --dataset lfw --pred outputs/lfw --gt data/lfw/gt

Compare several checkpoints (one sub-folder of predictions per checkpoint,
folder names containing `epoch_<N>`), and save plots:
    python scripts/evaluate.py --dataset celeba --pred outputs/celeba_sweep \
                               --gt data/celeba/gt --sweep --plots docs/figures

Ground-truth layout: `<gt>/glasses/...` and `<gt>/mask/...`
  celeba: files named by CelebA id, e.g. `glasses/002037.jpg`
  lfw:    files named like the input image, e.g. `mask/Aaron_Guiel_0001_mask.jpg`
"""
import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visus.evaluation import evaluate_predictions  # noqa: E402


def epoch_of(name: str):
    m = re.search(r"epoch_(\d+)", name)
    return int(m.group(1)) if m else None


def print_row(name, r):
    print(f"{name:<40} {r.miou('glasses'):>8.4f} {r.miou('mask'):>8.4f} {r.miou():>8.4f} "
          f"{r.background_iou():>8.4f} {r.two_class_miou():>8.4f} {r.mean_dice():>8.4f} {r.count:>5}")


def plot_sweep(results: dict, out_dir: Path, dataset: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted((epoch_of(n), r) for n, r in results.items() if epoch_of(n) is not None)
    if not rows:
        return
    epochs = [e for e, _ in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    for prompt, marker in (("glasses", "o"), ("mask", "s"), (None, "D")):
        ax.plot(epochs, [r.miou(prompt) for _, r in rows], marker=marker,
                linewidth=2, label=(prompt or "overall").capitalize())
    ax.set(xlabel="Epoch", ylabel="Occlusion IoU", title=f"Occlusion IoU vs. epoch ({dataset})", ylim=(0, 1))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(out_dir / f"miou_vs_epoch_{dataset}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, [r.mean_dice() for _, r in rows], marker="o", linewidth=2)
    ax.set(xlabel="Epoch", ylabel="Dice loss", title=f"Dice loss vs. epoch ({dataset})")
    ax.grid(alpha=0.3)
    fig.savefig(out_dir / f"dice_vs_epoch_{dataset}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plots saved to {out_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=("celeba", "lfw"), required=True)
    p.add_argument("--pred", required=True, help="folder of *_mask_binary.png predictions")
    p.add_argument("--gt", required=True, help="ground-truth folder with glasses/ and mask/")
    p.add_argument("--sweep", action="store_true", help="--pred holds one sub-folder per checkpoint")
    p.add_argument("--plots", default=None, help="save sweep plots to this folder")
    a = p.parse_args()

    pred = Path(a.pred)
    runs = sorted(d for d in pred.iterdir() if d.is_dir()) if a.sweep else [pred]

    print("IoU of the occluded region (glasses / mask / both), background IoU, two-class mIoU, Dice loss")
    print(f"{'run':<40} {'glasses':>8} {'mask':>8} {'occ-IoU':>8} {'bg-IoU':>8} {'2cls':>8} {'dice':>8} {'n':>5}")
    print("-" * 100)
    results = {}
    for run in runs:
        r = evaluate_predictions(run, a.gt, a.dataset)
        if r.count == 0:
            print(f"{run.name:<40} no matching predictions")
            continue
        results[run.name] = r
        print_row(run.name, r)

    if a.sweep and results:
        best = max(results, key=lambda n: results[n].miou())
        print(f"\nBest: {best}  (occlusion IoU {results[best].miou():.4f})")
        if a.plots:
            plot_sweep(results, Path(a.plots), a.dataset)


if __name__ == "__main__":
    main()
