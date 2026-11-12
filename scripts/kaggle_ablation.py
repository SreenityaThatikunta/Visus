"""
Kaggle GPU job: analysis of the trained v2 model, then ablations. Push with `kaggle/push.sh ablate`.

1. scripts/analyze.py on the v2 checkpoint (output of the visus-train notebook): per-category
   IoU, prompt robustness, figures.
2. Ablations: the v2 model and three variants, each trained the same short way (6 epochs on
   10 000 training images) and scored on the full test split:
     full          v2 as released
     frozen_clip   no CLIP fine-tuning (--trainable-blocks 0)
     no_rgb        head sees only the CLIP attention map (--no-rgb)
     v1_text       v1's 3-number text features instead of the CLIP text embedding

Outputs in /kaggle/working: analysis/ and ablation/<name>/{history.json,test.json}, ablation.json.
"""
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time

EPOCHS = 6
MAX_SAMPLES = 10000
RUNS = {
    "full": "",
    "frozen_clip": "--trainable-blocks 0",
    "no_rgb": "--no-rgb",
    "v1_text": "--text-mode attention",
}
WORK = "/tmp/work"
OUT = "/kaggle/working"


def sh(cmd, check=True):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=check).returncode


def locate(tar_name, marker, dest):
    """Folder holding `marker`; Kaggle unpacks .tar files uploaded as datasets but not notebook outputs."""
    for root, dirs, files in os.walk("/kaggle/input"):
        if tar_name in files:
            os.makedirs(dest, exist_ok=True)
            with tarfile.open(os.path.join(root, tar_name)) as tar:
                tar.extractall(dest)
            return dest
        if os.path.exists(os.path.join(root, marker)):
            return root
    sh("find /kaggle/input -maxdepth 5 | head -60")
    raise FileNotFoundError(marker)


def main():
    t0 = time.time()
    os.makedirs(WORK, exist_ok=True)
    sh(f"{sys.executable} -m pip install -q ftfy regex git+https://github.com/openai/CLIP.git")
    code = locate("visus_code.tar", "visus/__init__.py", f"{WORK}/code")
    repo = f"{WORK}/repo"
    shutil.copytree(code, repo, dirs_exist_ok=True)
    legacy = os.path.join(locate("visus_legacy.tar", "legacy/visus_epoch50.pt", WORK), "legacy")
    data = os.path.join(locate("visus_occlusion_v1.tar", "visus_occlusion_v1/meta.csv", WORK), "visus_occlusion_v1")
    ckpt = os.path.join(locate("-", "run_v2/checkpoints/best.pt", WORK), "run_v2/checkpoints/best.pt")
    print("data", data, "| checkpoint", ckpt, flush=True)
    os.chdir(repo)
    py = sys.executable

    sh(f"{py} scripts/analyze.py --checkpoint {ckpt} --data {data} --lfw-images {legacy}/lfw/images "
       f"--lfw-gt {legacy}/lfw/gt --output {OUT}/analysis", check=False)

    summary = {}
    for name, flags in RUNS.items():
        run = f"{OUT}/ablation/{name}"
        rc = sh(f"{py} scripts/train.py --data {data} --output {run} --epochs {EPOCHS} "
                f"--max-samples {MAX_SAMPLES} --batch-size 64 --workers 4 {flags}", check=False)
        if rc == 0:
            sh(f"{py} scripts/evaluate.py --checkpoint {run}/checkpoints/best.pt --data {data} --split test "
               f"--workers 4 --json {run}/test.json", check=False)
        if os.path.exists(f"{run}/test.json"):
            with open(f"{run}/test.json") as fh:
                summary[name] = json.load(fh)["sets"]["synthetic test"]["model"]
        shutil.rmtree(f"{run}/checkpoints", ignore_errors=True)   # keep the output small
        with open(f"{OUT}/ablation.json", "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"[{(time.time() - t0) / 60:.0f} min] {name}: {summary.get(name)}", flush=True)
    print(f"ALL DONE in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
