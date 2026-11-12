"""
Kaggle GPU job: sanity checks -> train -> evaluate. Push with `kaggle/push.sh train`
(after `kaggle/push.sh code`); inputs are attached as Kaggle sources:

  * sreenityathatikunta/visus-code            this repo as visus_code.tar
  * sreenityathatikunta/visus-legacy-eval     old LFW/CelebA test sets + visus_epoch50.pt
  * kernel output of visus-generate-occlusions: visus_occlusion_v1.tar

Outputs in /kaggle/working: run_v2/ (best.pt, last.pt, history.json, training_curves.png,
samples/), results_v2.json, results_legacy.json, sanity.json.
"""
import json
import os
import subprocess
import sys
import tarfile
import time

QUICK = False          # True: 1 epoch on 2 000 images, to shake out bugs cheaply
EPOCHS = 20
BATCH = 64
WORK = "/tmp/work"
OUT = "/kaggle/working"


def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def locate(tar_name, marker, dest):
    """Return a folder holding `marker` (a relative path). Kaggle unpacks .tar files uploaded
    as datasets, but not those in notebook outputs, so accept either form."""
    t = time.time()
    for root, dirs, files in os.walk("/kaggle/input"):
        if tar_name in files:
            os.makedirs(dest, exist_ok=True)
            with tarfile.open(os.path.join(root, tar_name)) as tar:
                tar.extractall(dest)
            print(f"extracted {tar_name} in {time.time() - t:.0f}s", flush=True)
            base = dest
            break
        if os.path.exists(os.path.join(root, marker)):
            base = root
            print(f"found {marker} under {root}", flush=True)
            break
    else:
        sh("find /kaggle/input -maxdepth 4 | head -50")
        raise FileNotFoundError(f"neither {tar_name} nor {marker} under /kaggle/input")
    found = os.path.join(base, marker)
    if not os.path.exists(found):
        raise FileNotFoundError(found)
    return base


def sanity(repo, data):
    """Gradients reach the unfrozen CLIP blocks, frozen ones stay put, and 6 images can be overfit."""
    import random
    import torch
    sys.path.insert(0, repo)
    from visus.data import OcclusionDataset, make_loader
    from visus.evaluation import evaluate_model
    from visus.losses import BCEDiceLoss
    from visus.models import ModelConfig, build_model_and_optimizer

    torch.manual_seed(0), random.seed(0)
    model, opt = build_model_and_optimizer(ModelConfig(), learning_rate=1e-3, clip_learning_rate=1e-5, device="cuda")
    blocks = model.clip_model.visual.transformer.resblocks
    ds = OcclusionDataset(data, "train", augment=False, max_samples=6, negative_prob=0.0)
    batch = next(iter(make_loader(ds, 6, shuffle=False, num_workers=0)))
    imgs, txt, msk = batch["images"].cuda(), batch["texts"].cuda(), batch["masks"].cuda()
    crit = BCEDiceLoss()
    frozen0 = blocks[0].mlp.c_fc.weight.detach().clone()
    tuned0 = blocks[-2].mlp.c_fc.weight.detach().clone()
    report = {}
    for step in range(150):
        model.train(), opt.zero_grad()
        loss = crit(model(imgs, txt)["logits"], msk)
        loss.backward()
        if step == 0:
            report["grad_norm_last3_blocks"] = [
                None if b.ln_1.weight.grad is None else float(b.ln_1.weight.grad.norm()) for b in blocks[-3:]]
            report["grad_frozen_block0"] = blocks[0].ln_1.weight.grad is not None
            report["loss_start"] = float(loss)
        opt.step()
    report["loss_end"] = float(loss)
    report["tuned_block_changed"] = not torch.equal(tuned0, blocks[-2].mlp.c_fc.weight)
    report["frozen_block_unchanged"] = torch.equal(frozen0, blocks[0].mlp.c_fc.weight)
    ev = OcclusionDataset(data, "train", augment=False, max_samples=6)
    report["overfit"] = evaluate_model(model, make_loader(ev, 12, shuffle=False, num_workers=0), "cuda").best()
    model.cleanup()
    print("SANITY", json.dumps(report, indent=2), flush=True)
    with open(f"{OUT}/sanity.json", "w") as fh:
        json.dump(report, fh, indent=2)
    ok = (all(g for g in report["grad_norm_last3_blocks"]) and not report["grad_frozen_block0"]
          and report["tuned_block_changed"] and report["frozen_block_unchanged"]
          and report["loss_end"] < 0.5 * report["loss_start"])
    if not ok:
        raise SystemExit("sanity checks failed, not training")
    del model, opt
    torch.cuda.empty_cache()


def main():
    t0 = time.time()
    os.makedirs(WORK, exist_ok=True)
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
    sh(f"{sys.executable} -m pip install -q ftfy regex git+https://github.com/openai/CLIP.git")
    # code is copied to a writable place (Kaggle inputs are read-only; train.py writes nothing there,
    # but Python may want to write __pycache__)
    code = locate("visus_code.tar", "visus/__init__.py", f"{WORK}/code")
    repo = f"{WORK}/repo"
    if code != repo:
        sh(f"rm -rf {repo} && cp -r {code} {repo}")
    legacy = os.path.join(locate("visus_legacy.tar", "legacy/visus_epoch50.pt", WORK), "legacy")
    data = os.path.join(locate("visus_occlusion_v1.tar", "visus_occlusion_v1/meta.csv", WORK),
                        "visus_occlusion_v1")
    print("repo", repo, "| data", data, "| legacy", legacy, flush=True)
    os.chdir(repo)

    sanity(repo, data)

    run = f"{OUT}/run_v2"
    quick = "--epochs 1 --max-samples 2000" if QUICK else f"--epochs {EPOCHS}"
    sh(f"{sys.executable} scripts/train.py --data {data} --output {run} {quick} "
       f"--batch-size {BATCH} --workers 4")

    common = (f"--data {data} --split test --lfw-images {legacy}/lfw/images --lfw-gt {legacy}/lfw/gt "
              f"--celeba-images {legacy}/celeba/images --celeba-gt {legacy}/celeba/gt --workers 4")
    sh(f"{sys.executable} scripts/evaluate.py --checkpoint {run}/checkpoints/best.pt {common} "
       f"--json {OUT}/results_v2.json")
    sh(f"{sys.executable} scripts/evaluate.py --checkpoint {legacy}/visus_epoch50.pt {common} "
       f"--json {OUT}/results_legacy.json")
    print(f"ALL DONE in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
