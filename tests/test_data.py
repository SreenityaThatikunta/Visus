import csv

import numpy as np
import pytest
import torch
from PIL import Image

from visus.data import EVAL_PROMPTS, OcclusionDataset, PairDataset, collate

FIELDS = ["id", "split", "face_found", "has_glasses", "has_mask", "glasses_style", "lens_type",
          "mask_type", "mask_pattern", "mask_color", "glasses_px", "mask_px", "error"]


@pytest.fixture
def dataset_dir(tmp_path):
    """Three 512x512 images: glasses only, mask only, both."""
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    rows = []
    for i, (g, m) in enumerate([(1, 0), (0, 1), (1, 1)]):
        id_ = f"{i:05d}"
        Image.fromarray(np.full((512, 512, 3), 128, np.uint8)).save(tmp_path / "images" / f"{id_}.jpg")
        for kind, on, box in (("glasses", g, (150, 200, 350, 260)), ("mask", m, (160, 300, 360, 470))):
            a = np.zeros((512, 512), np.uint8)
            if on:
                x0, y0, x1, y1 = box
                a[y0:y1, x0:x1] = 255
            Image.fromarray(a).save(tmp_path / "masks" / f"{id_}_{kind}.png")
        rows.append({"id": id_, "split": "test", "face_found": 1, "has_glasses": g, "has_mask": m,
                     "glasses_px": 200 * 60 * g, "mask_px": 200 * 170 * m})
    with open(tmp_path / "meta.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS, restval="")
        w.writeheader()
        w.writerows(rows)
    return tmp_path


def test_eval_enumerates_every_pair(dataset_dir):
    ds = OcclusionDataset(dataset_dir, "test")
    assert len(ds) == 6                                  # 3 images x 2 prompts
    items = [ds[i] for i in range(len(ds))]
    assert sum(it["present"] for it in items) == 4
    for it in items:
        assert it["image"].shape == (3, 224, 224) and it["mask"].shape == (1, 224, 224)
        assert it["text"] == EVAL_PROMPTS[it["kind"]]
        assert (it["mask"].sum() > 0) == it["present"]   # absent object -> empty target


def test_training_negative_prompts(dataset_dir, tmp_path):
    # relabel everything as train, then always ask for the absent object
    meta = (dataset_dir / "meta.csv").read_text().replace(",test,", ",train,")
    (dataset_dir / "meta.csv").write_text(meta)
    ds = OcclusionDataset(dataset_dir, "train", negative_prob=1.0)
    assert len(ds) == 3
    for i in range(2):                                   # the two single-object images
        it = ds[i]
        assert not it["present"] and it["mask"].sum() == 0
    assert ds[2]["present"]                              # both objects: nothing is absent


def test_collate_and_legacy_masks(dataset_dir, tmp_path):
    # legacy ground truth marks the occlusion in black
    gt = np.full((512, 512), 255, np.uint8)
    gt[150:260, 150:350] = 0
    Image.fromarray(gt).save(tmp_path / "legacy_gt.png")
    ds = PairDataset([(str(dataset_dir / "images" / "00000.jpg"), "glasses", str(tmp_path / "legacy_gt.png"))],
                     occluded_is_white=False)
    b = collate([ds[0], ds[0]])
    assert b["images"].shape == (2, 3, 224, 224) and b["texts"].shape == (2, 77)
    frac = b["masks"][0].mean().item()
    assert 0.05 < frac < 0.12                            # ~110*200 / 512^2 = 8.4% occluded
    assert b["present"].dtype == torch.bool
