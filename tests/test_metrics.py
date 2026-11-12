import math

import torch

from visus.evaluation import Metrics
from visus.losses import BCEDiceLoss, DiceLoss


def _square(size=32, lo=8, hi=24):
    m = torch.zeros(1, 1, size, size)
    m[..., lo:hi, lo:hi] = 1
    return m


def _batch():
    """Two present pairs (glasses, mask) and two absent pairs (empty targets)."""
    gt = torch.cat([_square(), _square(), torch.zeros(2, 1, 32, 32)])
    kinds = ["glasses", "mask", "glasses", "mask"]
    present = torch.tensor([True, True, False, False])
    return gt, kinds, present


def test_perfect_prediction():
    gt, kinds, present = _batch()
    m = Metrics((0.5,))
    m.update(gt.clone(), gt, kinds, present)
    r = m.summary(0.5)
    assert r["iou"] == 1.0 and r["clean_rate"] == 1.0
    assert Metrics.score(r) == 1.0
    assert (r["n_present"], r["n_absent"]) == (2, 2)


def test_empty_prediction_scores_zero():
    gt, kinds, present = _batch()
    m = Metrics((0.5,))
    m.update(torch.zeros_like(gt), gt, kinds, present)
    r = m.summary(0.5)
    assert r["iou"] == 0.0 and r["clean_rate"] == 1.0
    assert Metrics.score(r) == 0.0
    # "overall" alone would reward predicting nothing; that is why it is not the selection metric
    assert r["overall"] == 0.5


def test_ignoring_the_prompt_scores_zero():
    gt, kinds, present = _batch()
    pred = torch.cat([_square(), _square(), _square(), _square()])  # segments something every time
    m = Metrics((0.5,))
    m.update(pred, gt, kinds, present)
    r = m.summary(0.5)
    assert r["iou"] == 1.0 and r["clean_rate"] == 0.0
    assert Metrics.score(r) == 0.0


def test_partial_overlap_iou():
    gt = _square(lo=0, hi=16)            # 16x16 = 256 px
    pred = _square(lo=8, hi=24)          # overlap 8x8 = 64, union 256 + 256 - 64 = 448
    m = Metrics((0.5,))
    m.update(pred, gt, ["mask"], torch.tensor([True]))
    assert math.isclose(m.summary(0.5)["iou_mask"], 64 / 448, rel_tol=1e-6)


def test_best_threshold_is_picked_by_score():
    gt, kinds, present = _batch()
    probs = gt * 0.7                      # only thresholds below 0.7 recover the objects
    m = Metrics((0.5, 0.8))
    m.update(probs, gt, kinds, present)
    assert m.best("score")["threshold"] == 0.5


def test_losses():
    target = _square()
    good, bad = (target * 20 - 10), (10 - target * 20)
    loss = BCEDiceLoss()
    assert loss(good, target) < 0.05 < loss(bad, target)
    assert DiceLoss()(target, target) < 1e-6
    assert loss(torch.full_like(target, -12.0), torch.zeros_like(target)) < 0.05  # empty target is fine
