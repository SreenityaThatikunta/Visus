"""
Build the Visus occlusion dataset on Kaggle (CPU notebook, internet on).

Input : 30k clean CelebA-HQ faces (the `Celeba_occlusion_uncropped` folder of a Kaggle dataset).
Output: /kaggle/working/visus_occlusion_v1.tar with
          images/<id>.jpg              512x512 face with synthetic occlusion(s)
          masks/<id>_glasses.png       ground truth, occluded = 255 (white), else 0
          masks/<id>_mask.png          ground truth, occluded = 255 (white), else 0
        plus meta.csv, summary.json and preview.png next to it.

Both ground-truth files exist for every image; one is empty when that occlusion is absent,
so a prompt for an object that is not there has an all-black target.

Occlusions
  * face masks: MaskTheFace (MIT, github.com/aqeelanwar/MaskTheFace) templates, textures and
    colours, warped onto dlib landmarks. Its alpha channel gives an exact per-pixel label.
  * glasses: drawn procedurally from dlib eye landmarks (round / rectangle / wayfarer /
    aviator / cat-eye frames; opaque, mirrored, tinted or clear lenses). The rendered
    footprint is the label.
Mix: 40% glasses only, 40% mask only, 20% both. Labels are exact, unlike the old
landmark-polygon masks.

Split: seeded 3 000-image test, 1 500 val, rest train. A few textures are held out of
train/val and only appear in test.
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import bz2
from multiprocessing import Pool

import numpy as np
import cv2

LIMIT = 0            # >0: only process this many images (smoke test)
SIZE = 512
SEED = 1234
N_TEST, N_VAL = 3000, 1500
WORKERS = os.cpu_count() or 4

WORK = "/tmp/visus"
OUT_DIR = f"{WORK}/visus_occlusion_v1"
MTF_DIR = f"{WORK}/MaskTheFace"
PRED_PATH = f"{WORK}/shape_predictor_68_face_landmarks.dat"
KAGGLE_OUT = "/kaggle/working"


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------

def sh(cmd: str) -> None:
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def setup() -> None:
    os.makedirs(WORK, exist_ok=True)
    try:
        import dlib  # noqa: F401
    except ImportError:
        try:
            sh(f"{sys.executable} -m pip install -q dlib-bin")
        except subprocess.CalledProcessError:
            sh(f"{sys.executable} -m pip install -q dlib")
    for mod in ("imutils", "dotmap"):
        try:
            __import__(mod)
        except ImportError:
            sh(f"{sys.executable} -m pip install -q {mod}")

    if not os.path.isdir(MTF_DIR):
        sh(f"git clone -q https://github.com/aqeelanwar/MaskTheFace {MTF_DIR}")
        sh(f"git -C {MTF_DIR} checkout -q caf4d6c")

    if not os.path.exists(PRED_PATH):
        urls = [
            "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
            "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2",
        ]
        for url in urls:
            try:
                print("downloading", url, flush=True)
                tmp = PRED_PATH + ".bz2"
                urllib.request.urlretrieve(url, tmp)
                with bz2.BZ2File(tmp) as fr, open(PRED_PATH, "wb") as fw:
                    shutil.copyfileobj(fr, fw)
                os.remove(tmp)
                break
            except Exception as e:  # try the next mirror
                print("  failed:", e, flush=True)
        if not os.path.exists(PRED_PATH):
            raise RuntimeError("could not download the dlib landmark model")


def find_faces_dir() -> str:
    name = "Celeba_occlusion_uncropped"
    for root in ("/kaggle/input/datasets/sreenityathatikunta/celeba", "/kaggle/input/celeba"):
        for cand in (f"{root}/{name}/{name}", f"{root}/{name}"):
            if os.path.isdir(cand) and not os.path.isdir(os.path.join(cand, os.listdir(cand)[0])):
                return cand
    raise FileNotFoundError("face images folder not found under /kaggle/input")


# ---------------------------------------------------------------------------
# worker state
# ---------------------------------------------------------------------------

DET = PRED = AF = FACE_UTILS = None


def init_worker() -> None:
    global DET, PRED, AF, FACE_UTILS
    cv2.setNumThreads(1)
    os.chdir(MTF_DIR)                    # MaskTheFace uses relative paths (masks/masks.cfg)
    sys.path.insert(0, MTF_DIR)
    import dlib
    from imutils import face_utils
    from utils import aux_functions as af
    DET = dlib.get_frontal_face_detector()
    PRED = dlib.shape_predictor(PRED_PATH)
    AF, FACE_UTILS = af, face_utils


class MaskArgs:
    def __init__(self, pattern="", pattern_weight=0.5, color="", color_weight=0.5):
        self.pattern, self.pattern_weight = pattern, pattern_weight
        self.color, self.color_weight = color, color_weight


# ---------------------------------------------------------------------------
# face masks (MaskTheFace)
# ---------------------------------------------------------------------------

MASK_TYPES = ["surgical", "surgical_blue", "surgical_green", "N95", "KN95", "cloth", "cloth", "cloth"]
MASK_COLORS = ["#0473e2", "#e54294", "#ff0000", "#000000", "#ffffff", "#2e8b57", "#808080",
               "#f4c430", "#6a5acd", "#8b4513", "#20b2aa", "#ff8c00", "#1e3a5f", "#c0c0c0"]


def apply_face_mask(img, rect, pts, rng, textures):
    landmarks = AF.shape_to_landmarks(pts)
    six, angle = AF.get_six_points(landmarks, img)
    mtype = rng.choice(MASK_TYPES)
    args, pattern, color = MaskArgs(), "", ""
    r = rng.random()
    if mtype in ("cloth", "surgical") and r < 0.5 and textures:
        pattern = rng.choice(textures)
        args.pattern, args.pattern_weight = pattern, rng.uniform(0.5, 1.0)
    elif r < 0.85:
        color = rng.choice(MASK_COLORS)
        args.color, args.color_weight = color, rng.uniform(0.3, 0.9)
    out, alpha = AF.mask_face(img.copy(), AF.rect_to_bb(rect), six, angle, args, type=mtype)
    gt = alpha > 127
    if gt.sum() < 50:
        raise ValueError("empty face mask")
    meta = {"mask_type": mtype, "mask_pattern": os.path.basename(pattern), "mask_color": color}
    return out, gt, meta


# ---------------------------------------------------------------------------
# glasses (procedural)
# ---------------------------------------------------------------------------

FRAME_COLORS = [(20, 20, 20), (10, 10, 10), (40, 40, 40), (35, 60, 95), (30, 45, 70),
                (40, 160, 200), (180, 180, 180), (200, 200, 205), (30, 30, 150),
                (150, 60, 20), (60, 120, 30), (230, 230, 230), (90, 30, 120)]   # BGR
DARK_LENS = [(15, 15, 15), (30, 30, 30), (25, 35, 45), (40, 30, 20), (20, 40, 25), (50, 50, 50)]
STYLES = ["round", "rect", "wayfarer", "aviator", "cateye"]


def _lens_outline(style, rng, n=120):
    """Closed lens outline in local coords (u,v in about [-1,1]); +u is the OUTER side."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    p = {"round": 2.0, "rect": rng.uniform(4, 7), "wayfarer": rng.uniform(3, 5),
         "aviator": 2.2, "cateye": rng.uniform(2.5, 3.5)}[style]
    c, s = np.cos(t), np.sin(t)
    u = np.sign(c) * np.abs(c) ** (2 / p)
    v = np.sign(s) * np.abs(s) ** (2 / p)          # +v is down
    if style == "wayfarer":
        u = u * (1 + 0.18 * (-v))                    # wider at the top
    elif style == "aviator":
        inner = np.clip(-u, 0, 1)
        v = np.where(v > 0, v * (1 + 0.35 * inner), v * 0.85)   # drops low near the nose
    elif style == "cateye":
        v = np.where(v < 0, v - 0.35 * np.clip(u, 0, 1) ** 2, v)  # raised outer corner
    return np.stack([u, v], 1)


def _poly_mask(shape, pts):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1, lineType=cv2.LINE_8)
    return m


def _line_mask(shape, pts, thick, closed):
    m = np.zeros(shape, np.uint8)
    cv2.polylines(m, [np.round(pts).astype(np.int32)], closed, 1, max(1, int(round(thick))), cv2.LINE_8)
    return m


def apply_glasses(img, pts, rng, S=4):
    h, w = img.shape[:2]
    c_r, c_l = pts[36:42].mean(0), pts[42:48].mean(0)          # image-left eye, image-right eye
    d = np.linalg.norm(c_l - c_r)
    ang = math.atan2(c_l[1] - c_r[1], c_l[0] - c_r[0]) + math.radians(rng.uniform(-2, 2))
    mid = (c_r + c_l) / 2
    ca, sa = math.cos(ang), math.sin(ang)

    style = rng.choice(STYLES)
    lens_w = d * rng.uniform(0.40, 0.50)                         # half-width of one lens
    ratio = {"round": rng.uniform(0.85, 1.0), "rect": rng.uniform(0.50, 0.65),
             "wayfarer": rng.uniform(0.62, 0.75), "aviator": rng.uniform(0.75, 0.88),
             "cateye": rng.uniform(0.55, 0.68)}[style]
    lens_h = lens_w * ratio
    lens_cx = d * rng.uniform(0.49, 0.55)
    lens_cy = d * rng.uniform(0.0, 0.10)
    frame_t = lens_w * rng.uniform(0.06, 0.16)
    outline = _lens_outline(style, rng)

    def to_img(xy):                                             # glasses frame -> image px (scaled)
        x, y = xy[:, 0], xy[:, 1]
        return np.stack([mid[0] + x * ca - y * sa, mid[1] + x * sa + y * ca], 1) * S

    lenses = []
    for side in (-1, 1):                                        # -1: image-left lens
        u, v = outline[:, 0] * side, outline[:, 1]              # mirror so +u is outward
        lenses.append(to_img(np.stack([side * lens_cx + u * lens_w, lens_cy + v * lens_h], 1)))

    lens_kind = rng.choices(["dark", "mirror", "tint", "clear"], [0.45, 0.2, 0.2, 0.15])[0]
    frame_col = np.array(rng.choice(FRAME_COLORS), np.float32)
    if lens_kind == "clear":
        frame_t *= rng.uniform(0.5, 0.8)

    HS, WS = h * S, w * S
    col = np.zeros((HS, WS, 3), np.float32)
    alpha = np.zeros((HS, WS), np.float32)
    foot = np.zeros((HS, WS), np.uint8)

    def paint(m, color, a):
        nonlocal col, alpha
        m = m.astype(np.float32) * a
        col = col * (1 - m[..., None]) + np.asarray(color, np.float32) * m[..., None]
        alpha = alpha + m * (1 - alpha)

    # lenses
    for L in lenses:
        m = _poly_mask((HS, WS), L)
        foot |= m
        if lens_kind == "dark":
            paint(m, rng.choice(DARK_LENS), rng.uniform(0.93, 1.0))
        elif lens_kind == "tint":
            paint(m, rng.choice(DARK_LENS + [(60, 40, 120), (40, 90, 140), (120, 60, 40)]), rng.uniform(0.5, 0.8))
        elif lens_kind == "clear":
            paint(m, (235, 235, 235), rng.uniform(0.03, 0.12))
        else:  # mirror: vertical gradient between two bright colours
            y0, y1 = L[:, 1].min(), L[:, 1].max()
            top = np.array([rng.randint(0, 255) for _ in range(3)], np.float32)
            bot = np.array([rng.randint(0, 255) for _ in range(3)], np.float32)
            ys = np.clip((np.arange(HS, dtype=np.float32) - y0) / max(1.0, y1 - y0), 0, 1)
            grad = top[None, :] * (1 - ys[:, None]) + bot[None, :] * ys[:, None]
            mf = m.astype(np.float32) * 0.97
            col = col * (1 - mf[..., None]) + grad[:, None, :] * mf[..., None]
            alpha = alpha + mf * (1 - alpha)
        if lens_kind in ("dark", "mirror") and rng.random() < 0.5:   # highlight streak
            x0, x1 = L[:, 0].min(), L[:, 0].max()
            y0, y1 = L[:, 1].min(), L[:, 1].max()
            band = np.zeros((HS, WS), np.uint8)
            p1 = (int(x0 + 0.2 * (x1 - x0)), int(y1))
            p2 = (int(x0 + 0.6 * (x1 - x0)), int(y0))
            cv2.line(band, p1, p2, 1, max(2, int(0.08 * (x1 - x0))))
            paint(band & m, (255, 255, 255), rng.uniform(0.15, 0.35))

    # frames
    if not (lens_kind != "clear" and rng.random() < 0.1):           # 10% rimless sunglasses
        for L in lenses:
            m = _line_mask((HS, WS), L, frame_t * S, True)
            foot |= m
            paint(m, frame_col, 1.0)

    # bridge (inner-top points of both lenses, arched upward)
    a_, b_ = lenses[0], lenses[1]
    pa = a_[np.argmax(a_[:, 0] * ca + a_[:, 1] * sa - 0.6 * (-a_[:, 0] * sa + a_[:, 1] * ca))]
    pb = b_[np.argmin(b_[:, 0] * ca + b_[:, 1] * sa + 0.6 * (-b_[:, 0] * sa + b_[:, 1] * ca))]
    lift = d * S * rng.uniform(0.0, 0.06)
    pm = (pa + pb) / 2 + np.array([sa, -ca]) * lift
    bridge = np.stack([pa, pm, pb])
    m = _line_mask((HS, WS), bridge, frame_t * S * rng.uniform(0.9, 1.4), False)
    foot |= m
    paint(m, frame_col, 1.0)
    if style == "aviator":                                          # double bridge
        off = np.array([sa, -ca]) * frame_t * S * 2.2
        m = _line_mask((HS, WS), bridge + off, frame_t * S * 0.8, False)
        foot |= m
        paint(m, frame_col, 1.0)

    # temples (short arms going outward from the outer-top corner)
    for side, L in zip((-1, 1), lenses):
        along = (L[:, 0] - mid[0] * S) * ca + (L[:, 1] - mid[1] * S) * sa
        outer = L[np.argmax(along * side - 0.4 * ((L[:, 1] - mid[1] * S) * ca - (L[:, 0] - mid[0] * S) * sa))]
        length = d * S * rng.uniform(0.15, 0.35)
        end = outer + np.array([ca, sa]) * side * length + np.array([-sa, ca]) * length * 0.1
        m = _line_mask((HS, WS), np.stack([outer, end]), frame_t * S * 0.8, False)
        foot |= m
        paint(m, frame_col, 1.0)

    # downsample (premultiplied) and composite
    pre = cv2.resize(col * alpha[..., None], (w, h), interpolation=cv2.INTER_AREA)
    a = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_AREA)
    out = img.astype(np.float32) * (1 - a[..., None]) + pre
    gt = cv2.resize(foot.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) > 0.5
    meta = {"glasses_style": style, "lens_type": lens_kind}
    return np.clip(out, 0, 255).astype(np.uint8), gt, meta


# ---------------------------------------------------------------------------
# per-image job
# ---------------------------------------------------------------------------

def process(job):
    src, fid, split, seed, textures = job
    rng = random.Random(seed)
    rec = {"id": fid, "split": split, "face_found": 0, "has_glasses": 0, "has_mask": 0,
           "glasses_style": "", "lens_type": "", "mask_type": "", "mask_pattern": "",
           "mask_color": "", "glasses_px": 0, "mask_px": 0, "error": ""}
    try:
        img = cv2.imread(src)
        img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        rects = DET(img, 0) or DET(img, 1)
        if not rects:
            return rec
        rect = max(rects, key=lambda r: r.width() * r.height())
        pts = FACE_UTILS.shape_to_np(PRED(img, rect)).astype(np.float32)
        rec["face_found"] = 1

        kind = rng.choices(["glasses", "mask", "both"], [0.4, 0.4, 0.2])[0]
        g_gt = np.zeros((SIZE, SIZE), bool)
        m_gt = np.zeros((SIZE, SIZE), bool)
        if kind in ("mask", "both"):
            try:
                img, m_gt, meta = apply_face_mask(img, rect, pts.astype(np.int32), rng, textures)
                rec.update(meta)
                rec["has_mask"] = 1
            except Exception as e:  # odd pose: fall back to glasses only
                rec["error"] = f"mask:{type(e).__name__}"
                kind = "glasses"
        if kind in ("glasses", "both"):
            img, g_gt, meta = apply_glasses(img, pts, rng)
            rec.update(meta)
            rec["has_glasses"] = 1
            m_gt &= ~g_gt                         # glasses sit on top of the mask

        cv2.imwrite(f"{OUT_DIR}/images/{fid}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 93])
        cv2.imwrite(f"{OUT_DIR}/masks/{fid}_glasses.png", g_gt.astype(np.uint8) * 255)
        cv2.imwrite(f"{OUT_DIR}/masks/{fid}_mask.png", m_gt.astype(np.uint8) * 255)
        rec["glasses_px"], rec["mask_px"] = int(g_gt.sum()), int(m_gt.sum())
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"[:200]
    return rec


# ---------------------------------------------------------------------------
# preview
# ---------------------------------------------------------------------------

def make_preview(records, n=24, cols=6, cell=200):
    ok = [r for r in records if r["has_glasses"] or r["has_mask"]]
    rng = random.Random(0)
    picks = rng.sample(ok, min(n, len(ok)))
    rows = math.ceil(len(picks) / cols)
    canvas = np.full((rows * cell * 2, cols * cell, 3), 255, np.uint8)
    for i, r in enumerate(picks):
        img = cv2.imread(f"{OUT_DIR}/images/{r['id']}.jpg")
        g = cv2.imread(f"{OUT_DIR}/masks/{r['id']}_glasses.png", 0) > 0
        m = cv2.imread(f"{OUT_DIR}/masks/{r['id']}_mask.png", 0) > 0
        ov = img.copy()
        ov[g] = (0.4 * ov[g] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
        ov[m] = (0.4 * ov[m] + 0.6 * np.array([0, 255, 0])).astype(np.uint8)
        y, x = (i // cols) * cell * 2, (i % cols) * cell
        canvas[y:y + cell, x:x + cell] = cv2.resize(img, (cell, cell))
        canvas[y + cell:y + 2 * cell, x:x + cell] = cv2.resize(ov, (cell, cell))
        label = f"{r['id']} {r['glasses_style'] or ''} {r['mask_type'] or ''}".strip()
        cv2.putText(canvas, label, (x + 4, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2)
        cv2.putText(canvas, label, (x + 4, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
    cv2.imwrite(f"{KAGGLE_OUT}/preview.png", canvas)


# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    setup()
    faces = find_faces_dir()
    files = sorted(f for f in os.listdir(faces) if f.lower().endswith((".jpg", ".png")))
    print(f"{len(files)} faces in {faces}", flush=True)

    rng = random.Random(SEED)
    order = files[:]
    rng.shuffle(order)
    split = {f: "test" for f in order[:N_TEST]}
    split.update({f: "val" for f in order[N_TEST:N_TEST + N_VAL]})
    if LIMIT:
        order = order[:LIMIT]

    textures = sorted(
        os.path.join("masks/textures", d, f)
        for d in os.listdir(f"{MTF_DIR}/masks/textures")
        for f in os.listdir(f"{MTF_DIR}/masks/textures/{d}")
    )
    held_out = set(random.Random(SEED + 1).sample(textures, 5))
    train_tex = [t for t in textures if t not in held_out]
    test_tex = sorted(held_out)

    os.makedirs(f"{OUT_DIR}/images", exist_ok=True)
    os.makedirs(f"{OUT_DIR}/masks", exist_ok=True)
    jobs = []
    for i, f in enumerate(order):
        s = split.get(f, "train")
        jobs.append((os.path.join(faces, f), os.path.splitext(f)[0], s, SEED * 100003 + i,
                     test_tex if s == "test" else train_tex))

    records = []
    with Pool(WORKERS, initializer=init_worker) as pool:
        for k, rec in enumerate(pool.imap_unordered(process, jobs, chunksize=16), 1):
            records.append(rec)
            if k % 1000 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)

    records.sort(key=lambda r: r["id"])
    keys = list(records[0].keys())
    with open(f"{OUT_DIR}/meta.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, keys)
        wr.writeheader()
        wr.writerows(records)

    made = [r for r in records if r["has_glasses"] or r["has_mask"]]
    summary = {
        "processed": len(records),
        "written": len(made),
        "no_face": sum(1 - r["face_found"] for r in records),
        "errors": sum(bool(r["error"]) and not (r["has_glasses"] or r["has_mask"]) for r in records),
        "mask_fallbacks": sum(r["error"].startswith("mask:") for r in records),
        "by_split": {s: sum(r["split"] == s for r in made) for s in ("train", "val", "test")},
        "glasses": sum(r["has_glasses"] for r in made),
        "masks": sum(r["has_mask"] for r in made),
        "both": sum(r["has_glasses"] and r["has_mask"] for r in made),
        "held_out_textures": [os.path.basename(t) for t in test_tex],
        "median_glasses_pct": float(np.median([r["glasses_px"] for r in made if r["has_glasses"]] or [0]) / SIZE ** 2 * 100),
        "median_mask_pct": float(np.median([r["mask_px"] for r in made if r["has_mask"]] or [0]) / SIZE ** 2 * 100),
        "error_examples": [r["error"] for r in records if r["error"]][:10],
        "seconds": round(time.time() - t0),
    }
    with open(f"{OUT_DIR}/summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2), flush=True)

    make_preview(records)
    shutil.copy(f"{OUT_DIR}/meta.csv", KAGGLE_OUT)
    shutil.copy(f"{OUT_DIR}/summary.json", KAGGLE_OUT)
    with tarfile.open(f"{KAGGLE_OUT}/visus_occlusion_v1.tar", "w") as tar:
        tar.add(OUT_DIR, arcname="visus_occlusion_v1")
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
