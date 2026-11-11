"""
Generate ground-truth occlusion masks with MediaPipe Face Mesh.

Files whose name contains `_glasses` get an eye-region mask; files containing
`_mask` get a lower-face mask. Output: `<stem>_glasses.png` / `<stem>_mask.png`.

    python scripts/data/generate_gt_masks.py --input data/celeba/occluded --output data/celeba/masks

Requires `pip install mediapipe`.
"""
import argparse
import os
import numpy as np
from PIL import Image
import mediapipe as mp
import cv2


def generate_eye_mask(img_path, face_mesh):
    img_bgr = cv2.imread(img_path)
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(img_rgb)
    mask = np.full((h, w), 255, dtype=np.uint8)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            indices = [54, 284, 323, 93]
            points = np.array([
                (int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h))
                for i in indices
            ])
            cv2.fillPoly(mask, [points], 0)
    return mask

def generate_mask_mask(img_path, face_mesh, extend_down_px=10):
    img_bgr = cv2.imread(img_path)
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(img_rgb)
    mask = np.full((h, w), 255, dtype=np.uint8)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            indices = [197, 137, 58, 136, 149, 152, 378, 365, 435, 366]
            points = []
            for i in indices:
                x = int(face_landmarks.landmark[i].x * w)
                y = int(face_landmarks.landmark[i].y * h)
                points.append((x, y))

            max_y = max(p[1] for p in points)
            shifted_points = []
            for (x, y) in points:
                if abs(y - max_y) < 10:
                    y = min(h - 1, y + extend_down_px)
                shifted_points.append((x, y))

            points_np = np.array(shifted_points)
            cv2.fillPoly(mask, [points_np], 0)
    return mask

def mask_create(input_path, face_mesh):
    filename = os.path.basename(input_path)
    mask = None

    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        print(f"Skipping non-image file: {filename}")
    elif "_glasses" in filename.lower():
        print(f"Generating eye mask for: {filename}")
        mask = generate_eye_mask(input_path, face_mesh)
    elif "_mask" in filename.lower():
        print(f"Generating mask mask for: {filename}")
        mask = generate_mask_mask(input_path, face_mesh)
    else:
        print(f"Skipping: {filename}")

    if mask is None:
        return os.path.splitext(filename)[0], None

    mask_img = Image.fromarray(mask)
    return os.path.splitext(filename)[0], mask_img


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    input_dir, output_dir = a.input, a.output

    os.makedirs(output_dir, exist_ok=True)

    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

    for fname in os.listdir(input_dir):
        input_path = os.path.join(input_dir, fname)

        name, mask_img = mask_create(input_path, mp_face_mesh)

        if mask_img is None:
            continue

        if "_glasses" in fname.lower():
            save_path = os.path.join(output_dir, name + "_glasses.png")
        elif "_mask" in fname.lower():
            save_path = os.path.join(output_dir, name + "_mask.png")
        else:
            continue

        mask_img.save(save_path)

    mp_face_mesh.close()