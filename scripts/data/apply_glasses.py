"""
Overlay a transparent glasses PNG onto every face in a folder (dlib 68-point landmarks).

    python scripts/data/apply_glasses.py --input data/celeba/raw --output data/celeba/glasses \
        --overlay glasses.png --predictor shape_predictor_68_face_landmarks.dat

Requires `pip install dlib cvzone` and dlib's 68-point model
(http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).
"""
import argparse
import os

import cv2
import cvzone
import dlib

def apply_sunglasses(face_image, overlay_path, detector, predictor):
    
    overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
    if overlay is None:
        raise FileNotFoundError(f"Error: '{overlay_path}' not found.")
    
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = detector(gray)
    if len(faces) == 0:
        print("No faces detected!")
        return face_image
    
    for face in faces:
        landmarks = predictor(gray, face)
        
        # Get eye coordinates
        left_eye_x = (landmarks.part(36).x + landmarks.part(39).x) // 2
        left_eye_y = (landmarks.part(36).y + landmarks.part(39).y) // 2
        right_eye_x = (landmarks.part(42).x + landmarks.part(45).x) // 2
        right_eye_y = (landmarks.part(42).y + landmarks.part(45).y) // 2
        
        # Calculate width and height
        width = int(2 * (right_eye_x - left_eye_x))
        height = int(width * 1.2)
        
        # Resize sunglasses
        overlay_resized = cv2.resize(overlay, (width, height))
        
        # Determine placement
        x = left_eye_x - int(width * 0.3)
        y = left_eye_y - int(height * 0.5)
        
        # Overlay
        face_image = cvzone.overlayPNG(face_image, overlay_resized, [x, y])
    
    return face_image


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--overlay", required=True, help="RGBA glasses image")
    p.add_argument("--predictor", default="shape_predictor_68_face_landmarks.dat")
    a = p.parse_args()

    os.makedirs(a.output, exist_ok=True)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(a.predictor)

    for filename in sorted(os.listdir(a.input)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        face_img = cv2.imread(os.path.join(a.input, filename))
        if face_img is None:
            print(f"Could not load {filename}, skipping")
            continue
        stem, ext = os.path.splitext(filename)
        out = apply_sunglasses(face_img, a.overlay, detector, predictor)
        cv2.imwrite(os.path.join(a.output, f"{stem}_glasses{ext}"), out)

    print("Done.")
