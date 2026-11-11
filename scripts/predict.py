"""
Segment glasses / face masks in one image or a folder of images.

    python scripts/predict.py --checkpoint checkpoints/visus_epoch50.pt \
                              --input assets/samples --output outputs/demo

With `--prompt auto` (default) the prompt is read from the filename
(`..._glasses.jpg` -> "glasses", `..._mask.jpg` -> "mask"). Pass `--prompt glasses`
or `--prompt mask` to force one.

Writes `<stem>_<prompt>_mask_binary.png` (the format `scripts/evaluate.py` expects)
and a `<stem>_<prompt>_viz.png` figure for every image.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm  # noqa: E402

from visus.inference import DEFAULT_THRESHOLD, Predictor, list_images, prompts_from_filename  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="path to a .pt checkpoint")
    p.add_argument("--input", required=True, help="image file or directory")
    p.add_argument("--output", default="outputs/predictions")
    p.add_argument("--prompt", default="auto", help='"auto", "glasses", "mask", or any text')
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--no-figures", action="store_true", help="only write binary masks")
    p.add_argument("--device", default=None, help="cuda / mps / cpu (auto-detected if omitted)")
    a = p.parse_args()

    predictor = Predictor(a.checkpoint, device=a.device, threshold=a.threshold)
    images = list_images(a.input)
    print(f"{len(images)} image(s) on {predictor.device}")

    skipped = 0
    for img_path in tqdm(images, unit="img"):
        prompts = prompts_from_filename(img_path) if a.prompt == "auto" else [a.prompt]
        if not prompts:
            skipped += 1
            continue
        for prompt in prompts:
            result = predictor.predict(img_path, prompt)
            stem = f"{img_path.stem.lower()}_{prompt.replace(' ', '-')}"
            Predictor.save(result, a.output, stem, with_figure=not a.no_figures)

    if skipped:
        print(f"Skipped {skipped} image(s) with no 'glasses'/'mask' in the name (use --prompt).")
    print(f"Outputs written to {a.output}")
    predictor.cleanup()


if __name__ == "__main__":
    main()
