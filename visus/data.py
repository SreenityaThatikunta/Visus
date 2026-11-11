import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import clip


class OcclusionDataset(Dataset):
    """Pairs occluded face images with binary masks and samples a text prompt per image."""

    _PROMPTS = {
        "glasses": [
            "Glasses", "Sunglass", "Eyewear", "Covered eyes", "Occluded eyes",
        ],
        "mask": [
            "Mask", "Face mask", "Occluded region", "Covered face", "Non face region",
        ],
        "default": [
            "Face", "Human face", "Visible face", "Uncovered face", "Clear face region",
        ],
    }

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        clip_model_name: str = "ViT-B/32",
        image_size: int = 224,
        max_samples: Optional[int] = None,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size

        _, self.clip_preprocess = clip.load(clip_model_name, device="cpu")

        self.image_files = self._collect_valid_pairs()
        if max_samples:
            self.image_files = self.image_files[:max_samples]

        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > 0.5).float()),
        ])

        print(f"Found {len(self.image_files)} valid image-mask pairs")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_valid_pairs(self) -> list[Path]:
        mask_stems = {m.stem for m in self.masks_dir.glob("*")}
        valid: list[Path] = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            for img in self.images_dir.glob(ext):
                if f"{img.stem}_mask" in mask_stems or f"{img.stem}_glasses" in mask_stems:
                    valid.append(img)
        return sorted(valid)

    def _find_mask(self, img_path: Path) -> Optional[Path]:
        for suffix in ("_mask", "_glasses"):
            for ext in (img_path.suffix, ".png", ".jpg", ".jpeg", ".bmp"):
                candidate = self.masks_dir / f"{img_path.stem}{suffix}{ext}"
                if candidate.exists():
                    return candidate
        return None

    def _text_label(self, filename: str) -> str:
        name = filename.lower()
        if "glasses" in name:
            key = "glasses"
        elif "mask" in name:
            key = "mask"
        else:
            key = "default"
        return random.choice(self._PROMPTS[key])

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.image_files[idx]

        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new("RGB", (self.image_size, self.image_size))

        # Load mask
        mask_path = self._find_mask(img_path)
        if mask_path is None:
            print(f"Warning: no mask for {img_path.name}")
            mask = Image.new("L", (self.image_size, self.image_size), 0)
        else:
            try:
                mask = Image.open(mask_path).convert("L")
            except Exception as e:
                print(f"Error loading mask {mask_path}: {e}")
                mask = Image.new("L", (self.image_size, self.image_size), 0)

        return {
            "image": self.clip_preprocess(image),
            "text": self._text_label(img_path.name),
            "mask": self.mask_transform(mask),
            "filename": img_path.name,
        }


def _collate(batch: list[dict]) -> dict:
    return {
        "images": torch.stack([b["image"] for b in batch]),
        "texts": clip.tokenize([b["text"] for b in batch]),
        "masks": torch.stack([b["mask"] for b in batch]),
        "filenames": [b["filename"] for b in batch],
    }


def create_dataloaders(
    images_dir: str,
    masks_dir: str,
    batch_size: int = 8,
    train_split: float = 0.8,
    clip_model_name: str = "ViT-B/32",
    num_workers: int = 4,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    dataset = OcclusionDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        clip_model_name=clip_model_name,
        max_samples=max_samples,
    )

    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
    )

    print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    pin = torch.cuda.is_available()
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=pin,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

    return train_loader, val_loader
