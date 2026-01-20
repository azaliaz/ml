"""
src/dataset.py

YOLO-ready PyTorch Dataset + collate + DataLoader factory for the SKU110 -> dataset_yolo layout.

Assumptions (compatible with your scripts/prepare_data.py):
 - dataset_root/
     - train/images/*.jpg
     - train/labels/*.txt   (YOLO format: class x_center y_center w h, normalized)
     - val/
     - test/

Exports:
 - YOLODataset
 - collate_fn
 - get_dataloader

Features:
 - Applies albumentations transforms (expects bboxes in 'yolo' format)
 - Returns image Tensor (C,H,W) and targets dict with 'boxes' (x1,y1,x2,y2 absolute px) and 'labels'
 - Optionally return normalized 'yolo_boxes' in targets if return_yolo_boxes=True
 - Robust to empty label files
"""

from pathlib import Path
from typing import Callable, List, Optional, Tuple, Dict, Any

import os
import math
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

# Try to reuse helper from src.data if present (parse_yolo_label_file)
try:
    from src.data import parse_yolo_label_file
except Exception:
    def parse_yolo_label_file(label_path: str) -> Tuple[List[List[float]], List[int]]:
        """Fallback parser for YOLO txt label files.
        Each non-empty line: <class_id> <x_center> <y_center> <w> <h>
        Returns (bboxes, class_ids) where bboxes are [x_c, y_c, w, h] (floats, normalized) and class_ids are ints.
        """
        bboxes = []
        cls = []
        if not os.path.exists(label_path):
            return bboxes, cls
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    class_id = int(float(parts[0]))
                    x_c = float(parts[1])
                    y_c = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                except Exception:
                    continue
                bboxes.append([x_c, y_c, w, h])
                cls.append(class_id)
        return bboxes, cls


def yolo_to_xyxy(yolo_box: List[float], img_w: int, img_h: int) -> List[float]:
    """Convert single YOLO bbox [x_c, y_c, w, h] normalized -> [x1, y1, x2, y2] absolute pixel coords.
    Clamps to image boundaries.
    """
    x_c, y_c, w, h = yolo_box
    x_center = x_c * img_w
    y_center = y_c * img_h
    bw = w * img_w
    bh = h * img_h
    x1 = x_center - bw / 2.0
    y1 = y_center - bh / 2.0
    x2 = x_center + bw / 2.0
    y2 = y_center + bh / 2.0
    # clamp
    x1 = max(0.0, min(x1, img_w - 1.0))
    y1 = max(0.0, min(y1, img_h - 1.0))
    x2 = max(0.0, min(x2, img_w - 1.0))
    y2 = max(0.0, min(y2, img_h - 1.0))
    return [x1, y1, x2, y2]


def xyxy_to_yolo_abs(box: List[float], img_w: int, img_h: int) -> List[float]:
    """Convert absolute xyxy -> normalized YOLO (x_c, y_c, w, h).
    Input box: [x1,y1,x2,y2] absolute pixels.
    """
    x1, y1, x2, y2 = box
    x1 = float(x1)
    y1 = float(y1)
    x2 = float(x2)
    y2 = float(y2)
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    if img_w == 0 or img_h == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x_c = (x1 + x2) / 2.0 / img_w
    y_c = (y1 + y2) / 2.0 / img_h
    w = bw / img_w
    h = bh / img_h
    # clamp
    x_c = min(max(x_c, 0.0), 1.0)
    y_c = min(max(y_c, 0.0), 1.0)
    w = min(max(w, 0.0), 1.0)
    h = min(max(h, 0.0), 1.0)
    return [x_c, y_c, w, h]


class YOLODataset(Dataset):
    """PyTorch Dataset for YOLO-style dataset layout.

    Args:
        dataset_root: Path to dataset root (contains train/val/test directories)
        split: 'train'|'val'|'test'
        transforms: albumentations transform (expects 'yolo' bboxes and label_fields=['category_ids'])
        img_exts: tuple of allowed image extensions
        return_yolo_boxes: if True, include `yolo_boxes` (normalized) in targets dict
        keep_empty: if True, include images without boxes (empty targets)
    """

    def __init__(
        self,
        dataset_root: str,
        split: str = "train",
        transforms: Optional[Callable] = None,
        img_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"),
        return_yolo_boxes: bool = False,
        keep_empty: bool = True,
    ) -> None:
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.transforms = transforms
        self.img_exts = img_exts
        self.return_yolo_boxes = return_yolo_boxes
        self.keep_empty = keep_empty

        self.images_dir = self.dataset_root / split / "images"
        self.labels_dir = self.dataset_root / split / "labels"

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images dir not found: {self.images_dir}")
        # build list of images
        self.image_paths = sorted(
            [p for p in self.images_dir.iterdir() if p.suffix.lower() in self.img_exts]
        )

        # optionally filter images that are missing labels? We'll keep all but warn
        filtered = []
        for p in self.image_paths:
            label_path = self.labels_dir / (p.stem + ".txt")
            if not label_path.exists() and not self.keep_empty:
                # пропускаем изображение без меток
                continue
            filtered.append(p)
        self.image_paths = filtered
        if not self.labels_dir.exists():
            import warnings
            warnings.warn(f"Labels dir not found: {self.labels_dir} — метки будут считаться пустыми.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _read_image(self, path: Path) -> Tuple[np.ndarray, int, int]:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            arr = np.array(im)
        return arr, h, w

    def _read_labels(self, label_path: Path, img_w: int, img_h: int) -> Tuple[List[List[float]], List[int]]:
        bboxes, labels = parse_yolo_label_file(str(label_path))
        # parse_yolo_label_file returns normalized yolo boxes by contract
        return bboxes, labels

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        img_path = self.image_paths[idx]
        label_path = self.labels_dir / (img_path.stem + ".txt")

        image, img_h, img_w = self._read_image(img_path)

        bboxes_yolo, labels = self._read_labels(label_path, img_w, img_h)

        # albumentations expects bboxes as list of lists and category_ids
        if self.transforms is not None:
            # If there are no bboxes, pass empty lists — albumentations handles it
            transformed = self.transforms(image=image, bboxes=bboxes_yolo, category_ids=labels)
            image_t = transformed["image"]  # expected ToTensorV2 -> torch.Tensor
            bboxes_yolo_t = transformed.get("bboxes", [])
            labels_t = transformed.get("category_ids", [])

            # Convert YOLO bboxes normalized -> xyxy absolute (after transform image size)
            # Need the transformed image size
            # image_t is Tensor CxHxW
            _, H, W = image_t.shape
            boxes_xyxy = [yolo_to_xyxy(b, W, H) for b in bboxes_yolo_t]
        else:
            # convert raw numpy image -> tensor
            # images likely different sizes; convert to float tensor in 0..1
            image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            boxes_xyxy = [yolo_to_xyxy(b, img_w, img_h) for b in bboxes_yolo]
            labels_t = labels

        # convert lists -> tensors
        if len(boxes_xyxy) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.tensor(boxes_xyxy, dtype=torch.float32)
            labels_tensor = torch.tensor(labels_t, dtype=torch.int64)

        targets: Dict[str, Any] = {
            "boxes": boxes_tensor,  # [N,4] x1,y1,x2,y2 absolute pixels (relative to transformed image)
            "labels": labels_tensor,  # [N]
            "image_id": torch.tensor([idx]),
        }

        if self.return_yolo_boxes:
            # compute normalized YOLO boxes for returned (use the image_t size)
            _, H, W = image_t.shape
            yolo_norm = []
            for b in boxes_tensor.tolist():
                yolo_norm.append(xyxy_to_yolo_abs(b, W, H))
            if len(yolo_norm) == 0:
                yolo_tensor = torch.zeros((0, 4), dtype=torch.float32)
            else:
                yolo_tensor = torch.tensor(yolo_norm, dtype=torch.float32)
            targets["yolo_boxes"] = yolo_tensor

        return image_t, targets


def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, Any]]]) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """Collate function for variable-size targets. Returns batched images tensor and list of target dicts.

    Images will be stacked (they must have same H,W inside a batch — ensured if transforms pad to img_size).
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    # stack images
    try:
        images_tensor = torch.stack(images, dim=0)
    except Exception as e:
        # helpful error when sizes mismatch
        raise RuntimeError("Could not stack images into batch. Are your transforms producing consistent sizes?") from e
    return images_tensor, targets


def get_dataloader(
    dataset_root: str,
    split: str,
    transforms: Optional[Callable],
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    **dataset_kwargs,
) -> DataLoader:
    """Factory: creates YOLODataset + DataLoader with collate_fn.

    dataset_kwargs are forwarded to YOLODataset.
    """
    ds = YOLODataset(dataset_root, split=split, transforms=transforms, **dataset_kwargs)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    return loader


# If run as script -> quick sanity check (won't run heavy transforms)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--img_size", type=int, default=640)
    args = parser.parse_args()

    # minimal transform: resize+pad+to tensor to test
    try:
        from src.data import get_val_transforms
        transforms = get_val_transforms(args.img_size)
    except Exception:
        transforms = None

    dl = get_dataloader(args.dataset_root, args.split, transforms=transforms, batch_size=4, num_workers=0)
    for images, targets in dl:
        print("Batch images:", images.shape)
        print("Targets example:", {k: v.shape if torch.is_tensor(v) else type(v) for k, v in targets[0].items()})
        break
