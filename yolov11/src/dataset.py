"""
src/dataset.py

YOLO-ready PyTorch Dataset + collate + DataLoader factory.

Ожидает layout:
 dataset_root/
   train/images/*.jpg
   train/labels/*.txt
   val/...
   test/...
"""

from pathlib import Path
from typing import Callable, List, Optional, Tuple, Dict, Any

import os
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

# импортируем парсер из augmentations (убедись, что parse_yolo_label_file определён в src/augmentations.py)
from src.augmentations import parse_yolo_label_file


def yolo_to_xyxy(yolo_box: List[float], img_w: int, img_h: int) -> List[float]:
    """Convert single YOLO bbox [x_c, y_c, w, h] normalized -> [x1, y1, x2, y2] absolute px"""
    # защита: нормализованные значения могут немного выйти за [0,1] из-за численной ошибки -> клипируем
    x_c, y_c, w, h = [float(max(0.0, min(1.0, v))) for v in yolo_box]
    x_center = x_c * img_w
    y_center = y_c * img_h
    bw = w * img_w
    bh = h * img_h
    x1 = x_center - bw / 2.0
    y1 = y_center - bh / 2.0
    x2 = x_center + bw / 2.0
    y2 = y_center + bh / 2.0
    # clamp to image bounds
    x1 = max(0.0, min(x1, img_w - 1.0))
    y1 = max(0.0, min(y1, img_h - 1.0))
    x2 = max(0.0, min(x2, img_w - 1.0))
    y2 = max(0.0, min(y2, img_h - 1.0))
    return [x1, y1, x2, y2]


def xyxy_to_yolo_abs(box: List[float], img_w: int, img_h: int) -> List[float]:
    """Convert absolute xyxy -> normalized YOLO (x_c, y_c, w, h)."""
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    if img_w == 0 or img_h == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x_c = (x1 + x2) / 2.0 / img_w
    y_c = (y1 + y2) / 2.0 / img_h
    w = bw / img_w
    h = bh / img_h
    x_c = min(max(x_c, 0.0), 1.0)
    y_c = min(max(y_c, 0.0), 1.0)
    w = min(max(w, 0.0), 1.0)
    h = min(max(h, 0.0), 1.0)
    return [x_c, y_c, w, h]


class YOLODataset(Dataset):
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

        # collect image paths
        self.image_paths = sorted([p for p in self.images_dir.iterdir() if p.suffix.lower() in self.img_exts])

        filtered = []
        for p in self.image_paths:
            label_path = self.labels_dir / (p.stem + ".txt")
            if not label_path.exists() and not self.keep_empty:
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
        # parse_yolo_label_file должен возвращать нормализованные боксы (x_c, y_c, w, h)
        return parse_yolo_label_file(str(label_path))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        img_path = self.image_paths[idx]
        label_path = self.labels_dir / (img_path.stem + ".txt")

        image, img_h, img_w = self._read_image(img_path)
        bboxes_yolo, labels = self._read_labels(label_path, img_w, img_h)

        if self.transforms is not None:
            # albumentations expects lists
            transformed = self.transforms(image=image, bboxes=bboxes_yolo, category_ids=labels)
            image_t = transformed["image"]
            bboxes_yolo_t = transformed.get("bboxes", [])
            labels_t = transformed.get("category_ids", [])
            # image_t may be tensor (C,H,W) or numpy (H,W,C) depending on transforms
            if isinstance(image_t, torch.Tensor):
                if image_t.ndim != 3:
                    raise RuntimeError("Трансформ вернул изображение в неожиданном формате (не C,H,W).")
                _, H, W = image_t.shape
                # boxes -> absolute
                boxes_xyxy = [yolo_to_xyxy(b, W, H) for b in bboxes_yolo_t]
            else:
                # numpy H,W,C
                H, W = image_t.shape[:2]
                boxes_xyxy = [yolo_to_xyxy(b, W, H) for b in bboxes_yolo_t]
                # convert image to tensor
                image_t = torch.from_numpy(image_t).permute(2, 0, 1).float() / 255.0
        else:
            # raw numpy -> convert to tensor (0..1)
            image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            boxes_xyxy = [yolo_to_xyxy(b, img_w, img_h) for b in bboxes_yolo]
            labels_t = labels

        # ensure labels_t exists
        labels_t = locals().get("labels_t", labels)

        # lists -> tensors
        if len(boxes_xyxy) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.tensor(boxes_xyxy, dtype=torch.float32)
            labels_tensor = torch.tensor(labels_t, dtype=torch.int64)

        targets: Dict[str, Any] = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx]),
        }

        if self.return_yolo_boxes:
            _, H, W = image_t.shape
            yolo_norm = [xyxy_to_yolo_abs(b, W, H) for b in boxes_tensor.tolist()]
            yolo_tensor = torch.tensor(yolo_norm, dtype=torch.float32) if len(yolo_norm) else torch.zeros((0, 4), dtype=torch.float32)
            targets["yolo_boxes"] = yolo_tensor

        return image_t, targets


def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, Any]]]) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    try:
        images_tensor = torch.stack(images, dim=0)
    except Exception as e:
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


if __name__ == "__main__":
    # Quick sanity check
    import argparse
    from src.augmentations import get_val_transforms

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--img_size", type=int, default=640)
    args = parser.parse_args()

    transforms = get_val_transforms(args.img_size)
    dl = get_dataloader(args.dataset_root, args.split, transforms=transforms, batch_size=4, num_workers=0)
    for images, targets in dl:
        print("Batch images:", images.shape)
        print("Targets example:", {k: (v.shape if torch.is_tensor(v) else type(v)) for k, v in targets[0].items()})
        break
