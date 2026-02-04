#!/usr/bin/env python3
"""
Preview Albumentations augmentations for YOLO dataset.

- Загружает случайные изображения из датасета
- Применяет трансформации из src/augmentations.py
- Отображает результат с бокcами рядом с оригиналом
"""

import os
import random
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from src import augmentations as augs
import yaml
import albumentations as A


NUM_SAMPLES = 15
IMG_SIZE = 320
DATA_YAML_PATH = Path("configs/data.yaml")


def load_yolo_dataset(data_yaml_path):
    """Возвращает список dict: {'img_path': Path, 'bboxes': [[x_c, y_c, w, h]], 'labels':[int]}"""
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Data YAML not found: {data_yaml_path}")
    with open(data_yaml_path, "r") as f:
        dd = yaml.safe_load(f)
    dataset_root = Path(dd.get("path", "."))

    images = []
    img_files = list(dataset_root.rglob("*.jpg")) + list(dataset_root.rglob("*.png"))

    for img_path in img_files:
        txt_path = img_path.with_suffix(".txt")
        bboxes = []
        labels = []
        if txt_path.exists():
            with open(txt_path, "r") as ftxt:
                for line in ftxt:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        x_c, y_c, w, h = map(float, parts[1:5])
                        bboxes.append([x_c, y_c, w, h])
                        labels.append(cls_id)
        images.append({
            "img_path": img_path,
            "bboxes": bboxes,
            "labels": labels,
        })
    return images

def draw_boxes(image, bboxes, labels=None):
    """Отображение боксов на numpy image (RGB, 0..255)"""
    img = image.copy()
    H, W, _ = img.shape
    for i, box in enumerate(bboxes):
        x_c, y_c, w, h = box
        x1 = int((x_c - w / 2) * W)
        y1 = int((y_c - h / 2) * H)
        x2 = int((x_c + w / 2) * W)
        y2 = int((y_c + h / 2) * H)
        cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
        if labels:
            cv2.putText(img, str(labels[i]), (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return img

def main():
    dataset = load_yolo_dataset(DATA_YAML_PATH)
    if not dataset:
        print("Dataset is empty or no valid images found.")
        return

    samples = random.sample(dataset, min(NUM_SAMPLES, len(dataset)))
    train_augs = augs.get_train_augmentations(IMG_SIZE)

    for idx, sample in enumerate(samples):
        img = cv2.imread(str(sample["img_path"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bboxes = sample["bboxes"]
        labels = sample["labels"]

        transform = A.Compose(
            train_augs,
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])
        )

        augmented = transform(image=img, bboxes=bboxes, class_labels=labels)
        img_aug = augmented["image"]
        bboxes_aug = augmented["bboxes"]
        labels_aug = augmented["class_labels"]

        img_orig_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_orig_drawn = draw_boxes(img_orig_resized, bboxes, labels)
        img_aug_drawn = draw_boxes(img_aug, bboxes_aug, labels_aug)

        # горизонтальное соединение
        combined = np.concatenate([img_orig_drawn, img_aug_drawn], axis=1)

        plt.figure(figsize=(12,6))
        plt.imshow(combined)
        plt.title(f"Sample {idx+1}: {sample['img_path'].name} (Left=original, Right=augmented)")
        plt.axis("off")
    plt.show()

if __name__ == "__main__":
    import numpy as np
    main()
