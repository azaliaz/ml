# src/augmentations.py
"""
Аугментации для детекции (albumentations), готовые для использования с YOLO-style
аннотациями (x_center, y_center, width, height) — все значения нормализованы (0..1).

Экспортируем:
 - get_train_transforms(img_size)
 - get_val_transforms(img_size)
 - get_test_transforms(img_size)
 - parse_yolo_label_file(label_path)
"""

from typing import List, Tuple
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


def _make_random_resized_crop(img_size: int, **kwargs):
    """
    Создаём RandomResizedCrop в робастном режиме — разные версии albumentations
    принимают либо size=(h,w) либо height=..., width=...
    """
    try:
        # современный API: size=(h,w)
        return A.RandomResizedCrop(size=(img_size, img_size), **kwargs)
    except TypeError:
        # fallback для старых версий
        return A.RandomResizedCrop(height=img_size, width=img_size, **kwargs)


def get_train_transforms(img_size: int = 640):
    """
    Возвращает albumentations.Compose для обучения.
    Ожидает bboxes в формате 'yolo' (x_center, y_center, w, h) — нормализованные.
    """
    try:
        crop = _make_random_resized_crop(img_size, scale=(0.8, 1.0), ratio=(0.75, 1.33), p=0.5)
    except Exception as e:
        raise RuntimeError("Не удалось создать RandomResizedCrop — проверьте версию albumentations.") from e

    # Собираем трансформы
    transforms = [
        crop,
        # геометрические (Affine вместо ShiftScaleRotate чтобы убрать предупреждение)
        A.HorizontalFlip(p=0.5),
        A.Affine(translate_percent=0.06, scale=(0.88, 1.12), rotate=10, border_mode=0, p=0.5),

        # цветовые / шумы (без специфичных kwargs, чтобы избежать несовместимостей)
        A.OneOf(
            [
                A.GaussNoise(p=1.0),  # без var_limit — более совместимо
                A.ISONoise(p=1.0),
            ],
            p=0.2,
        ),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.3),
        A.OneOf([A.CLAHE(p=1.0), A.Equalize(p=1.0), A.RandomGamma(p=1.0)], p=0.3),

        # мелкое размытие / дефекты
        A.OneOf([A.MotionBlur(blur_limit=3, p=1.0), A.MedianBlur(blur_limit=3, p=1.0), A.Blur(blur_limit=3, p=1.0)], p=0.1),

        # паддинг до размера
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, p=1.0),

        # нормализация и конвертация в тензор
        # mean/std используются как для ImageNet (совместимо с большинством предобученных бэков)
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(format="yolo", label_fields=["category_ids"], min_visibility=0.3),
    )


def get_val_transforms(img_size: int = 640):
    """
    Трансформации для валидации: только ресайз/normalize -> tensor.
    """
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["category_ids"], min_visibility=0.0),
    )


def get_test_transforms(img_size: int = 640):
    """Те же, что и для валидации."""
    return get_val_transforms(img_size)




def parse_yolo_label_file(label_path: str) -> Tuple[List[List[float]], List[int]]:
    """
    Robust parser for YOLO .txt:
    - lines: <class_id> <x_center> <y_center> <w> <h>
    - returns list of [x_c, y_c, w, h] (floats normalized) and list of class ids
    - if file missing -> returns ([], [])
    - ensures boxes are clamped so that x1>=0, x2<=1, y1>=0, y2<=1
    """
    p = Path(label_path)
    bboxes = []
    cls = []
    if not p.exists():
        return bboxes, cls

    with p.open("r", encoding="utf-8") as f:
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

            # сначала приведём к числам и защитим от NaN
            x_c = float(x_c); y_c = float(y_c); w = float(w); h = float(h)

            # вычислим абсолютные нормализованные границы, затем зажмём в [0,1]
            x1 = max(0.0, x_c - w / 2.0)
            y1 = max(0.0, y_c - h / 2.0)
            x2 = min(1.0, x_c + w / 2.0)
            y2 = min(1.0, y_c + h / 2.0)

            # если после клипирования бокса площадь нулевая — пропускаем
            if x2 <= x1 or y2 <= y1:
                continue

            # пересчитаем обратно в (x_c, y_c, w, h) гарантируя, что x2<=1 и x1>=0
            x_c_new = (x1 + x2) / 2.0
            y_c_new = (y1 + y2) / 2.0
            w_new = x2 - x1
            h_new = y2 - y1

            # небольшая дополнительная защита: ещё раз клип
            x_c_new = float(max(0.0, min(1.0, x_c_new)))
            y_c_new = float(max(0.0, min(1.0, y_c_new)))
            w_new = float(max(0.0, min(1.0, w_new)))
            h_new = float(max(0.0, min(1.0, h_new)))

            bboxes.append([x_c_new, y_c_new, w_new, h_new])
            cls.append(class_id)

    return bboxes, cls
