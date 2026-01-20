# src/augmentations.py
"""
Аугментации для детекции (albumentations), готовые для использования с YOLO-style
аннотациями (x_center, y_center, width, height) — все значения нормализованы (0..1).

Функции:
 - get_train_transforms(img_size)
 - get_val_transforms(img_size)
 - get_test_transforms(img_size)

Пример использования в Dataset.__getitem__:
    # image: numpy array HxWxC (uint8)
    # bboxes: list of [x_c, y_c, w, h] (normalized floats)
    # labels: list of int (class ids)
    transform = get_train_transforms(640)
    out = transform(image=image, bboxes=bboxes, category_ids=labels)
    img_t = out['image']                # torch.Tensor, 3 x H x W
    bboxes_t = out['bboxes']            # list of [x_c, y_c, w, h] normalized
    labels_t = out['category_ids']      # list of ints
"""

from typing import List, Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_size: int = 640):
    """
    Возвращает albumentations.Compose для обучения.
    Ожидает bboxes в формате 'yolo' (x_center, y_center, w, h) — нормализованные.
    """
    return A.Compose(
        [
            # случайный ресайз/обрезка -> улучшает обучение при разных масштабах
            A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.8, 1.0), ratio=(0.75, 1.33), p=0.5),

            # геометрические
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.06, scale_limit=0.12, rotate_limit=10, border_mode=0, p=0.5
            ),

            # цветовые / шумы
            A.OneOf(
                [
                    A.GaussNoise(var_limit=(10.0, 50.0)),
                    A.ISONoise(),
                ],
                p=0.2,
            ),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.3),
            A.OneOf([A.CLAHE(p=1), A.Equalize(p=1), A.RandomGamma(p=1)], p=0.3),

            # мелкое размытие / дефекты
            A.OneOf([A.MotionBlur(blur_limit=3), A.MedianBlur(blur_limit=3), A.Blur(blur_limit=3)], p=0.1),

            # иногда паддинг чтобы сохранить соотношение
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, p=1.0),

            # нормализация и конвертация в тензор
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),

            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["category_ids"],
            min_visibility=0.3,  # удалить боксы, видимость которых < 30%
        ),
    )


def get_val_transforms(img_size: int = 640):
    """
    Трансформации для валидации: только ресайз/normalize -> tensor.
    """
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, p=1.0),
            A.Normalize(),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["category_ids"], min_visibility=0.0),
    )


def get_test_transforms(img_size: int = 640):
    """
    Трансформации для тестирования — то же, что валидация.
    """
    return get_val_transforms(img_size)


def parse_yolo_label_file(label_path: str) -> Tuple[List[List[float]], List[int]]:
    """
    Простой парсер YOLO .txt файла:
    каждая строка: <class_id> <x_center> <y_center> <w> <h>
    Возвращает (bboxes, class_ids), где bboxes - list of [x_c, y_c, w, h] (floats)
    """
    bboxes = []
    cls = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            x_c = float(parts[1])
            y_c = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            bboxes.append([x_c, y_c, w, h])
            cls.append(class_id)
    return bboxes, cls
