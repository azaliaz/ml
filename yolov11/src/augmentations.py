# src/augmentations.py
"""
Наборы аугментаций на основе albumentations для задач детекции.
Добавлены:
 - больше видов цветовых/шумовых/геометрических трансформаций
 - кастомная трансформация "полоски" (stripes) через A.Lambda
 - аккуратное комбинирование через OneOf, чтобы не применять сразу всё
BBox format: 'yolo' (x_center, y_center, w, h) — нормализованные 0..1.
"""
from typing import List, Callable
import random
import numpy as np
import albumentations as A

# -----------------------
# Утилита: генерация полосок
# -----------------------
def _add_random_stripes(image: np.ndarray, **kwargs) -> np.ndarray:
    """
    Рисует несколько полупрозрачных полос (горизонтальных или вертикальных)
    поверх изображения. Возвращает изменённый image (uint8 RGB).
    **kwargs нужен для совместимости с albumentations 1.3+
    """
    img = image.copy()
    H, W = img.shape[:2]
    n_stripes = random.randint(1, 6)
    orientation = random.choice(["horizontal", "vertical"])
    color_base = random.choice([(0, 0, 0), (255, 255, 255)]) if random.random() < 0.85 else tuple(np.random.randint(0, 256, 3))
    alpha = random.uniform(0.2, 0.6)

    for _ in range(n_stripes):
        if orientation == "horizontal":
            h = random.randint(max(4, H // 50), max(8, H // 6))
            y = random.randint(0, H - 1)
            y1 = max(0, y - h // 2)
            y2 = min(H, y1 + h)
            overlay = np.full((y2 - y1, W, 3), color_base, dtype=np.uint8)
            img[y1:y2, :, :] = (img[y1:y2, :, :].astype(np.float32) * (1 - alpha) +
                                overlay.astype(np.float32) * alpha).astype(np.uint8)
        else:
            w = random.randint(max(4, W // 50), max(8, W // 6))
            x = random.randint(0, W - 1)
            x1 = max(0, x - w // 2)
            x2 = min(W, x1 + w)
            overlay = np.full((H, x2 - x1, 3), color_base, dtype=np.uint8)
            img[:, x1:x2, :] = (img[:, x1:x2, :].astype(np.float32) * (1 - alpha) +
                                overlay.astype(np.float32) * alpha).astype(np.uint8)
    return img

def Stripes(p: float = 0.3) -> A.Lambda:
    return A.Lambda(image=_add_random_stripes, p=p)



# -----------------------
# Основные пайплайны
# -----------------------
def get_train_augmentations(img_size: int = 640) -> List[Callable]:
    """
    Возвращает список albumentations трансформов для обучения.
    Используем комбинацию лёгких эффектов + OneOf для тяжёлых, чтобы
    одно изображение не получало сразу всё.
    """
    aug_list = [
        # легкие размытия/шумы (иногда одно из трёх)
        A.OneOf(
            [
                A.MotionBlur(blur_limit=7, p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
                A.GaussianBlur(blur_limit=5, p=1.0),
            ],
            p=0.25,
        ),

        # дополнительные шумы/ISO-подобные шумы
        A.GaussNoise(var_limit=(5.0, 50.0), p=0.18),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.12),

        # цвет и контраст
        A.CLAHE(clip_limit=2.0, p=0.18),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.6),
        A.HueSaturationValue(hue_shift_limit=18, sat_shift_limit=30, val_shift_limit=12, p=0.45),
        A.RandomGamma(gamma_limit=(80, 120), p=0.25),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.25),
        A.ChannelShuffle(p=0.08),

        # мелкие искажения/перспектива (выбирается одно из сильных искажений)
        A.OneOf(
            [
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                A.OpticalDistortion(distort_limit=0.08, shift_limit=0.05, p=1.0),
                A.Perspective(scale=(0.02, 0.07), p=1.0),
                A.ElasticTransform(alpha=1.0, sigma=50, alpha_affine=20, p=1.0),
            ],
            p=0.25,
        ),

        # геометрические трансформации
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.12),
        A.RandomRotate90(p=0.15),
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.12, rotate_limit=18, p=0.5),

        # эффект "царапин/полосок" — кастом
        Stripes(p=0.25),

        # имитация заслонений/отсечения
        A.CoarseDropout(max_holes=10, max_height=40, max_width=40, min_holes=1, min_height=8, min_width=8, p=0.2),

        # текстурные/фильтровые эффекты (иногда одно из них)
        A.OneOf(
            [
                A.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.3), p=1.0),
                A.Emboss(alpha=(0.1, 0.3), strength=(0.1, 0.3), p=1.0),
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=1.0),
                A.RandomSunFlare(src_radius=100, p=1.0),
            ],
            p=0.18,
        ),

        # финальный ресайз к целевому размеру
        A.Resize(img_size, img_size, p=1.0),
    ]
    return aug_list


def get_val_augmentations(img_size: int = 640) -> List[Callable]:
    """
    Набор для валидации — минимальный: ресайз.
    """
    return [A.Resize(img_size, img_size, p=1.0)]


def get_test_augmentations(img_size: int = 640) -> List[Callable]:
    return get_val_augmentations(img_size)
