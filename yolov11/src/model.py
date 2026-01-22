"""
src/model.py

Обёртка над Ultralytics YOLO для использования в MLflow-пайплайне.

Особенности:
 - Автозагрузка предобученных весов (если указано имя модели, например "yolov11n.pt" / "yolov11n")
 - Попытка извлечь число классов (nc) из весов и предупреждение при несоответствии
 - Единый интерфейс: train / val / predict / save / load
"""

from pathlib import Path
from typing import Optional, Any, Union
import warnings

import torch
from ultralytics import YOLO


class YOLOv11Model:
    def __init__(self, weights: Union[str, Path], num_classes: Optional[int] = None, device: Optional[str] = None) -> None:
        """
        Args:
            weights: путь к .pt файлу или имя предобученной модели (например 'yolov11n.pt' или 'yolov11n')
            num_classes: желаемое число классов (int) или None, тогда будет попытка прочесть из весов
            device: 'cpu', 'cuda', 'mps' или None (авто)
        """
        self.weights = str(weights)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Если локальный файл не найден, попробуем использовать имя модели для автозагрузки
        wpath = Path(self.weights)
        if not wpath.exists():
            # пользователю полезно знать, что будет скачано
            print(f"Weights file '{self.weights}' not found locally — will try to load '{self.weights}' via ultralytics (auto-download).")

        # Попытка загрузить модель (Ultralytics сам скачает веса, если это имя модели)
        try:
            self.model = YOLO(self.weights)
        except Exception as e:
            raise RuntimeError(f"Не удалось загрузить модель/веса '{self.weights}': {e}") from e

        # Попытка прочитать число классов из весов/объекта модели
        nc_from_weights = None
        try:
            if hasattr(self.model, "model") and getattr(self.model.model, "args", None):
                nc_from_weights = (self.model.model.args or {}).get("nc", None)
            elif getattr(self.model, "args", None):
                nc_from_weights = (self.model.args or {}).get("nc", None)
        except Exception:
            nc_from_weights = None

        try:
            nc_from_weights = int(nc_from_weights) if nc_from_weights is not None else None
        except Exception:
            nc_from_weights = None

        # Установим self.num_classes: приоритет аргумента, затем инфо из весов, иначе None
        if num_classes is not None:
            self.num_classes = int(num_classes)
        else:
            self.num_classes = nc_from_weights

        # Предупреждение при несоответствии
        if nc_from_weights is not None and self.num_classes is not None and nc_from_weights != self.num_classes:
            warnings.warn(
                f"num_classes ({self.num_classes}) != nc in weights ({nc_from_weights}). "
                "Если вы дообучаете на другом количестве классов — убедитесь, что голова модели скорректирована."
            )

        # Попытка переместить модель на устройство
        try:
            self.model.to(self.device)
        except Exception:
            warnings.warn(f"Не удалось переместить модель на устройство '{self.device}'.")

    # -------------------------
    # Основной интерфейс
    # -------------------------
    def train(self, **kwargs) -> Any:
        """Запуск обучения. Аргументы прокидываются в ultralytics.YOLO.train()."""
        return self.model.train(**kwargs)

    def val(self, **kwargs) -> Any:
        """Валидация / оценка. Аргументы прокидываются в ultralytics.YOLO.val()."""
        return self.model.val(**kwargs)

    def predict(self, **kwargs) -> Any:
        """Инференс. Аргументы прокидываются в ultralytics.YOLO.predict()."""
        return self.model.predict(**kwargs)

    def save(self, path: Union[str, Path]) -> None:
        """Сохраняет веса модели в указанный путь."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.model.save(str(path))
        except Exception as e:
            raise RuntimeError(f"Ошибка при сохранении модели в '{path}': {e}") from e

    # -------------------------
    # Утилиты загрузки
    # -------------------------
    @staticmethod
    def load(path: Union[str, Path], device: Optional[str] = None) -> "YOLOv11Model":
        """
        Загружает модель из весов или имени модели.
        Возвращает обёртку; num_classes попытаемся извлечь из весов.
        """
        path = Path(path)
        # позволим YOLO сам разрулить (локальный файл или автозагрузка)
        model_obj = YOLO(str(path))

        # попытка извлечь nc
        nc = None
        try:
            if hasattr(model_obj, "model") and getattr(model_obj.model, "args", None):
                nc = (model_obj.model.args or {}).get("nc", None)
        except Exception:
            nc = None
        if nc is None:
            try:
                nc = (model_obj.args or {}).get("nc", None) if getattr(model_obj, "args", None) else None
            except Exception:
                nc = None
        try:
            nc = int(nc) if nc is not None else None
        except Exception:
            nc = None

        wrapper = YOLOv11Model.__new__(YOLOv11Model)
        wrapper.weights = str(path)
        wrapper.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        wrapper.model = model_obj
        wrapper.num_classes = nc

        try:
            wrapper.model.to(wrapper.device)
        except Exception:
            pass
        return wrapper

    # useful helpers
    @property
    def names(self):
        """Список имён классов, если доступно."""
        return getattr(self.model, "names", None)


if __name__ == "__main__":
    # Быстрая проверка
    m = YOLOv11Model(weights="yolov11n.pt", num_classes=1, device="cpu")
    print("Model loaded. Weights:", m.weights, "num_classes:", m.num_classes)
