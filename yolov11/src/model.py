"""
src/model.py

Обёртка над Ultralytics YOLOv11n для использования в MLflow-пайплайне.

Задачи модуля:
- Загрузка предобученных весов (yolov11n.pt)
- Корректная инициализация под нужное число классов
- Единый интерфейс: train / val / predict
- Сохранение и загрузка весов

Важно:
Ultralytics YOLO сам считает loss и метрики, но мы используем эту обёртку
для явного контроля в train.py / eval.py и логирования в MLflow.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Union
import warnings

import torch
from ultralytics import YOLO


class YOLOv11Model:
    """
    High-level wrapper around ultralytics.YOLO.
    """

    def __init__(
        self,
        weights: Union[str, Path],
        num_classes: int,
        device: Optional[str] = None,
    ) -> None:
        """
        Args:
            weights: путь к .pt файлу или имя предобученной модели (например 'yolov11n.pt')
            num_classes: количество классов датасета
            device: 'cpu', 'cuda', 'mps' или None (авто)
        """
        self.weights = str(weights)
        self.num_classes = int(num_classes)
        self.device = device

        # Попытка загрузить модель (Ultralytics сам скачает веса при необходимости)
        self.model = YOLO(self.weights)
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

        if nc_from_weights is not None and nc_from_weights != self.num_classes:
            warnings.warn(
                f"num_classes ({self.num_classes}) != nc from weights ({nc_from_weights}). "
                "Убедитесь, что это ожидаемо; иначе возможно несоответствие головы модели.",
            )

        # Опционально: принудительно установить nc в объекте (закомментировано — включайте
        # только если уверены в API вашей версии ultralytics)
        # try:
        #     if hasattr(self.model, "model") and getattr(self.model.model, "args", None) is not None:
        #         self.model.model.args['nc'] = int(self.num_classes)
        # except Exception:
        #     warnings.warn("Не удалось принудительно установить nc в объект модели; пропускаю.")

        # Перемещаем модель на устройство, если указано
        if self.device is not None:
            try:
                self.model.to(self.device)
            except Exception:
                warnings.warn(f"Не удалось переместить модель на устройство '{self.device}'.")

    def train(self, **kwargs) -> Any:
        """
        Запуск обучения. Все аргументы прокидываются в ultralytics.YOLO.train().
        Возвращаемый тип зависит от версии ultralytics.
        """
        return self.model.train(**kwargs)

    def val(self, **kwargs) -> Any:
        """
        Запуск валидации. Аргументы прокидываются в YOLO.val().
        """
        return self.model.val(**kwargs)

    def predict(self, **kwargs) -> Any:
        """
        Инференс. Пример:
            model.predict(source='images/', conf=0.25, iou=0.5)
        """
        return self.model.predict(**kwargs)


    def save(self, path: Union[str, Path]) -> None:
        """Сохраняет веса модели в указанный путь."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:

            self.model.save(str(path))
        except Exception as e:
            raise RuntimeError(f"Ошибка при сохранении модели в '{path}': {e}") from e

    @staticmethod
    def load(path: Union[str, Path], device: Optional[str] = None) -> "YOLOv11Model":
        """
        Загружает модель из весов.
        Возвращает обёртку; num_classes будет попытка извлечена из весов (или None).
        """
        path = Path(path)
        model = YOLO(str(path))


        nc = None
        try:
            if hasattr(model, "model") and getattr(model.model, "args", None):
                nc = (model.model.args or {}).get("nc", None)
        except Exception:
            nc = None

        if nc is None:
            try:
                if getattr(model, "args", None):
                    nc = (model.args or {}).get("nc", None)
            except Exception:
                nc = None

        try:
            nc = int(nc) if nc is not None else None
        except Exception:
            nc = None

        wrapper = YOLOv11Model.__new__(YOLOv11Model)
        wrapper.weights = str(path)
        wrapper.num_classes = nc
        wrapper.device = device
        wrapper.model = model

        if device is not None:
            try:
                wrapper.model.to(device)
            except Exception:
                warnings.warn(f"Не удалось переместить модель на устройство '{device}'.")

        return wrapper

    @property
    def names(self):
        """Возвращает список имён классов (если доступно)."""
        return getattr(self.model, "names", None)

    def export(self, **kwargs) -> Any:
        """
        Экспорт модели (onnx, torchscript, etc.).
        Пример: model.export(format='onnx', opset=12)
        """
        return self.model.export(**kwargs)


if __name__ == "__main__":
    m = YOLOv11Model(
        weights="yolov11n.pt",
        num_classes=1,
        device="cpu",
    )
    print("Model loaded. Classes:", m.num_classes)
