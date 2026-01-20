"""
src/metrics.py

Явный расчёт метрик детекции для YOLO-пайплайна:
- precision
- recall
- F1-score
- mAP@0.5
- mAP@0.5:0.95

Использует ultralytics Metrics, но возвращает значения
в удобном для MLflow формате (dict).
"""

from typing import Dict, Any


def extract_metrics_from_ultralytics(results) -> Dict[str, float]:
    """
    Извлекает ключевые метрики из результата YOLO.val().

    Args:
        results: объект, возвращаемый ultralytics.YOLO.val()

    Returns:
        dict с метриками
    """
    # Ultralytics возвращает объект с атрибутом .box
    box = results.box

    metrics = {
        "precision": float(box.mp),        # mean precision
        "recall": float(box.mr),           # mean recall
        "map50": float(box.map50),         # mAP@0.5
        "map50_95": float(box.map),        # mAP@0.5:0.95
    }

    # F1 = 2PR / (P + R)
    p = metrics["precision"]
    r = metrics["recall"]
    metrics["f1"] = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)

    return metrics


def log_metrics_mlflow(metrics: Dict[str, float], step: int | None = None):
    """
    Утилита для логирования метрик в MLflow.
    """
    import mlflow

    for k, v in metrics.items():
        mlflow.log_metric(k, v, step=step)
