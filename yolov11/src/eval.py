"""
src/eval.py

Evaluation / testing script for YOLOv11n:
- загружает обученные веса (best.pt / last.pt)
- прогоняет валидацию или тест
- считает метрики (precision, recall, F1, mAP)
- логирует результаты в MLflow

Запуск:
    python src/eval.py --config configs/train.yaml --weights runs/yolov11_sku110/weights/best.pt --split val

Или для теста:
    python src/eval.py --config configs/train.yaml --weights runs/yolov11_sku110/weights/best.pt --split test
"""

import argparse
from pathlib import Path
import yaml
import mlflow
import torch

from src.model import YOLOv11Model
from src.metrics import extract_metrics_from_ultralytics, log_metrics_mlflow


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)



def main(cfg_path: str, weights_path: str, split: str):
    cfg = load_yaml(cfg_path)

    mlflow_cfg = cfg.get("mlflow", {})
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "yolov11"))

    run_name = f"eval_{Path(weights_path).stem}_{split}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("eval_split", split)
        mlflow.log_param("weights", weights_path)

        # log config
        mlflow.log_artifact(cfg_path, artifact_path="configs")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        mlflow.log_param("device", device)

        model = YOLOv11Model.load(weights_path, device=device)

        train_cfg = cfg["train"]

        val_kwargs = dict(
            data=train_cfg["data_yaml"],
            imgsz=train_cfg.get("img_size", 640),
            batch=train_cfg.get("batch_size", 16),
            device=device,
            split=split,   # 'val' or 'test'
            save=False,
            verbose=False,
        )

        results = model.val(**val_kwargs)

        if not hasattr(results, "box") and isinstance(results, dict):
            import warnings
            warnings.warn(
                "Unexpected results structure from ultralytics; attempting fallback parsing."
            )

        metrics = extract_metrics_from_ultralytics(results)
        log_metrics_mlflow(metrics)

        import json
        mlflow.log_metrics(metrics)  # дублируем для надежности
        with open("metrics_eval.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact("metrics_eval.json", artifact_path="metrics")

        mlflow.log_artifact(weights_path, artifact_path="weights")

        print(f"Evaluation finished on split='{split}'. Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to train.yaml")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights (.pt)")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split")
    args = parser.parse_args()

    main(args.config, args.weights, args.split)
