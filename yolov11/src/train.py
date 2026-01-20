"""
src/train.py

Основной training script:
- загружает config
- запускает обучение YOLOv11n
- считает и логирует метрики
- сохраняет артефакты в MLflow

Запуск:
    python src/train.py --config configs/train.yaml
"""

import argparse
from pathlib import Path
import yaml
import mlflow
import mlflow.pytorch

import torch

from src.model import YOLOv11Model
from src.metrics import extract_metrics_from_ultralytics, log_metrics_mlflow



def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(cfg_path: str):
    cfg = load_yaml(cfg_path)

    mlflow_cfg = cfg.get("mlflow", {})
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "yolov11"))

    with mlflow.start_run(run_name=mlflow_cfg.get("run_name")):
        mlflow.log_artifact(cfg_path, artifact_path="configs")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        mlflow.log_param("device", device)

        model_cfg = cfg["model"]

        weights_path = model_cfg.get("weights", "yolov11n.pt")


        model = YOLOv11Model(
            weights=weights_path,
            num_classes=cfg["num_classes"],
            device=device,
        )

        train_cfg = cfg["train"]

        train_kwargs = dict(
            data=train_cfg["data_yaml"],
            epochs=train_cfg.get("epochs", 50),
            imgsz=train_cfg.get("img_size", 640),
            batch=train_cfg.get("batch_size", 16),
            lr0=train_cfg.get("lr", 1e-3),
            optimizer=train_cfg.get("optimizer", "AdamW"),
            weight_decay=train_cfg.get("weight_decay", 5e-4),
            device=device,
            project=train_cfg.get("project", "runs"),
            name=train_cfg.get("name", "exp"),
            exist_ok=True,
        )

        # log hyperparameters
        # логируем все параметры (без вложенных dict/list)
        flat_params = {k: v for k, v in train_kwargs.items() if isinstance(v, (int, float, str, bool))}
        mlflow.log_params(flat_params)

        results = model.train(**train_kwargs)


        metrics = extract_metrics_from_ultralytics(results)
        log_metrics_mlflow(metrics)

        weights_dir = Path(train_kwargs["project"]) / train_kwargs["name"] / "weights"
        if weights_dir.exists():
            mlflow.log_artifacts(str(weights_dir), artifact_path="weights")

        print("Training finished. Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to train.yaml")
    args = parser.parse_args()

    main(args.config)
