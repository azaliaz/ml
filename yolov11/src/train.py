# src/train.py
"""
Основной training script для YOLOv11:
- загружает config
- запускает обучение
- логирует метрики в MLflow
- применяет кастомные Albumentations аугментации для обучения

Запуск:
    python src/train.py --config configs/train.yaml
Опции:
    --weights, --epochs, --batch, --img_size
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import yaml
import logging
import mlflow
import mlflow.pytorch
import torch
from typing import Optional, Dict, Any
import csv
import os

# -------------------------
# Новый импорт кастомных аугментаций
from src import augmentations as augs
# -------------------------

from src.model import YOLOv11Model
from src.metrics import extract_metrics_from_ultralytics, log_metrics_mlflow

# MLflow store
mlflow_store = Path.cwd() / "runs" / "mlflow"
mlflow_store.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri(f"file:{mlflow_store}")
os.environ["YOLO_DISABLE_MLFLOW"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# -------------------------
# Вспомогательные функции
# -------------------------
def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def resolve_num_classes(cfg: dict, data_yaml_path: Optional[Path]) -> Optional[int]:
    if cfg.get("num_classes") is not None:
        return int(cfg["num_classes"])
    if data_yaml_path and data_yaml_path.exists():
        try:
            dd = load_yaml(data_yaml_path)
            if dd.get("nc") is not None:
                return int(dd["nc"])
        except Exception:
            logger.warning("Не удалось прочитать nc из data_yaml: %s", data_yaml_path)
    return None

def _sanitize_colname(col: str) -> str:
    name = col.strip()
    for ch in ["/", " ", ":", ","]:
        name = name.replace(ch, "_")
    name = name.replace("(", "").replace(")", "")
    while "__" in name:
        name = name.replace("__", "_")
    return name.strip("_")

def log_epoch_metrics_from_results_csv(results_csv: Path):
    if not results_csv.exists():
        logger.info("results.csv не найден: %s — пропускаю логирование по эпохам.", results_csv)
        return
    logger.info("Читаю результаты по эпохам из %s", results_csv)
    metric_tokens = {
        "precision": ["precision"],
        "recall": ["recall"],
        "map50": ["map50", "map@0.5", "mAP50"],
        "map50_95": ["map50_95", "map@0.5:0.95", "mAP", "map"],
    }
    with results_csv.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        headers = reader.fieldnames or []
        headers_l = [h.lower() for h in headers]

        found_cols: Dict[str, Optional[str]] = {k: None for k in metric_tokens.keys()}
        for metric, tokens in metric_tokens.items():
            for h, hl in zip(headers, headers_l):
                if any(tok.lower() in hl for tok in tokens):
                    found_cols[metric] = h
                    break

        epoch_col = None
        for h, hl in zip(headers, headers_l):
            if hl == "epoch" or hl.startswith("epoch"):
                epoch_col = h
                break
        if epoch_col is None and "epoch" in headers_l:
            epoch_col = headers[headers_l.index("epoch")]

        for row in reader:
            epoch = None
            if epoch_col is not None:
                try:
                    epoch = int(float(row.get(epoch_col, 0)))
                except Exception:
                    epoch = None
            if epoch is None:
                epoch = None

            logged = {}
            for metric, col in found_cols.items():
                if col is None:
                    continue
                raw = row.get(col, "")
                if raw == "" or raw is None:
                    continue
                try:
                    val = float(raw)
                except Exception:
                    try:
                        val = float("".join(ch for ch in raw if (ch.isdigit() or ch in ".-eE")))
                    except Exception:
                        continue
                mlflow.log_metric(metric, val, step=epoch)
                logged[metric] = val

            if ("precision" in logged) and ("recall" in logged):
                p = logged["precision"]
                r = logged["recall"]
                f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
                mlflow.log_metric("f1", float(f1), step=epoch)

            for col in headers:
                if col == epoch_col or col in found_cols.values():
                    continue
                raw = row.get(col, "")
                if raw == "" or raw is None:
                    continue
                try:
                    val = float(raw)
                except Exception:
                    continue
                name = _sanitize_colname(col)
                if name in logged:
                    name = f"aux_{name}"
                mlflow.log_metric(name, val, step=epoch)

# -------------------------
# Main
# -------------------------
def main(cfg_path: str, override: dict[str, Any] | None = None):
    cfg_path = Path(cfg_path)
    cfg = load_yaml(cfg_path)
    override = override or {}

    mlflow_cfg = cfg.get("mlflow", {})
    train_cfg = cfg.get("train", {})

    if "epochs" in override:
        train_cfg["epochs"] = int(override["epochs"])
    if "batch_size" in override:
        train_cfg["batch_size"] = int(override["batch_size"])
    if "img_size" in override:
        train_cfg["img_size"] = int(override["img_size"])
    if "weights" in override:
        cfg.setdefault("model", {})["weights"] = str(override["weights"])

    data_yaml_path = Path(train_cfg.get("data_yaml", "")) if train_cfg.get("data_yaml") else None
    num_classes = resolve_num_classes(cfg, data_yaml_path)
    if num_classes is None:
        logger.warning("num_classes не найден в configs/train.yaml или в data_yaml. Продолжим, но проверьте.")
    else:
        logger.info("num_classes = %s", num_classes)

    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "yolov11"))
    run = None
    try:
        run = mlflow.start_run(run_name=mlflow_cfg.get("run_name"))
        logger.info("MLflow run started: %s", run.info.run_id)
        try:
            mlflow.log_artifact(str(cfg_path), artifact_path="configs")
        except Exception as e:
            logger.warning("Не удалось залогировать configs/train.yaml: %s", e)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        mlflow.log_param("device", device)
        logger.info("Device: %s", device)

        model_cfg = cfg.get("model", {})
        weights_path = model_cfg.get("weights", "yolov11n.pt")
        if not Path(weights_path).exists():
            logger.info("Weights file '%s' не найден локально — Ultralytics будет пытаться autoload.", weights_path)
            mlflow.log_param("weights_autoload", True)
        else:
            mlflow.log_param("weights_local", str(weights_path))

        model = YOLOv11Model(
            weights=weights_path,
            num_classes=num_classes if num_classes is not None else 0,
            device=device,
        )
        logger.info("Model loaded. num_classes_from_cfg=%s num_classes_from_weights=%s",
                    num_classes, getattr(model, "num_classes", None))

        # -------------------------
        # Подключение кастомных аугментаций
        # -------------------------
        img_size = train_cfg.get("img_size", 640)
        use_custom_augs = not bool(train_cfg.get("disable_custom_augs", False))
        if use_custom_augs:
            try:
                custom_transforms = augs.get_train_augmentations(img_size=img_size)
                train_kwargs = dict(
                    data=train_cfg["data_yaml"],
                    epochs=train_cfg.get("epochs", 50),
                    imgsz=img_size,
                    batch=train_cfg.get("batch_size", 16),
                    lr0=train_cfg.get("lr", 1e-3),
                    optimizer=train_cfg.get("optimizer", "AdamW"),
                    weight_decay=train_cfg.get("weight_decay", 5e-4),
                    device=device,
                    project=train_cfg.get("project", "runs"),
                    name=train_cfg.get("name", "exp"),
                    exist_ok=True,
                    augmentations=custom_transforms,  # <- здесь передаем кастом
                )
                logger.info("Custom Albumentations transforms attached to training.")
            except Exception as e:
                logger.warning("Не удалось создать custom augmentations: %s — пропускаю.", e)
        else:
            train_kwargs = dict(
                data=train_cfg["data_yaml"],
                epochs=train_cfg.get("epochs", 50),
                imgsz=img_size,
                batch=train_cfg.get("batch_size", 16),
                lr0=train_cfg.get("lr", 1e-3),
                optimizer=train_cfg.get("optimizer", "AdamW"),
                weight_decay=train_cfg.get("weight_decay", 5e-4),
                device=device,
                project=train_cfg.get("project", "runs"),
                name=train_cfg.get("name", "exp"),
                exist_ok=True,
            )

        # Логирование параметров MLflow
        flat_params = {k: v for k, v in train_kwargs.items() if isinstance(v, (int, float, str, bool))}
        mlflow.log_params(flat_params)
        logger.info("Training with params: %s", flat_params)

        # -------------------------
        # Запуск обучения
        # -------------------------
        logger.info("Start training...")
        results = model.train(**train_kwargs)
        logger.info("Training finished.")

        try:
            metrics = extract_metrics_from_ultralytics(results)
            log_metrics_mlflow(metrics)
            logger.info("Final metrics: %s", metrics)
        except Exception as e:
            logger.warning("Не удалось извлечь/заложировать финальные метрики: %s", e)
            metrics = {}

        results_csv = Path(train_kwargs["project"]) / train_kwargs["name"] / "results.csv"
        try:
            log_epoch_metrics_from_results_csv(results_csv)
            if results_csv.exists():
                mlflow.log_artifact(str(results_csv), artifact_path="metrics")
        except Exception as e:
            logger.warning("Ошибка при логировании по эпохам из results.csv: %s", e)

        weights_dir = Path(train_kwargs["project"]) / train_kwargs["name"] / "weights"
        if weights_dir.exists():
            try:
                mlflow.log_artifacts(str(weights_dir), artifact_path="weights")
                logger.info("Logged weights artifacts from %s", weights_dir)
            except Exception as e:
                logger.warning("Не удалось залогировать веса: %s", e)
        else:
            logger.info("Weights dir not found: %s", weights_dir)

        if metrics:
            print("Training finished. Metrics:")
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (KeyboardInterrupt).")
        raise
    except Exception as e:
        logger.exception("Ошибка во время обучения: %s", e)
        raise
    finally:
        try:
            if run is not None:
                mlflow.end_run()
                logger.info("MLflow run ended.")
        except Exception as e:
            logger.warning("Ошибка при завершении MLflow run: %s", e)

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to train.yaml")
    parser.add_argument("--weights", type=str, default=None, help="Override weights path/name")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--img_size", type=int, default=None, help="Override img size")
    args = parser.parse_args()

    overrides: dict[str, Any] = {}
    if args.weights:
        overrides["weights"] = args.weights
    if args.epochs:
        overrides["epochs"] = args.epochs
    if args.batch:
        overrides["batch_size"] = args.batch
    if args.img_size:
        overrides["img_size"] = args.img_size

    main(args.config, override=overrides)
