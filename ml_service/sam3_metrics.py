"""Mask quality metrics for SAM3 / GND+SAM evaluation (no model deps)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class MaskPair:
    sample_id: str
    gt_path: Path
    pred_path: Path | None
    scenario: str = "default"
    latency_ms: float | None = None
    prompt: str = ""
    strategy: str = ""


def load_mask(path: Path) -> np.ndarray:
    """Load a binary mask from PNG or NPY."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        return (arr > 0).astype(np.uint8)
    if Image is None:
        raise RuntimeError("PIL is required to load image masks")
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 0).astype(np.uint8)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = a > 0
    bb = b > 0
    inter = float(np.logical_and(aa, bb).sum())
    union = float(np.logical_or(aa, bb).sum())
    if union <= 0.0:
        return 1.0 if inter <= 0.0 else 0.0
    return inter / union


def mask_dice(a: np.ndarray, b: np.ndarray) -> float:
    aa = a > 0
    bb = b > 0
    inter = float(np.logical_and(aa, bb).sum())
    sa = float(aa.sum())
    sb = float(bb.sum())
    if sa + sb <= 0.0:
        return 1.0
    return (2.0 * inter) / (sa + sb)


def mask_precision_recall(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Precision/recall treating `a` as prediction and `b` as ground truth."""
    aa = a > 0
    bb = b > 0
    inter = float(np.logical_and(aa, bb).sum())
    pred_pos = float(aa.sum())
    gt_pos = float(bb.sum())
    precision = inter / pred_pos if pred_pos > 0 else (1.0 if gt_pos == 0 else 0.0)
    recall = inter / gt_pos if gt_pos > 0 else (1.0 if pred_pos == 0 else 0.0)
    return precision, recall


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    if cv2 is None:
        # 4-neighbor boundary without OpenCV
        padded = np.pad(m, 1, mode="constant", constant_values=0)
        inner = padded[1:-1, 1:-1]
        eroded = (
            padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        )
        return (inner.astype(bool) & ~eroded.astype(bool)).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(m, kernel, iterations=1)
    return ((m > 0) & (eroded == 0)).astype(np.uint8)


def boundary_f1(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    tolerance_px: int = 2,
) -> float:
    pb = _mask_boundary(pred)
    gb = _mask_boundary(gt)
    if pb.sum() == 0 and gb.sum() == 0:
        return 1.0
    if pb.sum() == 0 or gb.sum() == 0:
        return 0.0
    if cv2 is not None and tolerance_px > 0:
        k = 2 * tolerance_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        pb_d = cv2.dilate(pb, kernel)
        gb_d = cv2.dilate(gb, kernel)
        tp_p = float(np.logical_and(pb > 0, gb_d > 0).sum())
        tp_g = float(np.logical_and(gb > 0, pb_d > 0).sum())
    else:
        tp_p = float(np.logical_and(pb > 0, gb > 0).sum())
        tp_g = tp_p
    prec = tp_p / float(pb.sum()) if pb.sum() > 0 else 0.0
    rec = tp_g / float(gb.sum()) if gb.sum() > 0 else 0.0
    if prec + rec <= 0.0:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)


def align_masks(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop/pad prediction to GT spatial shape."""
    gt_bin = (gt > 0).astype(np.uint8)
    pred_bin = (pred > 0).astype(np.uint8)
    gh, gw = gt_bin.shape[:2]
    ph, pw = pred_bin.shape[:2]
    if (ph, pw) == (gh, gw):
        return pred_bin, gt_bin
    out = np.zeros((gh, gw), dtype=np.uint8)
    h = min(ph, gh)
    w = min(pw, gw)
    out[:h, :w] = pred_bin[:h, :w]
    return out, gt_bin


def evaluate_mask_pair(
    gt: np.ndarray,
    pred: np.ndarray,
    *,
    boundary_tolerance_px: int = 2,
) -> dict[str, float]:
    pred_a, gt_a = align_masks(pred, gt)
    iou = mask_iou(pred_a, gt_a)
    dice = mask_dice(pred_a, gt_a)
    precision, recall = mask_precision_recall(pred_a, gt_a)
    bf1 = boundary_f1(pred_a, gt_a, tolerance_px=boundary_tolerance_px)
    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "boundary_f1": bf1,
    }


def evaluate_pairs(
    pairs: Sequence[MaskPair],
    *,
    boundary_tolerance_px: int = 2,
    ignore_missing_pred: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        base: dict[str, Any] = {
            "sample_id": pair.sample_id,
            "scenario": pair.scenario,
            "prompt": pair.prompt,
            "strategy": pair.strategy,
            "gt_path": str(pair.gt_path),
            "pred_path": str(pair.pred_path) if pair.pred_path else "",
            "latency_ms": pair.latency_ms,
            "ok": False,
            "error": "",
        }
        if pair.pred_path is None or not Path(pair.pred_path).exists():
            if ignore_missing_pred:
                continue
            base["error"] = "missing_prediction"
            rows.append(base)
            continue
        try:
            gt = load_mask(pair.gt_path)
            pred = load_mask(Path(pair.pred_path))
            metrics = evaluate_mask_pair(
                gt, pred, boundary_tolerance_px=boundary_tolerance_px
            )
            base.update(metrics)
            base["ok"] = True
        except Exception as exc:
            base["error"] = str(exc)
        rows.append(base)
    return rows


def summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    ok_rows = [r for r in rows if r.get("ok")]
    failures = total - len(ok_rows)
    summary: dict[str, Any] = {
        "samples_total": total,
        "samples_ok": len(ok_rows),
        "failures": failures,
        "failure_rate": (failures / total) if total else 0.0,
    }
    for key in ("iou", "dice", "precision", "recall", "boundary_f1"):
        vals = [float(r[key]) for r in ok_rows if r.get(key) is not None]
        summary[f"mean_{key}"] = float(np.mean(vals)) if vals else None
    latencies = [
        float(r["latency_ms"])
        for r in ok_rows
        if r.get("latency_ms") is not None and str(r["latency_ms"]).strip() != ""
    ]
    if latencies:
        summary["latency_p50_ms"] = float(np.percentile(latencies, 50))
        summary["latency_p95_ms"] = float(np.percentile(latencies, 95))
    else:
        summary["latency_p50_ms"] = None
        summary["latency_p95_ms"] = None
    return summary


def write_per_image_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "scenario",
        "prompt",
        "strategy",
        "ok",
        "error",
        "iou",
        "dice",
        "precision",
        "recall",
        "boundary_f1",
        "latency_ms",
        "gt_path",
        "pred_path",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def resolve_pairs_from_csv(csv_path: Path) -> list[MaskPair]:
    pairs: list[MaskPair] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sid = str(row.get("sample_id", "")).strip()
            gt = str(row.get("gt_path", "")).strip()
            if not sid or not gt:
                continue
            pred_raw = str(row.get("pred_path", "")).strip()
            pairs.append(
                MaskPair(
                    sample_id=sid,
                    gt_path=Path(gt),
                    pred_path=Path(pred_raw) if pred_raw else None,
                    scenario=str(row.get("scenario", "default") or "default"),
                    latency_ms=float(row["latency_ms"]) if row.get("latency_ms") else None,
                    prompt=str(row.get("prompt", "") or ""),
                    strategy=str(row.get("strategy", "") or ""),
                )
            )
    return pairs


def resolve_pairs_from_dirs(gt_dir: Path, pred_dir: Path) -> list[MaskPair]:
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)
    pairs: list[MaskPair] = []
    for gt_path in sorted(gt_dir.rglob("*")):
        if not gt_path.is_file():
            continue
        if gt_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".npy"}:
            continue
        rel = gt_path.relative_to(gt_dir)
        pred_path = pred_dir / rel
        if pred_path.suffix.lower() not in {".png", ".npy"}:
            pred_path = pred_path.with_suffix(".png")
        pairs.append(
            MaskPair(
                sample_id=str(rel.with_suffix("")),
                gt_path=gt_path,
                pred_path=pred_path if pred_path.exists() else None,
                scenario="default",
            )
        )
    return pairs


def pick_best_pred_mask(
    pred_masks: Iterable[np.ndarray],
    gt: np.ndarray,
) -> tuple[np.ndarray | None, float]:
    """Choose prediction with highest IoU to GT (for multi-mask outputs)."""
    best_mask: np.ndarray | None = None
    best_iou = -1.0
    for mask in pred_masks:
        metrics = evaluate_mask_pair(gt, mask)
        iou = float(metrics["iou"])
        if iou > best_iou:
            best_iou = iou
            best_mask = (mask > 0).astype(np.uint8)
    return best_mask, best_iou
