#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


@dataclass
class Pair:
    sample_id: str
    gt_path: Path
    pred_path: Path | None
    scenario: str = "default"
    latency_ms: float | None = None


def load_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    return arr > 0


def binary_metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    tp = float(np.logical_and(gt, pred).sum())
    fp = float(np.logical_and(~gt, pred).sum())
    fn = float(np.logical_and(gt, ~pred).sum())

    iou_den = tp + fp + fn
    iou = tp / iou_den if iou_den > 0 else 1.0

    dice_den = 2.0 * tp + fp + fn
    dice = (2.0 * tp) / dice_den if dice_den > 0 else 1.0

    precision_den = tp + fp
    precision = tp / precision_den if precision_den > 0 else 1.0

    recall_den = tp + fn
    recall = tp / recall_den if recall_den > 0 else 1.0

    return {"iou": iou, "dice": dice, "precision": precision, "recall": recall}


def boundary_f1(gt: np.ndarray, pred: np.ndarray, tolerance_px: int = 2) -> float:
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch for boundary F1: {gt.shape} != {pred.shape}")

    gt_b = np.logical_xor(gt, ndi.binary_erosion(gt))
    pr_b = np.logical_xor(pred, ndi.binary_erosion(pred))

    if gt_b.sum() == 0 and pr_b.sum() == 0:
        return 1.0
    if gt_b.sum() == 0 or pr_b.sum() == 0:
        return 0.0

    struct = ndi.generate_binary_structure(2, 1)
    gt_d = ndi.binary_dilation(gt_b, structure=struct, iterations=tolerance_px)
    pr_d = ndi.binary_dilation(pr_b, structure=struct, iterations=tolerance_px)

    matched_pred = np.logical_and(pr_b, gt_d).sum()
    matched_gt = np.logical_and(gt_b, pr_d).sum()

    p = matched_pred / pr_b.sum() if pr_b.sum() > 0 else 0.0
    r = matched_gt / gt_b.sum() if gt_b.sum() > 0 else 0.0
    den = p + r
    return (2.0 * p * r / den) if den > 0 else 0.0


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * q
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    return values[f] + (values[c] - values[f]) * (k - f)


def resolve_pairs_from_dirs(gt_dir: Path, pred_dir: Path, exts: tuple[str, ...]) -> list[Pair]:
    gt_files = [p for p in gt_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    pairs: list[Pair] = []
    for gt in gt_files:
        rel = gt.relative_to(gt_dir)
        pred = pred_dir / rel
        sample_id = rel.as_posix()
        pairs.append(Pair(sample_id=sample_id, gt_path=gt, pred_path=pred if pred.exists() else None))
    return pairs


def resolve_pairs_from_csv(csv_path: Path, exts: tuple[str, ...]) -> list[Pair]:
    pairs: list[Pair] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"sample_id", "gt_path", "pred_path"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"{csv_path} must contain columns: {sorted(required)}")
        for row in reader:
            gt = Path(row["gt_path"]).expanduser().resolve()
            pred_raw = (row.get("pred_path") or "").strip()
            pred = Path(pred_raw).expanduser().resolve() if pred_raw else None
            if gt.suffix.lower() not in exts:
                continue
            scenario = (row.get("scenario") or "default").strip() or "default"
            lat = row.get("latency_ms")
            latency_ms = float(lat) if lat and lat.strip() else None
            pairs.append(
                Pair(
                    sample_id=(row.get("sample_id") or gt.name).strip() or gt.name,
                    gt_path=gt,
                    pred_path=pred,
                    scenario=scenario,
                    latency_ms=latency_ms,
                )
            )
    return pairs


def mean(xs: Iterable[float]) -> float:
    arr = list(xs)
    return float(sum(arr) / len(arr)) if arr else float("nan")


def print_group_summary(name: str, rows: list[dict[str, float | str | bool]]) -> None:
    ok = [r for r in rows if not r["failed"]]
    failed = [r for r in rows if r["failed"]]
    lat = sorted([float(r["latency_ms"]) for r in ok if r["latency_ms"] is not None])
    ious = [float(r["iou"]) for r in ok]
    dices = [float(r["dice"]) for r in ok]
    precs = [float(r["precision"]) for r in ok]
    recs = [float(r["recall"]) for r in ok]
    bfs = [float(r["boundary_f1"]) for r in ok]

    print(f"\n[{name}]")
    print(f"  samples_total: {len(rows)}")
    print(f"  samples_ok:    {len(ok)}")
    print(f"  failures:      {len(failed)} ({(len(failed)/len(rows)*100.0):.1f}%)")
    if ok:
        print(f"  mean_iou:      {mean(ious):.4f}")
        print(f"  mean_dice:     {mean(dices):.4f}")
        print(f"  mean_precision:{mean(precs):.4f}")
        print(f"  mean_recall:   {mean(recs):.4f}")
        print(f"  mean_bf1:      {mean(bfs):.4f}")
    if lat:
        print(f"  latency_p50_ms:{median(lat):.2f}")
        p95 = pct(lat, 0.95)
        if p95 is not None:
            print(f"  latency_p95_ms:{p95:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SAM3 masks against GT masks.")
    parser.add_argument("--gt-dir", type=Path, help="Directory with GT masks.")
    parser.add_argument("--pred-dir", type=Path, help="Directory with predicted masks (same relative paths as gt).")
    parser.add_argument("--pairs-csv", type=Path, help="Optional CSV with columns: sample_id,gt_path,pred_path,scenario,latency_ms")
    parser.add_argument("--boundary-tolerance-px", type=int, default=2, help="Boundary matching tolerance in pixels.")
    parser.add_argument("--per-image-out", type=Path, default=Path("sam3_eval_per_image.csv"))
    parser.add_argument(
        "--ignore-missing-pred",
        action="store_true",
        help="Skip samples without predicted mask instead of counting them as failed.",
    )
    args = parser.parse_args()

    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    if args.pairs_csv is None and (args.gt_dir is None or args.pred_dir is None):
        raise SystemExit("Provide either --pairs-csv OR both --gt-dir and --pred-dir")

    if args.pairs_csv is not None:
        pairs = resolve_pairs_from_csv(args.pairs_csv, exts)
    else:
        pairs = resolve_pairs_from_dirs(args.gt_dir.resolve(), args.pred_dir.resolve(), exts)

    if not pairs:
        raise SystemExit("No mask pairs found.")

    rows: list[dict[str, float | str | bool]] = []
    for p in pairs:
        if args.ignore_missing_pred and (p.pred_path is None or (not p.pred_path.exists())):
            continue
        row: dict[str, float | str | bool] = {
            "sample_id": p.sample_id,
            "scenario": p.scenario,
            "failed": False,
            "latency_ms": p.latency_ms if p.latency_ms is not None else None,
            "iou": float("nan"),
            "dice": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "boundary_f1": float("nan"),
            "failure_reason": "",
        }
        try:
            if not p.gt_path.exists():
                raise FileNotFoundError(f"gt missing: {p.gt_path}")
            if p.pred_path is None or (not p.pred_path.exists()):
                raise FileNotFoundError(f"pred missing: {p.pred_path}")
            gt = load_mask(p.gt_path)
            pred = load_mask(p.pred_path)
            if gt.shape != pred.shape:
                raise ValueError(f"shape mismatch gt={gt.shape} pred={pred.shape}")
            m = binary_metrics(gt, pred)
            bf1 = boundary_f1(gt, pred, tolerance_px=args.boundary_tolerance_px)
            row.update(m)
            row["boundary_f1"] = bf1
        except Exception as e:
            row["failed"] = True
            row["failure_reason"] = str(e)
        rows.append(row)

    args.per_image_out.parent.mkdir(parents=True, exist_ok=True)
    with args.per_image_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "scenario",
                "failed",
                "failure_reason",
                "latency_ms",
                "iou",
                "dice",
                "precision",
                "recall",
                "boundary_f1",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("SAM3 evaluation summary")
    print(f"per-image report: {args.per_image_out}")
    print_group_summary("overall", rows)

    scenarios = sorted({str(r["scenario"]) for r in rows})
    if len(scenarios) > 1:
        for sc in scenarios:
            print_group_summary(f"scenario={sc}", [r for r in rows if r["scenario"] == sc])


if __name__ == "__main__":
    main()
