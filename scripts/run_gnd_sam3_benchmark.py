#!/usr/bin/env python3
"""
Run GND+SAM3 (or SAM3-native) on a benchmark manifest and compute mask metrics.

Server example:
  export SAM_BACKEND=sam3_native
  export PYTHONPATH=/home/agazizova/gitml/ml:$PYTHONPATH
  python scripts/run_gnd_sam3_benchmark.py \\
    --manifest benchmarks/coco_val_subset/manifest.jsonl \\
    --strategy both \\
    --score-threshold 0.3 \\
    --out-dir benchmarks/runs/run_001
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    samples: list[dict[str, Any]] = []
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "samples" in data:
            samples = list(data["samples"])
        elif isinstance(data, list):
            samples = data
        else:
            raise ValueError("JSON manifest must be a list or {\"samples\": [...]}")
    else:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def _save_mask_png(mask: np.ndarray, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(((mask > 0).astype(np.uint8) * 255)).save(path)


def _run_strategy(
    strategy: str,
    image_path: Path,
    prompt: str,
    score_threshold: float,
    max_boxes: int,
) -> tuple[list[np.ndarray], float]:
    """Return (masks, latency_ms)."""
    from ml_service.inference import (
        run_inference_grounding_dino,
        run_inference_sam_multimask,
        run_text_guided_segmentation,
        run_text_guided_segmentation_native,
    )
    from ml_service.models import MODEL_STORE

    t0 = time.perf_counter()
    masks: list[np.ndarray] = []

    if strategy == "sam3_native":
        rows = run_text_guided_segmentation_native(
            str(image_path),
            [prompt],
            score_threshold=score_threshold,
        )
        for row in rows:
            m = row.get("mask")
            if m is not None and int(np.asarray(m).sum()) > 0:
                masks.append((np.asarray(m) > 0).astype(np.uint8))
    elif strategy == "gnd_sam":
        boxes = run_inference_grounding_dino(
            str(image_path), prompt, score_threshold, max_boxes
        )
        for b in boxes:
            bbox = [int(v) for v in b["bbox"]]
            for cand in run_inference_sam_multimask(str(image_path), bbox, max_masks=3):
                m = cand.get("mask")
                if m is not None and int(np.asarray(m).sum()) > 0:
                    masks.append((np.asarray(m) > 0).astype(np.uint8))
        if not masks and MODEL_STORE.get("sam_automatic_generator") is not None:
            # Same fallback as production pipeline when GND finds nothing.
            rows = run_text_guided_segmentation(
                str(image_path),
                [prompt],
                score_threshold=score_threshold,
                max_boxes_per_prompt=max_boxes,
            )
            for row in rows:
                m = row.get("mask")
                if m is not None:
                    masks.append((np.asarray(m) > 0).astype(np.uint8))
    elif strategy == "gnd_sam_pipeline":
        rows = run_text_guided_segmentation(
            str(image_path),
            [prompt],
            score_threshold=score_threshold,
            max_boxes_per_prompt=max_boxes,
        )
        for row in rows:
            m = row.get("mask")
            if m is not None:
                masks.append((np.asarray(m) > 0).astype(np.uint8))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    latency_ms = (time.perf_counter() - t0) * 1000.0
    return masks, latency_ms


def _fmt_metric(value: Any, *, width: int = 8, pct: bool = False) -> str:
    if value is None:
        return f"{'n/a':>{width}}"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return f"{'n/a':>{width}}"
    if pct:
        return f"{(v * 100.0):>{width}.1f}%"
    return f"{v:>{width}.4f}"


def _summarize_grouped(rows: list[dict[str, Any]], key: str, summarize_fn: Any) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key) or "unknown"), []).append(row)
    return {name: summarize_fn(group_rows) for name, group_rows in sorted(groups.items())}


def _print_benchmark_report(
    *,
    manifest: Path,
    out_dir: Path,
    score_threshold: float,
    strategy_summaries: dict[str, dict[str, Any]],
    strategy_rows: dict[str, list[dict[str, Any]]],
    summarize_fn: Any,
) -> None:
    width = 88
    print("\n" + "=" * width)
    print(" BENCHMARK RESULTS".center(width))
    print("=" * width)
    print(f" Manifest : {manifest}")
    print(f" Output   : {out_dir.resolve()}")
    print(f" Threshold: {score_threshold}")

    headers = [
        ("Strategy", 18),
        ("OK/Total", 10),
        ("Fail%", 7),
        ("IoU", 8),
        ("Dice", 8),
        ("Prec", 8),
        ("Recall", 8),
        ("B-F1", 8),
        ("p50 ms", 9),
        ("p95 ms", 9),
    ]
    header_line = " ".join(h[0].ljust(h[1]) for h in headers)
    print("\n" + header_line)
    print("-" * len(header_line))

    for strategy, summary in strategy_summaries.items():
        total = int(summary.get("samples_total", 0))
        ok = int(summary.get("samples_ok", 0))
        row = [
            strategy[:18].ljust(18),
            f"{ok}/{total}".ljust(10),
            _fmt_metric(summary.get("failure_rate"), width=7, pct=True),
            _fmt_metric(summary.get("mean_iou")),
            _fmt_metric(summary.get("mean_dice")),
            _fmt_metric(summary.get("mean_precision")),
            _fmt_metric(summary.get("mean_recall")),
            _fmt_metric(summary.get("mean_boundary_f1")),
            _fmt_metric(summary.get("latency_p50_ms"), width=9),
            _fmt_metric(summary.get("latency_p95_ms"), width=9),
        ]
        print(" ".join(row))

    names = list(strategy_summaries.keys())
    if len(names) == 2:
        a, b = names[0], names[1]
        sa, sb = strategy_summaries[a], strategy_summaries[b]
        print("\n Delta ({} − {})".format(b, a))
        print("-" * 40)
        for key, label in (
            ("mean_iou", "IoU"),
            ("mean_dice", "Dice"),
            ("mean_precision", "Precision"),
            ("mean_recall", "Recall"),
            ("mean_boundary_f1", "Boundary F1"),
        ):
            va, vb = sa.get(key), sb.get(key)
            if va is not None and vb is not None:
                print(f"  {label:12s}: {vb - va:+.4f}")

    for strategy, rows in strategy_rows.items():
        by_prompt = _summarize_grouped(rows, "prompt", summarize_fn)
        if len(by_prompt) <= 1:
            continue
        print(f"\n Per prompt — {strategy}")
        print(f" {'Prompt':<16} {'N':>5} {'IoU':>8} {'Dice':>8} {'B-F1':>8}")
        print(" " + "-" * 50)
        for prompt, ps in by_prompt.items():
            print(
                f" {prompt[:16]:<16} {ps.get('samples_ok', 0):>5} "
                f"{_fmt_metric(ps.get('mean_iou'))} "
                f"{_fmt_metric(ps.get('mean_dice'))} "
                f"{_fmt_metric(ps.get('mean_boundary_f1'))}"
            )

        by_scenario = _summarize_grouped(rows, "scenario", summarize_fn)
        if len(by_scenario) > 1:
            print(f"\n Per scenario — {strategy}")
            print(f" {'Scenario':<20} {'N':>5} {'IoU':>8} {'Dice':>8} {'Fail%':>7}")
            print(" " + "-" * 52)
            for scenario, ss in by_scenario.items():
                print(
                    f" {scenario[:20]:<20} {ss.get('samples_ok', 0):>5} "
                    f"{_fmt_metric(ss.get('mean_iou'))} "
                    f"{_fmt_metric(ss.get('mean_dice'))} "
                    f"{_fmt_metric(ss.get('failure_rate'), pct=True)}"
                )

    print("\n Saved files")
    print("-" * 40)
    for strategy in strategy_summaries:
        print(f"  {out_dir / f'summary_{strategy}.json'}")
        print(f"  {out_dir / f'per_image_{strategy}.csv'}")
    print(f"  {out_dir / 'per_image_all.csv'}")
    print("=" * width + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark GND+SAM3 vs SAM3-native on a manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="JSONL/JSON manifest (see benchmarks/README.md).",
    )
    parser.add_argument(
        "--strategy",
        choices=("sam3_native", "gnd_sam", "gnd_sam_pipeline", "both"),
        default="both",
        help="both = run sam3_native and gnd_sam_pipeline, compare summaries.",
    )
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--max-boxes", type=int, default=10)
    parser.add_argument("--boundary-tolerance-px", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=Path("benchmark_runs/latest"))
    parser.add_argument("--sam-backend", type=str, default="", help="Override SAM_BACKEND env.")
    parser.add_argument("--limit", type=int, default=0, help="Max samples (0 = all).")
    parser.add_argument("--dry-run", action="store_true", help="Only list manifest samples.")
    args = parser.parse_args()

    if args.sam_backend:
        os.environ["SAM_BACKEND"] = args.sam_backend.strip()

    samples = _load_manifest(args.manifest.resolve())
    if args.limit > 0:
        samples = samples[: args.limit]

    if args.dry_run:
        print(f"manifest={args.manifest} samples={len(samples)}")
        for s in samples[:5]:
            print(f"  - {s.get('sample_id')} prompt={s.get('prompt')} image={s.get('image')}")
        return

    # Import after env is set (models load on first import).
    import ml_service.models  # noqa: F401
    from ml_service.sam3_metrics import (
        MaskPair,
        evaluate_pairs,
        load_mask,
        pick_best_pred_mask,
        summarize_rows,
        write_per_image_csv,
    )

    strategies = (
        ["sam3_native", "gnd_sam_pipeline"]
        if args.strategy == "both"
        else [args.strategy]
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = args.manifest.resolve().parent

    all_rows: list[dict[str, Any]] = []
    strategy_summaries: dict[str, dict[str, Any]] = {}
    strategy_rows: dict[str, list[dict[str, Any]]] = {}
    for strategy in strategies:
        pred_root = args.out_dir / "pred_masks" / strategy
        pred_root.mkdir(parents=True, exist_ok=True)
        pairs: list[MaskPair] = []

        print(f"\n=== strategy={strategy} samples={len(samples)} ===")
        for sample in samples:
            sid = str(sample["sample_id"])
            prompt = str(sample.get("prompt", sample.get("category", "object")))
            scenario = str(sample.get("scenario", "default"))
            image_path = Path(sample["image"])
            if not image_path.is_absolute():
                image_path = (manifest_dir / image_path).resolve()
            gt_path = Path(sample["gt_mask"])
            if not gt_path.is_absolute():
                gt_path = (manifest_dir / gt_path).resolve()

            if not image_path.exists():
                print(f"SKIP {sid}: missing image {image_path}")
                continue
            if not gt_path.exists():
                print(f"SKIP {sid}: missing gt {gt_path}")
                continue

            try:
                pred_masks, latency_ms = _run_strategy(
                    strategy,
                    image_path,
                    prompt,
                    args.score_threshold,
                    args.max_boxes,
                )
                gt = load_mask(gt_path)
                best_mask, _ = pick_best_pred_mask(pred_masks, gt)
                pred_path = pred_root / f"{sid}.png"
                if best_mask is None:
                    _save_mask_png(np.zeros_like(gt), pred_path)
                else:
                    _save_mask_png(best_mask, pred_path)
            except Exception as exc:
                print(f"FAIL {sid}: {exc}")
                pred_path = pred_root / f"{sid}.png"
                gt = load_mask(gt_path)
                _save_mask_png(np.zeros_like(gt), pred_path)
                latency_ms = None

            pairs.append(
                MaskPair(
                    sample_id=sid,
                    gt_path=gt_path,
                    pred_path=pred_path,
                    scenario=scenario,
                    latency_ms=latency_ms,
                    prompt=prompt,
                    strategy=strategy,
                )
            )

        rows = evaluate_pairs(
            pairs,
            boundary_tolerance_px=args.boundary_tolerance_px,
            ignore_missing_pred=False,
        )
        all_rows.extend(rows)
        strategy_rows[strategy] = rows
        per_strategy_csv = args.out_dir / f"per_image_{strategy}.csv"
        write_per_image_csv(rows, per_strategy_csv)
        summary = summarize_rows(rows)
        strategy_summaries[strategy] = summary
        summary_path = args.out_dir / f"summary_{strategy}.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    write_per_image_csv(all_rows, args.out_dir / "per_image_all.csv")

    _print_benchmark_report(
        manifest=args.manifest.resolve(),
        out_dir=args.out_dir,
        score_threshold=args.score_threshold,
        strategy_summaries=strategy_summaries,
        strategy_rows=strategy_rows,
        summarize_fn=summarize_rows,
    )


if __name__ == "__main__":
    main()
