#!/usr/bin/env python3
"""CLI: evaluate SAM3 predicted masks against GT masks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_service.sam3_metrics import (  # noqa: E402
    evaluate_pairs,
    resolve_pairs_from_csv,
    resolve_pairs_from_dirs,
    summarize_rows,
    write_per_image_csv,
)


def print_group_summary(name: str, rows: list[dict]) -> None:
    summary = summarize_rows(rows)
    print(f"\n[{name}]")
    print(f"  samples_total: {summary['samples_total']}")
    print(f"  samples_ok:    {summary['samples_ok']}")
    print(f"  failures:      {summary['failures']} ({summary['failure_rate'] * 100.0:.1f}%)")
    if summary.get("mean_iou") is not None:
        print(f"  mean_iou:      {summary['mean_iou']:.4f}")
        print(f"  mean_dice:     {summary['mean_dice']:.4f}")
        print(f"  mean_precision:{summary['mean_precision']:.4f}")
        print(f"  mean_recall:   {summary['mean_recall']:.4f}")
        print(f"  mean_bf1:      {summary['mean_boundary_f1']:.4f}")
    if summary.get("latency_p50_ms") is not None:
        print(f"  latency_p50_ms:{summary['latency_p50_ms']:.2f}")
    if summary.get("latency_p95_ms") is not None:
        print(f"  latency_p95_ms:{summary['latency_p95_ms']:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SAM3 masks against GT masks.")
    parser.add_argument("--gt-dir", type=Path, help="Directory with GT masks.")
    parser.add_argument("--pred-dir", type=Path, help="Directory with predicted masks (same relative paths as gt).")
    parser.add_argument(
        "--pairs-csv",
        type=Path,
        help="CSV: sample_id,gt_path,pred_path,scenario,latency_ms",
    )
    parser.add_argument("--boundary-tolerance-px", type=int, default=2)
    parser.add_argument("--per-image-out", type=Path, default=Path("sam3_eval_per_image.csv"))
    parser.add_argument(
        "--ignore-missing-pred",
        action="store_true",
        help="Skip samples without predicted mask instead of counting them as failed.",
    )
    args = parser.parse_args()

    if args.pairs_csv is None and (args.gt_dir is None or args.pred_dir is None):
        raise SystemExit("Provide either --pairs-csv OR both --gt-dir and --pred-dir")

    if args.pairs_csv is not None:
        pairs = resolve_pairs_from_csv(args.pairs_csv)
    else:
        pairs = resolve_pairs_from_dirs(args.gt_dir.resolve(), args.pred_dir.resolve())

    if not pairs:
        raise SystemExit("No mask pairs found.")

    rows = evaluate_pairs(
        pairs,
        boundary_tolerance_px=args.boundary_tolerance_px,
        ignore_missing_pred=args.ignore_missing_pred,
    )
    write_per_image_csv(rows, args.per_image_out)

    print("SAM3 evaluation summary")
    print(f"per-image report: {args.per_image_out}")
    print_group_summary("overall", rows)

    scenarios = sorted({str(r["scenario"]) for r in rows})
    if len(scenarios) > 1:
        for sc in scenarios:
            print_group_summary(f"scenario={sc}", [r for r in rows if r["scenario"] == sc])


if __name__ == "__main__":
    main()
