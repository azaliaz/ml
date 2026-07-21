#!/usr/bin/env python3
"""Compare ML preannotation COCO archive with final CVAT export (edit_rate, IoU, etc.)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.coco_drift_utils import (  # noqa: E402
    compare_preann_final,
    load_coco_from_path,
    load_coco_from_zip,
)


def _print_compare(result: dict) -> None:
    print(f"pre_total:           {result['pre_total']}")
    print(f"final_total:         {result['final_total']}")
    print(f"matched:             {result['matched']}")
    print(f"deleted:             {result['deleted']}")
    print(f"reclassified:        {result['reclassified']}")
    print(f"added:               {result['added']}")
    print(f"images_compared:     {result['images_compared']}")
    print(f"images_only_in_pre:  {result['images_only_in_pre']}")
    print(f"images_only_in_final:{result['images_only_in_final']}")
    if result["edit_rate"] is not None:
        print(f"edit_rate:           {result['edit_rate']:.4f}")
    if result["deletion_rate"] is not None:
        print(f"deletion_rate:       {result['deletion_rate']:.4f}")
    if result["addition_rate"] is not None:
        print(f"addition_rate:       {result['addition_rate']:.4f}")
    if result["mean_iou_matched"] is not None:
        print(f"mean_iou_matched:    {result['mean_iou_matched']:.4f}")


def _load(path: Path) -> tuple[dict, str | None]:
    if path.suffix.lower() == ".zip":
        return load_coco_from_zip(path)
    return load_coco_from_path(path), None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare preannotation COCO vs final COCO export from CVAT."
    )
    parser.add_argument("preann", type=Path, help="Preannotation .zip or .json")
    parser.add_argument("final", type=Path, help="Final export .zip or .json")
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold to treat boxes as the same object (default: 0.5)",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only")
    args = parser.parse_args()

    pre_path = args.preann.resolve()
    final_path = args.final.resolve()
    pre_coco, pre_member = _load(pre_path)
    final_coco, final_member = _load(final_path)

    stats = compare_preann_final(
        pre_coco,
        final_coco,
        iou_threshold=float(args.iou_threshold),
    ).to_dict()

    if args.json:
        payload = {
            "preann": str(pre_path),
            "preann_coco_member": pre_member,
            "final": str(final_path),
            "final_coco_member": final_member,
            "iou_threshold": float(args.iou_threshold),
            "metrics": stats,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(f"preann:  {pre_path}")
    if pre_member:
        print(f"  coco_member: {pre_member}")
    print(f"final:   {final_path}")
    if final_member:
        print(f"  coco_member: {final_member}")
    print(f"iou_threshold: {args.iou_threshold}")
    print()
    _print_compare(stats)


if __name__ == "__main__":
    main()
