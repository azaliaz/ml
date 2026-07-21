#!/usr/bin/env python3
"""Print aggregate statistics from a COCO zip/json archive (ML preannotation or CVAT export)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.coco_drift_utils import compute_batch_stats, load_coco_from_path, load_coco_from_zip  # noqa: E402


def _print_stats(stats_dict: dict, coco_member: str | None = None) -> None:
    if coco_member:
        print(f"coco_member: {coco_member}")
    print(f"images_count:              {stats_dict['images_count']}")
    print(f"annotations_count:         {stats_dict['annotations_count']}")
    print(f"categories_count:        {stats_dict['categories_count']}")
    print(f"images_without_annotations: {stats_dict['images_without_annotations']}")
    if stats_dict["mean_score"] is not None:
        print(f"mean_score:                {stats_dict['mean_score']:.4f}")
        print(f"min_score:                 {stats_dict['min_score']:.4f}")
        print(f"max_score:                 {stats_dict['max_score']:.4f}")
    else:
        print("mean_score:                (no score field in annotations)")
    if stats_dict["detections_per_image_mean"] is not None:
        print(f"detections_per_image_mean: {stats_dict['detections_per_image_mean']:.2f}")
        print(f"detections_per_image_min:  {stats_dict['detections_per_image_min']}")
        print(f"detections_per_image_max:  {stats_dict['detections_per_image_max']}")
    hist = stats_dict.get("class_histogram") or {}
    if hist:
        print("class_histogram:")
        for name, count in sorted(hist.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {name}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute batch statistics from a COCO archive (.zip) or JSON file."
    )
    parser.add_argument("archive", type=Path, help="Path to .zip or .json COCO file")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only")
    args = parser.parse_args()

    archive = args.archive.resolve()
    coco_member: str | None = None
    if archive.suffix.lower() == ".zip":
        coco, coco_member = load_coco_from_zip(archive)
    else:
        coco = load_coco_from_path(archive)

    stats = compute_batch_stats(coco).to_dict()

    if args.json:
        payload = {"archive": str(archive), "coco_member": coco_member, "stats": stats}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(f"archive: {archive}")
    _print_stats(stats, coco_member=coco_member)


if __name__ == "__main__":
    main()
