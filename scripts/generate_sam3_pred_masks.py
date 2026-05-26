#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import requests
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate SAM3 predicted masks via ml_service /preannotate")
    p.add_argument("--images-dir", type=Path, required=True, help="Directory with source images")
    p.add_argument("--gt-dir", type=Path, required=True, help="Directory with GT masks (names define evaluation set)")
    p.add_argument("--out-dir", type=Path, required=True, help="Directory to save predicted masks (*.png)")
    p.add_argument("--ml-url", type=str, default="http://127.0.0.1:8000", help="ML service base URL")
    p.add_argument("--score-threshold", type=float, default=0.25)
    p.add_argument("--max-boxes", type=int, default=10)
    p.add_argument("--latency-csv", type=Path, default=None, help="Optional output CSV with per-image latency")
    return p.parse_args()


def find_image_by_stem(images_dir: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def build_union_mask_from_zip(zip_bytes: bytes) -> np.ndarray | None:
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        ann_path = "annotations/annotations_coco.json"
        if ann_path not in zf.namelist():
            return None
        coco = json.loads(zf.read(ann_path).decode("utf-8"))
        anns = coco.get("annotations", [])
        if not anns:
            return None

        union: np.ndarray | None = None
        for ann in anns:
            mask_rel = ann.get("mask_path")
            if not isinstance(mask_rel, str) or not mask_rel.strip():
                continue
            zname = mask_rel.strip().lstrip("/")
            if zname not in zf.namelist():
                continue
            m = np.array(Image.open(io.BytesIO(zf.read(zname))).convert("L"), dtype=np.uint8) > 0
            if union is None:
                union = m
            else:
                if union.shape != m.shape:
                    continue
                union = np.logical_or(union, m)
        if union is None:
            return None
        return union.astype(np.uint8)


def main() -> None:
    args = parse_args()
    images_dir = args.images_dir.resolve()
    gt_dir = args.gt_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_masks = sorted([p for p in gt_dir.glob("*.png") if p.is_file()])
    if not gt_masks:
        raise SystemExit(f"No GT masks found in {gt_dir}")

    rows: list[dict[str, str]] = []
    endpoint = args.ml_url.rstrip("/") + "/preannotate"

    for i, gt in enumerate(gt_masks, start=1):
        stem = gt.stem
        img = find_image_by_stem(images_dir, stem)
        out_path = out_dir / f"{stem}.png"
        print(f"[{i}/{len(gt_masks)}] {stem}")

        if img is None:
            print(f"  - skip: image not found for stem={stem}")
            rows.append({"sample_id": stem, "status": "missing_image", "latency_ms": ""})
            continue

        payload = {
            "score_threshold": args.score_threshold,
            "max_boxes": args.max_boxes,
            "format": "coco",
            "task_type": "segmentation",
            "class_names": ["object"],
            "use_clip": False,
            "use_qwen": False,
        }
        t0 = time.perf_counter()
        with img.open("rb") as fh:
            files = [("images", (img.name, fh, "image/jpeg"))]
            resp = requests.post(endpoint, data={"payload": json.dumps(payload)}, files=files, timeout=1800)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        if resp.status_code != 200:
            print(f"  - fail: status={resp.status_code}")
            rows.append({"sample_id": stem, "status": f"http_{resp.status_code}", "latency_ms": f"{dt_ms:.2f}"})
            continue

        pred = build_union_mask_from_zip(resp.content)
        if pred is None:
            gt_arr = np.array(Image.open(gt).convert("L"))
            pred = np.zeros_like(gt_arr, dtype=np.uint8)
            print("  - warning: no masks in zip, saved empty mask")

        # normalize to binary png {0,255}
        Image.fromarray((pred > 0).astype(np.uint8) * 255, mode="L").save(out_path)
        rows.append({"sample_id": stem, "status": "ok", "latency_ms": f"{dt_ms:.2f}"})
        print(f"  - saved: {out_path.name} ({dt_ms:.1f} ms)")

    if args.latency_csv:
        args.latency_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.latency_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["sample_id", "status", "latency_ms"])
            w.writeheader()
            w.writerows(rows)
        print(f"Latency/status CSV saved: {args.latency_csv}")


if __name__ == "__main__":
    main()
