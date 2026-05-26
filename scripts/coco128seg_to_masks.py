#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert COCO128-seg txt polygons to binary PNG masks.")
    p.add_argument("--images-dir", type=Path, required=True, help="Path to coco128-seg/images/train2017")
    p.add_argument("--labels-dir", type=Path, required=True, help="Path to coco128-seg/labels/train2017")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory for GT masks (*.png)")
    return p.parse_args()


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted([p for p in args.labels_dir.glob("*.txt") if p.is_file()])
    if not label_files:
        raise SystemExit(f"No label txt files found in: {args.labels_dir}")

    converted = 0
    skipped = 0
    for lbl in label_files:
        stem = lbl.stem
        img_path = find_image(args.images_dir, stem)
        if img_path is None:
            skipped += 1
            continue

        with Image.open(img_path) as im:
            w, h = im.size

        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)

        text = lbl.read_text(encoding="utf-8").strip()
        if text:
            for line in text.splitlines():
                parts = line.strip().split()
                # YOLO-seg line: class x1 y1 x2 y2 ... (normalized)
                if len(parts) < 7:
                    continue
                coords = parts[1:]
                if len(coords) % 2 != 0:
                    continue
                pts = []
                for i in range(0, len(coords), 2):
                    x = max(0.0, min(1.0, float(coords[i]))) * w
                    y = max(0.0, min(1.0, float(coords[i + 1]))) * h
                    pts.append((x, y))
                if len(pts) >= 3:
                    draw.polygon(pts, fill=255)

        out = args.out_dir / f"{stem}.png"
        mask.save(out)
        converted += 1

    print(f"Converted: {converted}, skipped(no image): {skipped}, out_dir: {args.out_dir}")


if __name__ == "__main__":
    main()
