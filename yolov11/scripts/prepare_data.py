#!/usr/bin/env python3
"""
prepare_data.py

Usage (run from SKU110K_fixed directory):
    python prepare_data.py \
        --src . \
        --dst ../dataset_yolo \
        --train_csv annotations/annotations_train.csv \
        --val_csv annotations/annotations_val.csv \
        --test_csv annotations/annotations_test.csv \
        --out_configs ../configs

The script:
 - supports CSVs with or without header (auto-detect)
 - supports formats:
     image,x1,y1,x2,y2,class
     image,x1,y1,x2,y2,class,img_w,img_h
 - converts bbox -> YOLO format (x_center y_center width height, normalized)
 - copies images to dst/{train,val,test}/images and writes labels to dst/{...}/labels
 - writes configs/names.txt and configs/data.yaml
"""

import argparse
from pathlib import Path
import shutil
import pandas as pd
from PIL import Image
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")


def guess_has_header(csv_path: Path):
    # peek first non-empty line, check first token: if it looks like a filename (endswith image ext), then NO header
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            first = line.split(",")[0].strip().lower()
            if any(first.endswith(ext) for ext in IMG_EXTS) or first.startswith("train_") or first.startswith("test_") or first.startswith("val_"):
                return False
            # else likely header
            return True
    return True


def read_annotations(csv_path: Path):
    has_header = guess_has_header(csv_path)
    if has_header:
        df = pd.read_csv(csv_path)
    else:
        # read without header and assign possible columns
        df = pd.read_csv(csv_path, header=None)
        # try to map based on number of columns
        if df.shape[1] >= 8:
            df.columns = ["image", "x1", "y1", "x2", "y2", "class", "img_w", "img_h"] + [
                f"c{i}" for i in range(8, df.shape[1])
            ]
        elif df.shape[1] == 6:
            df.columns = ["image", "x1", "y1", "x2", "y2", "class"]
        else:
            raise RuntimeError(f"Unexpected number of columns ({df.shape[1]}) in {csv_path}. Please check CSV format.")
    return df


def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
    # Ensure float
    x1, y1, x2, y2, img_w, img_h = map(float, (x1, y1, x2, y2, img_w, img_h))
    # fix reversed coords
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    xc = (x1 + x2) / 2.0
    yc = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    # normalize
    if img_w == 0 or img_h == 0:
        return 0.0, 0.0, 0.0, 0.0
    return xc / img_w, yc / img_h, w / img_w, h / img_h


def process_split(csv_path: Path, images_dir: Path, dst_images_dir: Path, dst_labels_dir: Path, class_map: dict):
    logging.info(f"Processing split CSV: {csv_path}")
    df = read_annotations(csv_path)
    if df.empty:
        logging.warning(f"No rows in {csv_path}")
        return class_map, 0

    # ensure 'image' column exists
    if "image" not in df.columns:
        # try common alternatives
        candidates = [c for c in df.columns if "name" in c.lower() or "file" in c.lower() or "img" in c.lower()]
        if candidates:
            df = df.rename(columns={candidates[0]: "image"})
        else:
            df = df.rename(columns={df.columns[0]: "image"})
            logging.warning(f"Renamed first column to 'image'")

    # ensure bbox columns exist: prefer x1,y1,x2,y2
    bbox_cols = None
    if all(c in df.columns for c in ["x1", "y1", "x2", "y2"]):
        bbox_cols = ("x1", "y1", "x2", "y2")
    elif all(c in df.columns for c in ["xmin", "ymin", "xmax", "ymax"]):
        bbox_cols = ("xmin", "ymin", "xmax", "ymax")
    elif all(c in df.columns for c in ["x", "y", "w", "h"]):
        # assume x,y are top-left
        bbox_cols = ("x", "y", "w", "h")
    else:
        raise RuntimeError(f"Could not detect bbox columns in {csv_path}. Columns: {list(df.columns)}")

    # class column
    if "class" in df.columns:
        class_col = "class"
    else:
        # try to find likely class column
        candidates = [c for c in df.columns if "class" in c.lower() or "label" in c.lower() or "category" in c.lower()]
        class_col = candidates[0] if candidates else df.columns[5] if df.shape[1] > 5 else None
        if class_col is None:
            raise RuntimeError("Cannot detect class column.")
        logging.info(f"Using '{class_col}' as class column")

    # optional image size columns
    use_csv_sizes = all(c in df.columns for c in ["img_w", "img_h"])

    grouped = df.groupby("image")
    count_images = 0

    for img_name, group in grouped:
        src_img = images_dir / img_name
        if not src_img.exists():
            logging.warning(f"Image not found: {src_img} — skipping")
            continue

        dst_img = dst_images_dir / img_name
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_img, dst_img)

        # get image size either from CSV or by opening
        if use_csv_sizes:
            sample = group.iloc[0]
            img_w = float(sample["img_w"])
            img_h = float(sample["img_h"])
        else:
            try:
                with Image.open(src_img) as im:
                    img_w, img_h = im.size
            except Exception as e:
                logging.error(f"Cannot open image {src_img}: {e}")
                continue

        label_file = dst_labels_dir / (Path(img_name).stem + ".txt")
        label_file.parent.mkdir(parents=True, exist_ok=True)

        lines = []
        for _, row in group.iterrows():
            # read class label and map to int
            raw_cls = row[class_col]
            if pd.isna(raw_cls):
                continue
            cls_key = str(raw_cls).strip()
            if cls_key not in class_map:
                class_map[cls_key] = len(class_map)
            cls_id = class_map[cls_key]

            # bbox parsing
            if bbox_cols == ("x", "y", "w", "h"):
                x = float(row["x"])
                y = float(row["y"])
                w_box = float(row["w"])
                h_box = float(row["h"])
                x1 = x
                y1 = y
                x2 = x + w_box
                y2 = y + h_box
            else:
                x1 = float(row[bbox_cols[0]])
                y1 = float(row[bbox_cols[1]])
                x2 = float(row[bbox_cols[2]])
                y2 = float(row[bbox_cols[3]])

            # if coordinates appear normalized (<=1) convert to absolute
            max_coord = max(x1, y1, x2, y2)
            if max_coord <= 1.0:
                x1 *= img_w
                x2 *= img_w
                y1 *= img_h
                y2 *= img_h

            xc, yc, w_n, h_n = xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h)
            # clamp
            xc = min(max(xc, 0.0), 1.0)
            yc = min(max(yc, 0.0), 1.0)
            w_n = min(max(w_n, 0.0), 1.0)
            h_n = min(max(h_n, 0.0), 1.0)
            lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w_n:.6f} {h_n:.6f}")

        # write label file (even if empty)
        with open(label_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        count_images += 1

    logging.info(f"Finished {csv_path}: processed {count_images} images")
    return class_map, count_images


def main(args):
    src_root = Path(args.src).resolve()
    images_dir = src_root / "images"
    if not images_dir.exists():
        logging.error(f"Images dir not found: {images_dir}")
        sys.exit(1)

    dst_root = Path(args.dst).resolve()
    out_configs = Path(args.out_configs).resolve()
    out_configs.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": Path(args.train_csv) if args.train_csv else None,
        "val": Path(args.val_csv) if args.val_csv else None,
        "test": Path(args.test_csv) if args.test_csv else None,
    }

    # normalize csv paths
    for k, p in splits.items():
        if p:
            if not p.is_absolute():
                splits[k] = src_root / p

    # make dst dirs
    for s in splits:
        if splits[s] is None:
            continue
        (dst_root / s / "images").mkdir(parents=True, exist_ok=True)
        (dst_root / s / "labels").mkdir(parents=True, exist_ok=True)

    class_map = {}
    counts = {}
    for split, csv_path in splits.items():
        if csv_path is None:
            logging.info(f"No CSV for split {split}, skipping")
            continue
        if not csv_path.exists():
            logging.error(f"CSV not found: {csv_path}")
            sys.exit(1)
        class_map, n = process_split(csv_path, images_dir, dst_root / split / "images", dst_root / split / "labels", class_map)
        counts[split] = n

    # write names.txt (ordered by class id)
    names_by_id = [None] * len(class_map)
    for k, v in class_map.items():
        names_by_id[v] = k
    names_file = out_configs / "names.txt"
    with open(names_file, "w", encoding="utf-8") as f:
        f.write("\n".join(names_by_id))

    # write data.yaml
    data_yaml = out_configs / "data.yaml"
    rel_dst = dst_root
    yaml_lines = [
        f"path: {rel_dst}",
        f"train: {rel_dst}/train/images",
        f"val: {rel_dst}/val/images",
        f"test: {rel_dst}/test/images",
        f"nc: {len(names_by_id)}",
        "names: [" + ", ".join([f"'{n}'" for n in names_by_id]) + "]",
    ]
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_lines))

    logging.info("=== Summary ===")
    for k, v in counts.items():
        logging.info(f"{k}: {v} images")
    logging.info(f"Classes: {len(names_by_id)} saved to {names_file}")
    logging.info(f"data.yaml written to {data_yaml}")
    logging.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default=".", help="SKU110K_fixed root")
    parser.add_argument("--dst", type=str, default="../dataset_yolo", help="output dataset root")
    parser.add_argument("--train_csv", type=str, default="annotations/annotations_train.csv", help="train csv (relative to --src)")
    parser.add_argument("--val_csv", type=str, default="annotations/annotations_val.csv", help="val csv")
    parser.add_argument("--test_csv", type=str, default="annotations/annotations_test.csv", help="test csv")
    parser.add_argument("--out_configs", type=str, default="../configs", help="where to write names.txt and data.yaml")
    args = parser.parse_args()
    main(args)
