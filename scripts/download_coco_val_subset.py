#!/usr/bin/env python3
"""
Download a small COCO val2017 subset with per-instance GT masks for benchmarking.

Creates:
  <out-dir>/images/*.jpg
  <out-dir>/gt_masks/<sample_id>.png
  <out-dir>/manifest.jsonl

Example (server):
  python scripts/download_coco_val_subset.py \\
    --out-dir benchmarks/coco_val_subset \\
    --max-images 80 \\
    --categories person,bicycle,car,truck,bus
"""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent

COCO_VAL_IMAGE_URL = "http://images.cocodataset.org/val2017/{file_name}"
COCO_VAL_IMAGES_ZIP = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_VAL_ANN_MEMBER = "annotations/instances_val2017.json"

DEFAULT_CATEGORIES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "backpack",
    "handbag",
    "umbrella",
    "bottle",
    "chair",
    "bench",
)


def _download_bytes(url: str) -> bytes:
    print(f"Downloading {url} ...")
    with urlopen(url, timeout=600) as resp:
        return resp.read()


def _pick_annotation_file(cache_dir: Path) -> Path | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    candidates = [
        cache_dir / "instances_val2017.json",
        cache_dir / COCO_VAL_ANN_MEMBER,
    ]
    best: Path | None = None
    best_size = 0
    for path in candidates:
        if path.is_file() and path.stat().st_size > best_size:
            best = path
            best_size = path.stat().st_size
    # Real COCO instances val JSON is tens of MB.
    if best is not None and best_size > 1_000_000:
        return best
    return None


def _ensure_coco_annotations(cache_dir: Path, *, force: bool = False) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    ann_json = cache_dir / "instances_val2017.json"

    if not force:
        existing = _pick_annotation_file(cache_dir)
        if existing is not None:
            if existing != ann_json:
                print(f"Using annotations: {existing}")
            return existing

    # Remove stale/partial files.
    for path in (ann_json, cache_dir / COCO_VAL_ANN_MEMBER):
        if path.exists():
            path.unlink()

    data = _download_bytes(COCO_ANN_URL)
    with zipfile.ZipFile(BytesIO(data)) as zf:
        if COCO_VAL_ANN_MEMBER not in zf.namelist():
            raise RuntimeError(
                f"{COCO_VAL_ANN_MEMBER} not found in annotations zip. "
                f"Members sample: {zf.namelist()[:5]}"
            )
        zf.extract(COCO_VAL_ANN_MEMBER, path=cache_dir)

    extracted = cache_dir / COCO_VAL_ANN_MEMBER
    if not extracted.exists():
        raise RuntimeError(f"Failed to extract {COCO_VAL_ANN_MEMBER}")
    extracted.rename(ann_json)
    print(f"Annotations saved to {ann_json} ({ann_json.stat().st_size // (1024 * 1024)} MB)")
    return ann_json


def _ensure_val_image(cache_dir: Path, file_name: str, *, use_zip_fallback: bool) -> Path:
    """Download a single val2017 image (no need for the full 18GB zip)."""
    img_path = cache_dir / "val2017" / file_name
    if img_path.exists() and img_path.stat().st_size > 0:
        return img_path
    img_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        data = _download_bytes(COCO_VAL_IMAGE_URL.format(file_name=file_name))
        img_path.write_bytes(data)
        return img_path
    except Exception as exc:
        if not use_zip_fallback:
            raise RuntimeError(
                f"Failed to download {file_name} from COCO CDN: {exc}"
            ) from exc

    zip_path = cache_dir / "val2017.zip"
    if not zip_path.exists():
        print("Single-image download failed; downloading full val2017.zip (large, ~1GB)...")
        zip_path.write_bytes(_download_bytes(COCO_VAL_IMAGES_ZIP))
    with zipfile.ZipFile(zip_path) as zf:
        member = f"val2017/{file_name}"
        if member not in zf.namelist():
            raise FileNotFoundError(f"{member} not in val2017.zip")
        zf.extract(member, path=cache_dir)
    return img_path


def _collect_samples(
    coco,
    cat_ids: list[int],
    max_images: int,
    max_instances_per_image: int,
) -> dict[int, list[dict]]:
    """Map image_id -> list of annotations (non-crowd, selected categories)."""
    ann_ids = coco.getAnnIds(catIds=cat_ids)
    if not ann_ids:
        return {}

    img_to_anns: dict[int, list[dict]] = defaultdict(list)
    for ann in coco.loadAnns(ann_ids):
        if int(ann.get("iscrowd", 0)) != 0:
            continue
        if int(ann["category_id"]) not in cat_ids:
            continue
        img_to_anns[int(ann["image_id"])].append(ann)

    # Stable order: images with more matching instances first.
    ranked = sorted(
        img_to_anns.items(),
        key=lambda kv: (-len(kv[1]), kv[0]),
    )
    return dict(ranked[: max(1, max_images)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build COCO val2017 benchmark subset.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "benchmarks" / "coco_val_subset")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "benchmarks" / "_coco_cache")
    parser.add_argument("--max-images", type=int, default=80, help="Max COCO images to export.")
    parser.add_argument("--max-instances-per-image", type=int, default=3)
    parser.add_argument("--categories", type=str, default=",".join(DEFAULT_CATEGORIES))
    parser.add_argument(
        "--force-refresh-annotations",
        action="store_true",
        help="Re-download instances_val2017.json if cache looks invalid.",
    )
    parser.add_argument(
        "--use-zip-fallback",
        action="store_true",
        help="If single-image download fails, download full val2017.zip.",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Only write GT masks + manifest (images downloaded later).",
    )
    args = parser.parse_args()

    try:
        from pycocotools.coco import COCO
    except ImportError as exc:
        raise SystemExit("pycocotools is required: pip install pycocotools") from exc

    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Pillow is required") from exc

    cat_names = [c.strip() for c in args.categories.split(",") if c.strip()]
    cache_dir = args.cache_dir.resolve()
    ann_path = _ensure_coco_annotations(cache_dir, force=args.force_refresh_annotations)
    coco = COCO(str(ann_path))

    n_imgs = len(coco.getImgIds())
    n_anns = len(coco.getAnnIds())
    print(f"COCO loaded: images={n_imgs} annotations={n_anns} from {ann_path}")
    if n_imgs == 0 or n_anns == 0:
        raise SystemExit(
            "COCO annotations look empty. Delete benchmarks/_coco_cache and re-run, "
            "or use --force-refresh-annotations"
        )

    name_to_id = {c["name"]: c["id"] for c in coco.loadCats(coco.getCatIds())}
    cat_ids: list[int] = []
    for name in cat_names:
        if name not in name_to_id:
            print(f"WARN: unknown COCO category '{name}', skipping")
            continue
        cat_ids.append(int(name_to_id[name]))
    if not cat_ids:
        raise SystemExit("No valid categories selected.")

    img_to_anns = _collect_samples(
        coco,
        cat_ids,
        max_images=args.max_images,
        max_instances_per_image=args.max_instances_per_image,
    )
    print(
        f"Selected {len(img_to_anns)} images, "
        f"{sum(len(v) for v in img_to_anns.values())} annotations (before per-cat cap)"
    )
    if not img_to_anns:
        raise SystemExit(
            "No annotations matched selected categories. "
            "Check --categories or refresh annotations cache."
        )

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "gt_masks").mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.jsonl"
    n_written = 0
    n_images = 0
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for img_id, anns in img_to_anns.items():
            img_info = coco.loadImgs([img_id])[0]
            file_name = str(img_info["file_name"])
            rel_image = Path("images") / file_name
            dst_image = out_dir / rel_image

            if not args.skip_images:
                try:
                    src_img = _ensure_val_image(
                        cache_dir,
                        file_name,
                        use_zip_fallback=args.use_zip_fallback,
                    )
                    if not dst_image.exists():
                        dst_image.write_bytes(src_img.read_bytes())
                    n_images += 1
                except Exception as exc:
                    print(f"WARN: skip image {file_name}: {exc}")
                    continue

            by_cat: dict[int, list[dict]] = defaultdict(list)
            for ann in anns:
                by_cat[int(ann["category_id"])].append(ann)

            for cat_id, cat_anns in by_cat.items():
                cat_name = coco.loadCats([cat_id])[0]["name"]
                for idx, ann in enumerate(cat_anns[: args.max_instances_per_image]):
                    mask = coco.annToMask(ann)
                    sid = f"{Path(file_name).stem}_{cat_name}_{idx}"
                    rel_gt = Path("gt_masks") / f"{sid}.png"
                    Image.fromarray((mask.astype("uint8") * 255)).save(out_dir / rel_gt)
                    record = {
                        "sample_id": sid,
                        "image": str(rel_image).replace("\\", "/"),
                        "gt_mask": str(rel_gt).replace("\\", "/"),
                        "prompt": cat_name,
                        "category": cat_name,
                        "scenario": "coco_val2017",
                        "coco_image_id": int(img_id),
                        "coco_ann_id": int(ann["id"]),
                    }
                    mf.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_written += 1

    if n_written == 0:
        raise SystemExit(
            "Wrote 0 samples. Try --force-refresh-annotations or check network/image downloads."
        )

    readme = out_dir / "README.txt"
    readme.write_text(
        "COCO val2017 subset for SAM3/GND benchmark.\n"
        f"Samples: {n_written}\n"
        f"Images copied: {n_images}\n"
        f"Categories: {', '.join(cat_names)}\n"
        "Run:\n"
        "  python scripts/run_gnd_sam3_benchmark.py "
        f"--manifest {manifest_path} --strategy both --out-dir benchmarks/runs/coco_val\n",
        encoding="utf-8",
    )
    print(f"Wrote {n_written} samples ({n_images} images) to {out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
