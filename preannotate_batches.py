"""Batch calls to ml_service /preannotate and merge COCO ZIP archives."""
from __future__ import annotations

import io
import json
import logging
import zipfile
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional

import requests

logger = logging.getLogger("preannotate_batches")

COCO_ANN_PATH = "annotations/annotations_coco.json"


def merge_coco_zip_archives(archive_bytes_list: Iterable[bytes]) -> bytes:
    """Merge several ML preannotation ZIPs into one COCO archive for CVAT import."""
    merged_images: list[dict[str, Any]] = []
    merged_anns: list[dict[str, Any]] = []
    categories: list[dict[str, Any]] | None = None
    next_img_id = 1
    next_ann_id = 1
    preview_entry: tuple[str, bytes] | None = None

    for batch_idx, raw in enumerate(archive_bytes_list):
        with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
            if COCO_ANN_PATH not in zf.namelist():
                raise RuntimeError(f"Batch {batch_idx}: missing {COCO_ANN_PATH}")
            coco = json.loads(zf.read(COCO_ANN_PATH).decode("utf-8"))
            batch_categories = coco.get("categories") or []
            if categories is None:
                categories = batch_categories
            id_map: dict[int, int] = {}
            for img in coco.get("images") or []:
                if not isinstance(img, dict):
                    continue
                old_id = int(img.get("id", 0))
                new_img = dict(img)
                new_img["id"] = next_img_id
                id_map[old_id] = next_img_id
                merged_images.append(new_img)
                next_img_id += 1
            for ann in coco.get("annotations") or []:
                if not isinstance(ann, dict):
                    continue
                old_image_id = int(ann.get("image_id", 0))
                if old_image_id not in id_map:
                    continue
                new_ann = dict(ann)
                new_ann["id"] = next_ann_id
                new_ann["image_id"] = id_map[old_image_id]
                new_ann.pop("mask_path", None)
                merged_anns.append(new_ann)
                next_ann_id += 1
            if preview_entry is None:
                preview_names = sorted(n for n in zf.namelist() if n.startswith("previews/") and n.lower().endswith(".png"))
                if preview_names:
                    preview_entry = (preview_names[0], zf.read(preview_names[0]))

    if not categories:
        categories = []
    merged = {"images": merged_images, "annotations": merged_anns, "categories": categories}
    coco_bytes = json.dumps(merged, ensure_ascii=False).encode("utf-8")

    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(COCO_ANN_PATH, coco_bytes)
        zf.writestr("annotations/instances_default.json", coco_bytes)
        if preview_entry is not None:
            zf.writestr(preview_entry[0], preview_entry[1])
    out.seek(0)
    return out.read()


def call_preannotate_batches(
    ml_service_url: str,
    image_paths: List[str],
    ml_payload: dict[str, Any],
    *,
    batch_size: int = 20,
    timeout_s: int = 18000,
    status_cb: Optional[Callable[[dict[str, Any]], None]] = None,
) -> bytes:
    """Send images to ml_service in batches; return merged ZIP bytes."""
    if not image_paths:
        raise ValueError("image_paths is empty")
    batch_size = max(1, int(batch_size))
    batches = [image_paths[i : i + batch_size] for i in range(0, len(image_paths), batch_size)]
    total_batches = len(batches)
    logger.info("ML batched preannotate: %d images in %d batches (batch_size=%d)", len(image_paths), total_batches, batch_size)

    archive_parts: list[bytes] = []
    for batch_idx, batch_paths in enumerate(batches):
        if status_cb is not None:
            status_cb(
                {
                    "phase": "calling_ml",
                    "ml_batch_index": batch_idx + 1,
                    "ml_batch_total": total_batches,
                    "ml_batch_size": len(batch_paths),
                }
            )
        payload = dict(ml_payload)
        if batch_idx > 0:
            payload["use_qwen"] = False
            payload["use_qwen_flag"] = False

        opened: list[Any] = []
        try:
            files = []
            for p in batch_paths:
                fobj = open(p, "rb")
                opened.append(fobj)
                files.append(("images", (Path(p).name, fobj, "image/jpeg")))
            resp = requests.post(
                f"{ml_service_url.rstrip('/')}/preannotate",
                data={"payload": json.dumps(payload)},
                files=files,
                timeout=timeout_s,
            )
        finally:
            for f in opened:
                try:
                    f.close()
                except Exception:
                    pass

        if resp.status_code != 200:
            raise RuntimeError(
                f"ML batch {batch_idx + 1}/{total_batches} failed: "
                f"status={resp.status_code} body={resp.text[:2000]}"
            )
        archive_parts.append(resp.content)
        logger.info(
            "ML batch %d/%d ok (%d images, %d bytes)",
            batch_idx + 1,
            total_batches,
            len(batch_paths),
            len(resp.content),
        )

    return merge_coco_zip_archives(archive_parts)
