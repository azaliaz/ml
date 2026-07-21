"""Load COCO archives and compute batch stats / preann-vs-final comparison."""

from __future__ import annotations

import json
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


COCO_CANDIDATE_PATHS = (
    "annotations/annotations_coco.json",
    "annotations/instances_default.json",
    "instances_default.json",
    "annotations.json",
)


@dataclass(frozen=True)
class CocoBox:
    category_id: int
    category_name: str
    bbox: tuple[float, float, float, float]  # x, y, w, h
    score: float | None = None
    ann_id: int | None = None


@dataclass
class CocoBatchStats:
    images_count: int = 0
    annotations_count: int = 0
    categories_count: int = 0
    mean_score: float | None = None
    min_score: float | None = None
    max_score: float | None = None
    class_histogram: dict[str, int] = field(default_factory=dict)
    detections_per_image_mean: float | None = None
    detections_per_image_min: int | None = None
    detections_per_image_max: int | None = None
    images_without_annotations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "images_count": self.images_count,
            "annotations_count": self.annotations_count,
            "categories_count": self.categories_count,
            "mean_score": self.mean_score,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "class_histogram": dict(self.class_histogram),
            "detections_per_image_mean": self.detections_per_image_mean,
            "detections_per_image_min": self.detections_per_image_min,
            "detections_per_image_max": self.detections_per_image_max,
            "images_without_annotations": self.images_without_annotations,
        }


@dataclass
class CompareStats:
    pre_total: int = 0
    final_total: int = 0
    matched: int = 0
    deleted: int = 0
    reclassified: int = 0
    added: int = 0
    images_compared: int = 0
    images_only_in_pre: int = 0
    images_only_in_final: int = 0
    mean_iou_matched: float | None = None
    edit_rate: float | None = None
    deletion_rate: float | None = None
    addition_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pre_total": self.pre_total,
            "final_total": self.final_total,
            "matched": self.matched,
            "deleted": self.deleted,
            "reclassified": self.reclassified,
            "added": self.added,
            "images_compared": self.images_compared,
            "images_only_in_pre": self.images_only_in_pre,
            "images_only_in_final": self.images_only_in_final,
            "mean_iou_matched": self.mean_iou_matched,
            "edit_rate": self.edit_rate,
            "deletion_rate": self.deletion_rate,
            "addition_rate": self.addition_rate,
        }


def _looks_like_coco(data: Any) -> bool:
    return (
        isinstance(data, dict)
        and isinstance(data.get("images"), list)
        and isinstance(data.get("annotations"), list)
    )


def _find_coco_member(names: list[str]) -> str | None:
    for candidate in COCO_CANDIDATE_PATHS:
        if candidate in names:
            return candidate
    json_names = [n for n in names if n.lower().endswith(".json")]
    for name in json_names:
        lowered = name.lower()
        if "coco" in lowered or "instances" in lowered or lowered.endswith("annotations.json"):
            return name
    return json_names[0] if len(json_names) == 1 else None


def load_coco_from_zip(archive_path: str | Path) -> tuple[dict[str, Any], str]:
    archive_path = Path(archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    if archive_path.stat().st_size == 0:
        raise ValueError(f"Archive is empty: {archive_path}")

    with zipfile.ZipFile(archive_path, "r") as zf:
        member = _find_coco_member(zf.namelist())
        if member is None:
            raise ValueError(
                f"No COCO JSON found in {archive_path}. "
                f"Expected one of: {', '.join(COCO_CANDIDATE_PATHS)}"
            )
        raw = zf.read(member)
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception as e:
            raise ValueError(f"Failed to parse {member} in {archive_path}: {e}") from e
        if not _looks_like_coco(data):
            raise ValueError(f"{member} in {archive_path} is not a valid COCO document")
        return data, member


def load_coco_from_path(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    if path.suffix.lower() == ".zip":
        data, _ = load_coco_from_zip(path)
        return data
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not _looks_like_coco(data):
            raise ValueError(f"{path} is not a valid COCO document")
        return data
    raise ValueError(f"Unsupported path (use .zip or .json): {path}")


def normalize_image_key(file_name: str) -> str:
    return Path(str(file_name or "")).name.lower()


def _category_map(coco: dict[str, Any]) -> dict[int, str]:
    out: dict[int, str] = {}
    for cat in coco.get("categories") or []:
        if not isinstance(cat, dict):
            continue
        cid = cat.get("id")
        name = str(cat.get("name", "") or "").strip()
        if cid is not None:
            out[int(cid)] = name or f"id_{cid}"
    return out


def _parse_bbox(raw: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None
    try:
        x, y, w, h = (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return x, y, w, h


def boxes_by_image_key(coco: dict[str, Any]) -> dict[str, list[CocoBox]]:
    cat_map = _category_map(coco)
    image_id_to_key: dict[int, str] = {}
    for img in coco.get("images") or []:
        if not isinstance(img, dict):
            continue
        img_id = img.get("id")
        if img_id is None:
            continue
        image_id_to_key[int(img_id)] = normalize_image_key(str(img.get("file_name", "")))

    grouped: dict[str, list[CocoBox]] = defaultdict(list)
    for ann in coco.get("annotations") or []:
        if not isinstance(ann, dict):
            continue
        image_id = ann.get("image_id")
        if image_id is None:
            continue
        key = image_id_to_key.get(int(image_id))
        if not key:
            continue
        bbox = _parse_bbox(ann.get("bbox"))
        if bbox is None:
            continue
        cat_id = int(ann.get("category_id", 0))
        score_raw = ann.get("score")
        score = float(score_raw) if score_raw is not None else None
        grouped[key].append(
            CocoBox(
                category_id=cat_id,
                category_name=cat_map.get(cat_id, f"id_{cat_id}"),
                bbox=bbox,
                score=score,
                ann_id=int(ann["id"]) if ann.get("id") is not None else None,
            )
        )
    return dict(grouped)


def compute_batch_stats(coco: dict[str, Any]) -> CocoBatchStats:
    images = [i for i in (coco.get("images") or []) if isinstance(i, dict)]
    annotations = [a for a in (coco.get("annotations") or []) if isinstance(a, dict)]
    categories = [c for c in (coco.get("categories") or []) if isinstance(c, dict)]

    cat_map = _category_map(coco)
    scores: list[float] = []
    class_hist: Counter[str] = Counter()
    per_image_counts: Counter[int] = Counter()

    for ann in annotations:
        cat_id = int(ann.get("category_id", 0))
        class_hist[cat_map.get(cat_id, f"id_{cat_id}")] += 1
        if ann.get("score") is not None:
            scores.append(float(ann["score"]))
        image_id = ann.get("image_id")
        if image_id is not None:
            per_image_counts[int(image_id)] += 1

    image_ids = {int(img["id"]) for img in images if img.get("id") is not None}
    counts = [per_image_counts[iid] for iid in image_ids] if image_ids else list(per_image_counts.values())

    stats = CocoBatchStats(
        images_count=len(images),
        annotations_count=len(annotations),
        categories_count=len(categories),
        class_histogram=dict(class_hist),
        images_without_annotations=max(0, len(image_ids) - len(per_image_counts)),
    )
    if scores:
        stats.mean_score = sum(scores) / len(scores)
        stats.min_score = min(scores)
        stats.max_score = max(scores)
    if counts:
        stats.detections_per_image_mean = sum(counts) / len(counts)
        stats.detections_per_image_min = min(counts)
        stats.detections_per_image_max = max(counts)
    return stats


def bbox_iou_xywh(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh

    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    union = aw * ah + bw * bh - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def compare_preann_final(
    preann_coco: dict[str, Any],
    final_coco: dict[str, Any],
    *,
    iou_threshold: float = 0.5,
) -> CompareStats:
    pre_by_image = boxes_by_image_key(preann_coco)
    final_by_image = boxes_by_image_key(final_coco)

    all_keys = set(pre_by_image) | set(final_by_image)
    stats = CompareStats()
    matched_ious: list[float] = []

    for key in sorted(all_keys):
        pre_boxes = pre_by_image.get(key, [])
        final_boxes = final_by_image.get(key, [])
        if pre_boxes and not final_boxes:
            stats.images_only_in_pre += 1
        if final_boxes and not pre_boxes:
            stats.images_only_in_final += 1
        if pre_boxes and final_boxes:
            stats.images_compared += 1

        used_final: set[int] = set()
        for pre_box in pre_boxes:
            stats.pre_total += 1
            best_iou = 0.0
            best_j: int | None = None
            for j, fin_box in enumerate(final_boxes):
                if j in used_final:
                    continue
                iou = bbox_iou_xywh(pre_box.bbox, fin_box.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_j is None or best_iou < iou_threshold:
                stats.deleted += 1
                continue

            used_final.add(best_j)
            fin_box = final_boxes[best_j]
            if pre_box.category_id == fin_box.category_id:
                stats.matched += 1
                matched_ious.append(best_iou)
            else:
                stats.reclassified += 1

        stats.final_total += len(final_boxes)
        stats.added += len(final_boxes) - len(used_final)

    if stats.pre_total > 0:
        not_accepted = stats.deleted + stats.reclassified
        stats.edit_rate = not_accepted / stats.pre_total
        stats.deletion_rate = stats.deleted / stats.pre_total
    if stats.final_total > 0:
        stats.addition_rate = stats.added / stats.final_total
    if matched_ious:
        stats.mean_iou_matched = sum(matched_ious) / len(matched_ious)
    return stats
