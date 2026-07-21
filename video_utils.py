"""Helpers for native CVAT video tasks (Approach A: upload .mp4, preannotate by frame)."""
from __future__ import annotations

import json
import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence

logger = logging.getLogger("video_utils")

VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"})
# CVAT video chunk placeholder: one meta entry named like the source .mp4 while size > 1.
CVAT_FRAME_NAME_PATTERNS = (
    "{i:06d}.jpg",
    "frame_{i:06d}.jpg",
    "frame_{i:06d}",
    "{i}.jpg",
)


def is_video_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def is_raw_video_placeholder_name(name: str) -> bool:
    return is_video_file(name)


def expected_frame_count(params: VideoUploadParams) -> int:
    if params.max_frames is not None and int(params.max_frames) > 0:
        return int(params.max_frames)
    if params.stop_frame is not None:
        return (int(params.stop_frame) - int(params.start_frame)) // max(1, int(params.frame_step)) + 1
    return 2


def resolve_cvat_frame_list(meta: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Build per-frame metadata for CVAT video tasks.

    CVAT often returns a single frames[] entry named like the source .mp4 while
    meta['size'] is the real frame count (one entry per chunk, not per frame).
    """
    size = int(meta.get("size") or 0)
    raw_frames = meta.get("frames") or []
    if size <= 0:
        raise RuntimeError("CVAT data/meta has size=0")

    if len(raw_frames) == size:
        return [dict(f) for f in raw_frames]

    if len(raw_frames) == 1 and is_raw_video_placeholder_name(str(raw_frames[0].get("name", ""))):
        w = int(raw_frames[0].get("width") or 0)
        h = int(raw_frames[0].get("height") or 0)
        pattern = CVAT_FRAME_NAME_PATTERNS[0]
        logger.info(
            "CVAT video meta: chunk placeholder %r, expanding to %d frames as %s",
            raw_frames[0].get("name"),
            size,
            pattern,
        )
        return [{"name": pattern.format(i=i), "width": w, "height": h} for i in range(size)]

    if len(raw_frames) > 1:
        logger.warning(
            "CVAT frames list length %d != size %d; using listed frames only",
            len(raw_frames),
            size,
        )
        return [dict(f) for f in raw_frames]

    raise RuntimeError(f"Cannot resolve CVAT frame list (size={size}, frames={len(raw_frames)})")


def parse_frame_step(frame_filter: str | None) -> int:
    """Parse CVAT frame_filter, e.g. 'step=5' -> 5."""
    if not frame_filter:
        return 1
    for part in str(frame_filter).split(","):
        part = part.strip()
        m = re.match(r"^step\s*=\s*(\d+)$", part, flags=re.IGNORECASE)
        if m:
            return max(1, int(m.group(1)))
    return 1


def get_video_frame_count(video_path: str | Path) -> int:
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError("opencv-python is required for video processing") from e

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return max(0, count)
    finally:
        cap.release()


@dataclass(frozen=True)
class VideoUploadParams:
    frame_step: int = 5
    start_frame: int = 0
    max_frames: int | None = 300
    stop_frame: int | None = None

    def normalized(self, video_frame_count: int | None = None) -> VideoUploadParams:
        step = max(1, int(self.frame_step))
        start = max(0, int(self.start_frame))
        stop = self.stop_frame

        if self.max_frames is not None and self.max_frames > 0:
            computed = start + (int(self.max_frames) - 1) * step
            stop = computed if stop is None else min(int(stop), computed)

        if video_frame_count is not None and video_frame_count > 0:
            last_idx = video_frame_count - 1
            if stop is None:
                stop = last_idx
            else:
                stop = min(int(stop), last_idx)

        if stop is not None and stop < start:
            stop = start
        return VideoUploadParams(
            frame_step=step,
            start_frame=start,
            max_frames=self.max_frames,
            stop_frame=stop,
        )

    def to_cvat_upload_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "start_frame": self.start_frame,
            "frame_filter": f"step={self.frame_step}",
        }
        if self.stop_frame is not None:
            params["stop_frame"] = int(self.stop_frame)
        return params


def source_frame_index(task_frame_index: int, *, start_frame: int, frame_step: int) -> int:
    return int(start_frame) + int(task_frame_index) * int(frame_step)


@dataclass(frozen=True)
class ExtractedFrame:
    local_path: str
    cvat_name: str
    task_frame_index: int


def patch_coco_archive_for_cvat_video(
    archive_path: str | Path,
    extracted_frames: Sequence[ExtractedFrame],
) -> dict[str, Any]:
    """
    Rewrite COCO inside the ML zip so file_name matches CVAT video frame names.

    CVAT silently skips annotations when file_name does not match data/meta exactly.
    Also adds annotations/instances_default.json (Datumaro/CVAT convention).
    """
    archive_path = Path(archive_path)
    by_local_name = {Path(f.local_path).name: f for f in extracted_frames}

    with zipfile.ZipFile(archive_path, "r") as zf:
        ann_path = "annotations/annotations_coco.json"
        if ann_path not in zf.namelist():
            raise RuntimeError(f"{ann_path} missing in {archive_path}")
        coco = json.loads(zf.read(ann_path).decode("utf-8"))
        other_entries = {
            name: zf.read(name)
            for name in zf.namelist()
            if name != ann_path and name != "annotations/instances_default.json"
        }

    images = coco.get("images") or []
    id_remap: dict[int, int] = {}
    matched = 0
    unmatched: list[str] = []

    for img in images:
        if not isinstance(img, dict):
            continue
        local_base = Path(str(img.get("file_name", ""))).name
        entry = by_local_name.get(local_base)
        if entry is None:
            unmatched.append(local_base)
            continue
        old_id = int(img.get("id", 0))
        new_id = int(entry.task_frame_index) + 1
        img["file_name"] = entry.cvat_name
        img["id"] = new_id
        id_remap[old_id] = new_id
        matched += 1

    annotations = coco.get("annotations") or []
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        old_image_id = int(ann.get("image_id", 0))
        if old_image_id in id_remap:
            ann["image_id"] = id_remap[old_image_id]

    coco_bytes = json.dumps(coco, ensure_ascii=False).encode("utf-8")

    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(ann_path, coco_bytes)
        zf.writestr("annotations/instances_default.json", coco_bytes)
        for name, data in other_entries.items():
            zf.writestr(name, data)

    sample_names = [f.cvat_name for f in extracted_frames[:3]]
    logger.info(
        "Patched COCO for CVAT video import: matched=%d/%d unmatched=%s sample_cvat_names=%s",
        matched,
        len(images),
        unmatched[:5],
        sample_names,
    )
    return {
        "matched_images": matched,
        "total_images": len(images),
        "unmatched_file_names": unmatched[:20],
        "sample_cvat_names": sample_names,
    }


def _write_frame_image(frame_bgr: Any, out_path: Path) -> None:
    """Save BGR numpy frame; use Pillow (OpenCV imwrite often lacks JPEG on headless builds)."""
    import cv2
    from PIL import Image

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = out_path.suffix.lower()
    if suffix in (".jpg", ".jpeg", ".jpe", ""):
        img.save(out_path, format="JPEG", quality=90)
    elif suffix == ".png":
        img.save(out_path, format="PNG")
    elif suffix == ".webp":
        img.save(out_path, format="WEBP", quality=90)
    elif suffix in (".bmp", ".tif", ".tiff"):
        img.save(out_path)
    else:
        # Unknown CVAT frame extension — encode as JPEG bytes under the exact CVAT name.
        img.save(out_path, format="JPEG", quality=90)


def extract_video_frames_for_cvat(
    video_path: str | Path,
    out_dir: str | Path,
    frame_meta: Sequence[dict[str, Any]],
    *,
    start_frame: int,
    frame_step: int,
    progress_cb: Any | None = None,
) -> List[ExtractedFrame]:
    """
    Extract JPEG frames from a local video using CVAT task frame metadata.

    Saved file names must match FrameMeta.name from GET /api/tasks/{id}/data/meta
    so COCO import maps annotations to the correct frames.
    """
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError("opencv-python is required for video processing") from e

    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    saved: List[ExtractedFrame] = []
    try:
        total = len(frame_meta)
        for i, meta in enumerate(frame_meta):
            name = str(meta.get("name") or f"{i}.jpg").strip()
            if not name:
                name = f"{i}.jpg"
            src_idx = source_frame_index(i, start_frame=start_frame, frame_step=frame_step)
            cap.set(cv2.CAP_PROP_POS_FRAMES, src_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                logger.warning("Failed to read source frame %s for task frame %s (%s)", src_idx, i, name)
                continue
            out_path = out_dir / name
            try:
                _write_frame_image(frame, out_path)
            except Exception as e:
                raise RuntimeError(f"Failed to write frame {out_path}: {e}") from e
            saved.append(
                ExtractedFrame(local_path=str(out_path), cvat_name=name, task_frame_index=i)
            )
            if i == 0:
                logger.info("First CVAT frame name for extraction: %r -> %s", name, out_path)
            if progress_cb is not None:
                progress_cb(i + 1, total, name)
    finally:
        cap.release()

    if not saved:
        raise RuntimeError("No frames extracted from video")
    return saved
