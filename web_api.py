from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Path as FPath
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).resolve().parent / ".env")

from preannotate_batches import call_preannotate_batches
from cvat_client import (
    create_task_and_upload,
    export_annotations_by_id,
    get_task_data_meta,
    grant_validation_access_for_task,
    import_annotations_to_task,
    wait_for_cvat_video_meta,
)
from video_utils import (
    VIDEO_EXTENSIONS,
    VideoUploadParams,
    expected_frame_count,
    extract_video_frames_for_cvat,
    get_video_frame_count,
    parse_frame_step,
    patch_coco_archive_for_cvat_video,
    resolve_cvat_frame_list,
)

ML_SERVICE_URL = os.environ.get("ML_SERVICE_URL", "http://localhost:8000").rstrip("/")
ML_PREANNOT_BATCH_SIZE = int(os.environ.get("ML_PREANNOT_BATCH_SIZE", "20"))
ML_VIDEO_MAX_FRAMES = int(os.environ.get("ML_VIDEO_MAX_FRAMES", "500"))
CVAT_URL = os.environ.get("CVAT_URL", "http://localhost:8080").rstrip("/")
CVAT_TOKEN = os.environ.get("CVAT_TOKEN", "")
CVAT_INSECURE = os.environ.get("CVAT_INSECURE", "").lower() in ("1", "true", "yes")
CVAT_TOKEN_BEARER = os.environ.get("CVAT_TOKEN_BEARER", "").lower() in ("1", "true", "yes")
logger = logging.getLogger("web_api")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

UPLOAD_DIR = Path("uploads")
EXPORT_DIR = Path("exports")
PREANN_DIR = UPLOAD_DIR / "preannotations"
VIDEO_FRAMES_DIR = UPLOAD_DIR / "video_frames"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

for _p in (UPLOAD_DIR, EXPORT_DIR, PREANN_DIR, VIDEO_FRAMES_DIR):
    _p.mkdir(parents=True, exist_ok=True)

_default_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
_origins = [
    origin.strip()
    for origin in os.environ.get("CORS_ORIGINS", ",".join(_default_origins)).split(",")
    if origin.strip()
]


app = FastAPI(title="Web API for CVAT + ML")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ExportBody(BaseModel):
    format_name: str = Field(default="COCO 1.0")
    include_images: bool = Field(default=False)
    task_name: str = Field(default="task")


class GrantBody(BaseModel):
    reviewer_user: str


_STATUS_LOCK = threading.Lock()
_PREANN_STATUS: dict[int, dict[str, Any]] = {}


def _set_status(task_id: int, patch: dict[str, Any]) -> None:
    with _STATUS_LOCK:
        current = dict(_PREANN_STATUS.get(task_id, {}))
        current["updated_at"] = int(time.time())
        current.update(patch)
        _PREANN_STATUS[task_id] = current


def _build_preview_from_zip(archive_path: Path) -> tuple[str | None, str | None]:
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            preview_names = sorted(n for n in zf.namelist() if n.startswith("previews/") and n.lower().endswith(".png"))
            if not preview_names:
                return None, None
            preview_name = preview_names[0]
            preview_bytes = zf.read(preview_name)
            preview_b64 = base64.b64encode(preview_bytes).decode("ascii")
            return Path(preview_name).name, preview_b64
    except Exception:
        return None, None


def _validate_preannotation_archive(archive_path: Path) -> dict[str, Any]:
    if not archive_path.exists():
        raise RuntimeError("Архив предразметки не создан")
    if archive_path.stat().st_size == 0:
        raise RuntimeError("Архив предразметки пустой")

    with zipfile.ZipFile(archive_path, "r") as zf:
        names = set(zf.namelist())
        ann_path = "annotations/annotations_coco.json"
        if ann_path not in names:
            raise RuntimeError("В архиве нет annotations/annotations_coco.json")
        ann_raw = zf.read(ann_path)
        ann_json = json.loads(ann_raw.decode("utf-8"))
        annotations = ann_json.get("annotations", [])
        images = ann_json.get("images", [])
        categories = ann_json.get("categories", [])
        if not isinstance(annotations, list):
            raise RuntimeError("Некорректный формат annotations в COCO")
        if not isinstance(images, list):
            raise RuntimeError("Некорректный формат images в COCO")
        if not isinstance(categories, list):
            raise RuntimeError("Некорректный формат categories в COCO")
        if len(images) == 0:
            raise RuntimeError("ML не вернул ни одного изображения в COCO")
        if len(annotations) == 0:
            raise RuntimeError("ML не вернул ни одной аннотации (annotations=0)")
        return {
            "images_count": len(images),
            "annotations_count": len(annotations),
            "categories_count": len(categories),
        }


def _cvat_auth_headers() -> dict[str, str]:
    if not CVAT_TOKEN:
        return {}
    if CVAT_TOKEN_BEARER:
        return {"Authorization": f"Bearer {CVAT_TOKEN}"}
    return {"Authorization": f"Token {CVAT_TOKEN}"}


def _poll_cvat_request(rq_id: int | str, *, timeout_s: int = 600, poll_s: float = 2.0) -> dict[str, Any]:
    """
    Wait until CVAT request is completed or failed.
    Returns the JSON body from /api/requests/ on completion.
    CVAT Cloud (app.cvat.ai) often returns a composite rq_id that is not pollable (404) —
    import still completes; caller should treat poll_unavailable as non-fatal.
    """
    if rq_id is None:
        raise RuntimeError("rq_id is None")

    # Cloud composite id, e.g. action=import&target=task&target_id=123&subresource=annotations
    if isinstance(rq_id, str) and "=" in rq_id and not rq_id.isdigit():
        logger.warning(
            "CVAT composite rq_id is not pollable on cloud (%s); assuming import accepted",
            rq_id[:120],
        )
        return {"status": "poll_skipped", "rq_id": rq_id, "reason": "composite_rq_id_not_pollable"}

    if isinstance(rq_id, str) and rq_id.isdigit():
        url = f"{CVAT_URL}/api/requests/{int(rq_id)}"
    else:
        url = f"{CVAT_URL}/api/requests/{rq_id}"

    headers = _cvat_auth_headers()
    if not headers:
        raise RuntimeError("CVAT_TOKEN is not set; cannot poll import request status")

    start = time.time()
    last: dict[str, Any] | None = None
    while True:
        r = requests.get(url, headers=headers, timeout=30, verify=(not CVAT_INSECURE))
        if r.status_code == 404:
            logger.warning(
                "CVAT request poll 404 for rq_id=%s (cloud); import may already be complete",
                rq_id,
            )
            return {"status": "poll_unavailable", "rq_id": rq_id, "http_status": 404}
        if r.status_code != 200:
            raise RuntimeError(f"CVAT request poll failed: status={r.status_code} body={r.text[:2000]}")
        try:
            last = r.json()
        except Exception as e:
            raise RuntimeError(f"CVAT request poll returned non-JSON: {e}; body={r.text[:2000]}")

        status = str(last.get("status") or last.get("state") or last.get("result") or "").lower()
        if status in ("completed", "success", "finished", "ok"):
            return last
        if status in ("failed", "error"):
            raise RuntimeError(f"CVAT import request failed: {last}")

        if time.time() - start > timeout_s:
            raise TimeoutError(f"CVAT import request {rq_id} not finished after {timeout_s}s (last={last})")
        time.sleep(poll_s)


def _run_preannotation_for_task(
    task_id: int,
    local_paths: list[str],
    payload: dict[str, Any],
    *,
    media_type: str = "image",
    video_meta: dict[str, Any] | None = None,
) -> None:
    _set_status(
        task_id,
        {
            "status": "processing",
            "phase": "starting",
            "started_at": int(time.time()),
            "media_type": media_type,
        },
    )
    opened_files: list[Any] = []
    try:
        inference_paths = list(local_paths)
        extracted_frames: list[Any] = []

        if media_type == "video":
            if not local_paths:
                raise RuntimeError("Video path is missing for preannotation")
            video_path = local_paths[0]
            _set_status(task_id, {"status": "processing", "phase": "fetching_cvat_meta"})
            expected_frames = int(payload.get("video_expected_frames") or 2)
            meta = wait_for_cvat_video_meta(
                task_id,
                min_frames=expected_frames,
                timeout=int(payload.get("video_meta_timeout", 1800)),
            )
            frame_meta = resolve_cvat_frame_list(meta)
            _set_status(
                task_id,
                {
                    "cvat_task_size": int(meta.get("size") or 0),
                    "cvat_frames_resolved": len(frame_meta),
                    "video_expected_frames": expected_frames,
                },
            )
            if len(frame_meta) < 2:
                raise RuntimeError(
                    f"CVAT video task has only {len(frame_meta)} frame(s); "
                    "wait for decode or check frame_step / max_frames"
                )

            start_frame = int(meta.get("start_frame") or 0)
            frame_step = parse_frame_step(str(meta.get("frame_filter") or ""))
            ml_frame_cap = min(int(payload.get("video_max_frames") or 300), ML_VIDEO_MAX_FRAMES)
            if len(frame_meta) > ml_frame_cap:
                logger.warning(
                    "task=%s capping CVAT frames for ML %d -> %d",
                    task_id,
                    len(frame_meta),
                    ml_frame_cap,
                )
                frame_meta = frame_meta[:ml_frame_cap]
                _set_status(task_id, {"ml_frames_capped_to": ml_frame_cap})
            extract_dir = VIDEO_FRAMES_DIR / f"task_{task_id}_{int(time.time())}"
            _set_status(
                task_id,
                {
                    "status": "processing",
                    "phase": "extracting_frames",
                    "frames_total": len(frame_meta),
                    "frames_done": 0,
                },
            )

            def _frame_progress(done: int, total: int, name: str) -> None:
                if done == 1 or done == total or done % 10 == 0:
                    _set_status(
                        task_id,
                        {
                            "status": "processing",
                            "phase": "extracting_frames",
                            "frames_total": total,
                            "frames_done": done,
                            "last_frame_name": name,
                        },
                    )

            extracted_frames = extract_video_frames_for_cvat(
                video_path,
                extract_dir,
                frame_meta,
                start_frame=start_frame,
                frame_step=frame_step,
                progress_cb=_frame_progress,
            )
            inference_paths = [f.local_path for f in extracted_frames]
            _set_status(
                task_id,
                {
                    "frames_extracted": len(inference_paths),
                    "cvat_frame_count": len(frame_meta),
                    "cvat_frame_step": frame_step,
                    "sample_cvat_frame_names": [f.cvat_name for f in extracted_frames[:5]],
                },
            )
            logger.info(
                "task=%s video preannotation: extracted %d/%d frames (step=%s start=%s)",
                task_id,
                len(inference_paths),
                len(frame_meta),
                frame_step,
                start_frame,
            )

        logger.info(
            "task=%s preannotation start: files=%d media_type=%s task_type=%s use_qwen=%s use_clip=%s",
            task_id,
            len(inference_paths),
            media_type,
            payload.get("task_type"),
            bool(payload.get("use_qwen", False)),
            bool(payload.get("use_clip", False)),
        )

        classes = payload.get("classes") or []
        if not isinstance(classes, list):
            classes = [str(classes)]
        classes = [str(c).strip() for c in classes if str(c).strip()]
        task_type = str(payload.get("task_type", "detection"))
        classification_mode = str(payload.get("classification_mode", "object"))
        # Object classification: GND per-class is primary; SigLIP optional (often confuses PPE).
        default_siglip = task_type == "classification" and classification_mode == "image"
        ml_payload = {
            "score_threshold": float(payload.get("score_threshold", 0.35 if task_type == "classification" else 0.3)),
            "max_boxes": int(payload.get("max_boxes", 20 if task_type == "classification" else 10)),
            "format": "coco",
            "task_type": task_type,
            "class_names": classes,
            "text_prompts": classes if task_type == "segmentation" else (payload.get("text_prompts") or []),
            "classification_mode": str(payload.get("classification_mode", "object")),
            "class_size_hints": payload.get("class_size_hints") or {},
            "use_clip": bool(payload.get("use_clip", default_siglip)),
            "use_siglip": bool(payload.get("use_siglip", payload.get("use_clip", default_siglip))),
            "use_qwen": bool(payload.get("use_qwen", False)),
            "qwen_instruction": str(payload.get("qwen_instruction", "")),
        }
        _set_status(task_id, {"ml_payload": ml_payload})

        ml_health: dict[str, Any] | None = None
        try:
            health_resp = requests.get(f"{ML_SERVICE_URL}/health", timeout=8)
            if health_resp.status_code == 200:
                ml_health = health_resp.json()
                _set_status(task_id, {"ml_health": ml_health})
                logger.info(
                    "task=%s ml health: strategy=%s qwen_available=%s",
                    task_id,
                    ml_health.get("strategy_suggested"),
                    ml_health.get("qwen_available"),
                )
        except Exception as health_err:
            logger.warning("task=%s failed to fetch ml health: %s", task_id, health_err)

        t0 = time.time()
        batch_size = int(payload.get("ml_batch_size") or ML_PREANNOT_BATCH_SIZE)
        if media_type == "video" and len(inference_paths) > batch_size:
            archive_bytes = call_preannotate_batches(
                ML_SERVICE_URL,
                inference_paths,
                ml_payload,
                batch_size=batch_size,
                timeout_s=18000,
                status_cb=lambda patch: _set_status(task_id, patch),
            )
        else:
            _set_status(task_id, {"status": "processing", "phase": "calling_ml", "ml_batch_total": 1})
            files = []
            for p in inference_paths:
                fobj = open(p, "rb")
                opened_files.append(fobj)
                files.append(("images", (Path(p).name, fobj, "image/jpeg")))
            resp = requests.post(
                f"{ML_SERVICE_URL}/preannotate",
                data={"payload": json.dumps(ml_payload)},
                files=files,
                timeout=18000,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"ML service status {resp.status_code}: {resp.text[:2000]}")
            archive_bytes = resp.content

        elapsed = round(time.time() - t0, 2)
        logger.info("task=%s ml preannotate finished in %ss (%d files)", task_id, elapsed, len(inference_paths))

        archive_path = PREANN_DIR / f"preann_task_{task_id}_{int(time.time())}.zip"
        with open(archive_path, "wb") as fh:
            fh.write(archive_bytes)

        _set_status(task_id, {"status": "processing", "phase": "validating_archive"})
        archive_stats = _validate_preannotation_archive(archive_path)
        logger.info(
            "task=%s archive valid: images=%s annotations=%s categories=%s",
            task_id,
            archive_stats["images_count"],
            archive_stats["annotations_count"],
            archive_stats["categories_count"],
        )

        if media_type == "video" and extracted_frames:
            _set_status(task_id, {"status": "processing", "phase": "remapping_coco_for_cvat"})
            remap_stats = patch_coco_archive_for_cvat_video(archive_path, extracted_frames)
            _set_status(task_id, {"coco_remap_stats": remap_stats})
            if int(remap_stats.get("matched_images") or 0) == 0:
                raise RuntimeError(
                    "COCO file_name не совпали с кадрами CVAT; импорт был бы пустым. "
                    f"unmatched={remap_stats.get('unmatched_file_names')}"
                )

        _set_status(task_id, {"status": "processing", "phase": "importing_to_cvat"})
        rq_id = import_annotations_to_task(task_id, str(archive_path), format_name="COCO 1.0", timeout=600)
        _set_status(task_id, {"import_request_id": rq_id})

        cvat_rq_json: dict[str, Any] | None = None
        if rq_id is not None:
            _set_status(task_id, {"status": "processing", "phase": "waiting_cvat_import"})
            try:
                cvat_rq_json = _poll_cvat_request(rq_id, timeout_s=900, poll_s=2.0)
                _set_status(task_id, {"import_request_status": "completed", "import_request": cvat_rq_json})
            except Exception as e:
                _set_status(task_id, {"import_request_status": "failed", "import_request_error": str(e)})
                raise
        else:
            _set_status(task_id, {"import_request_status": "sync_or_unknown"})
        preview_name, preview_b64 = _build_preview_from_zip(archive_path)

        _set_status(
            task_id,
            {
                "status": "done",
                "phase": "done",
                "finished_at": int(time.time()),
                "archive_path": str(archive_path),
                "archive_size": archive_path.stat().st_size if archive_path.exists() else None,
                "preview_filename": preview_name,
                "preview_base64": preview_b64,
                "import_request_id": rq_id,
                "archive_stats": archive_stats,
            },
        )
        logger.info("task=%s preannotation done; import_request_id=%s", task_id, rq_id)
    except Exception as e:
        logger.exception("task=%s preannotation failed: %s", task_id, e)
        _set_status(task_id, {"status": "error", "phase": "error", "error": str(e)})
    finally:
        for f in opened_files:
            try:
                f.close()
            except Exception:
                pass


@app.get("/api/ml/health")
def ml_health() -> dict[str, Any]:
    try:
        r = requests.get(f"{ML_SERVICE_URL}/health", timeout=8)
        if r.status_code != 200:
            return {"ok": False, "error": f"ml_service status={r.status_code}", "data": None}
        return {"ok": True, "error": None, "data": r.json()}
    except Exception as e:
        return {"ok": False, "error": str(e), "data": None}


@app.post("/api/tasks/upload")
async def upload_task(payload_json: str = Form(...), files: list[Any] = File(...)) -> dict[str, Any]:
    try:
        payload = json.loads(payload_json)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid payload_json: {e}")

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    image_paths: list[str] = []
    video_paths: list[str] = []
    warnings: list[str] = []

    for f in files:
        name = Path(getattr(f, "filename", "")).name
        ext = Path(name).suffix.lower()
        if not name:
            continue
        if ext in VIDEO_EXTENSIONS:
            out_path = UPLOAD_DIR / f"{int(time.time() * 1000)}_{name}"
            data = await f.read()
            with open(out_path, "wb") as fh:
                fh.write(data)
            video_paths.append(str(out_path))
            continue
        if ext in IMAGE_EXTENSIONS:
            out_path = UPLOAD_DIR / f"{int(time.time() * 1000)}_{name}"
            data = await f.read()
            with open(out_path, "wb") as fh:
                fh.write(data)
            image_paths.append(str(out_path))
            continue
        warnings.append(f"Файл {name} пропущен (формат {ext} не поддерживается)")

    if video_paths and image_paths:
        raise HTTPException(status_code=400, detail="Нельзя загружать видео и изображения в одной задаче")
    if len(video_paths) > 1:
        raise HTTPException(status_code=400, detail="В одной задаче допускается только одно видео")

    media_type = "video" if video_paths else "image"
    local_paths = video_paths if video_paths else image_paths
    if not local_paths:
        raise HTTPException(status_code=400, detail="No supported image or video files after filtering")

    class_names = payload.get("classes") or ["object"]
    if not isinstance(class_names, list):
        class_names = [str(class_names)]
    labels = [{"name": str(c), "description": str(payload.get("class_description", "") or "")} for c in class_names if str(c).strip()]
    if not labels:
        labels = [{"name": "object", "description": ""}]

    upload_params: dict[str, Any] | None = None
    video_params: VideoUploadParams | None = None
    video_meta: dict[str, Any] | None = None
    upload_timeout = 600
    expected_frames = 2

    if media_type == "video":
        video_path = local_paths[0]
        try:
            frame_count = get_video_frame_count(video_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Cannot read video: {e}") from e

        video_params = VideoUploadParams(
            frame_step=int(payload.get("video_frame_step", 5)),
            start_frame=int(payload.get("video_start_frame", 0)),
            max_frames=int(payload.get("video_max_frames", 300)) if payload.get("video_max_frames") is not None else 300,
            stop_frame=int(payload["video_stop_frame"]) if payload.get("video_stop_frame") is not None else None,
        ).normalized(video_frame_count=frame_count)
        upload_params = video_params.to_cvat_upload_params()
        upload_timeout = int(payload.get("video_upload_timeout", 1800))
        expected_frames = expected_frame_count(video_params)
        logger.info(
            "Video upload params for %s: %s (source_frames=%s expected_cvat_frames=%s)",
            Path(video_path).name,
            upload_params,
            frame_count,
            expected_frames,
        )

    try:
        task_id = create_task_and_upload(
            local_paths,
            str(payload.get("task_name", "ml_preannot_task")),
            labels=labels,
            timeout=upload_timeout,
            upload_params=upload_params,
            video_min_frames=expected_frames if media_type == "video" else None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create/upload CVAT task: {e}") from e

    if media_type == "video":
        try:
            video_meta = get_task_data_meta(task_id)
        except Exception as e:
            logger.warning("task=%s failed to fetch data/meta after upload: %s", task_id, e)

    run_preannot = bool(payload.get("run_preannot", False))
    if run_preannot:
        thread_payload = dict(payload)
        if media_type == "video" and video_params is not None:
            thread_payload["video_expected_frames"] = expected_frame_count(video_params)
        _set_status(task_id, {"status": "processing", "media_type": media_type})
        thread = threading.Thread(
            target=_run_preannotation_for_task,
            args=(task_id, local_paths, thread_payload),
            kwargs={"media_type": media_type, "video_meta": None},
            daemon=True,
        )
        thread.start()
        preann: dict[str, Any] | None = {"status": "processing", "media_type": media_type}
    else:
        _set_status(task_id, {"status": "skipped", "media_type": media_type})
        preann = None

    result: dict[str, Any] = {
        "task_id": task_id,
        "task_name": str(payload.get("task_name", "ml_preannot_task")),
        "media_type": media_type,
        "warnings": warnings,
        "preannotation": preann,
    }
    if media_type == "video" and video_params is not None:
        result["video"] = {
            "frame_step": video_params.frame_step,
            "start_frame": video_params.start_frame,
            "stop_frame": video_params.stop_frame,
            "max_frames": video_params.max_frames,
            "cvat_frame_count": (video_meta or {}).get("size"),
        }
    return result


@app.get("/api/tasks/{task_id}/preannotation-status")
def preannotation_status(task_id: int = FPath(..., ge=1)) -> dict[str, Any]:
    with _STATUS_LOCK:
        st = _PREANN_STATUS.get(task_id)
    if st is None:
        raise HTTPException(status_code=404, detail="No preannotation status for this task")
    return st


@app.post("/api/tasks/{task_id}/export")
def export_task(task_id: int = FPath(..., ge=1), body: ExportBody = ExportBody()) -> FileResponse:
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in body.task_name).strip("_") or "task"
    out_zip = EXPORT_DIR / f"{safe_name}_{task_id}_{int(time.time())}.zip"

    def _local_preann_fallback() -> Path | None:
        with _STATUS_LOCK:
            st = dict(_PREANN_STATUS.get(task_id, {}))
        archive_path_raw = st.get("archive_path")
        if isinstance(archive_path_raw, str) and archive_path_raw.strip():
            candidate = Path(archive_path_raw)
            if candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0:
                return candidate

        # Process may have restarted and in-memory status can be empty:
        # pick the newest local preannotation archive for this task id.
        candidates = sorted(
            PREANN_DIR.glob(f"preann_task_{task_id}_*.zip"),
            key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
            reverse=True,
        )
        for candidate in candidates:
            if candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0:
                return candidate
        return None

    try:
        exported = export_annotations_by_id(
            task_id=int(task_id),
            out_zip_path=str(out_zip),
            format_name=body.format_name,
            include_images=body.include_images,
        )
    except Exception as e:
        # Fallback: if CVAT export is unavailable (e.g. org/task visibility mismatch),
        # return the already generated ML preannotation archive for this task.
        archive_path = _local_preann_fallback()
        if archive_path is not None:
            logger.warning(
                "task=%s CVAT export failed (%s); fallback to preannotation archive %s",
                task_id,
                e,
                archive_path,
            )
            return FileResponse(
                path=archive_path,
                media_type="application/zip",
                filename=f"{safe_name}_{task_id}_preannotation.zip",
            )
        raise HTTPException(status_code=500, detail=f"Export failed: {e}")

    file_path = Path(exported)
    if not file_path.exists():
        raise HTTPException(status_code=500, detail="Export finished but file not found")

    return FileResponse(path=file_path, media_type="application/zip", filename=file_path.name)


@app.post("/api/tasks/{task_id}/grant-validation")
def grant_validation(task_id: int = FPath(..., ge=1), body: GrantBody = GrantBody(reviewer_user="")) -> dict[str, Any]:
    reviewer = (body.reviewer_user or "").strip()
    if not reviewer:
        raise HTTPException(status_code=400, detail="reviewer_user is required")
    try:
        return grant_validation_access_for_task(task_id=task_id, reviewer_user=reviewer, timeout=120)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "ml_service_url": ML_SERVICE_URL}

