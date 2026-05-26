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

from cvat_client import (
    create_task_and_upload,
    export_annotations_by_id,
    grant_validation_access_for_task,
    import_annotations_to_task,
)

ML_SERVICE_URL = os.environ.get("ML_SERVICE_URL", "http://localhost:8000").rstrip("/")
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

for _p in (UPLOAD_DIR, EXPORT_DIR, PREANN_DIR):
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


def _poll_cvat_request(rq_id: int, *, timeout_s: int = 600, poll_s: float = 2.0) -> dict[str, Any]:
    """
    Wait until CVAT request is completed or failed.
    Returns the JSON body from /api/requests/{id} on completion.
    """
    if rq_id is None:
        raise RuntimeError("rq_id is None")
    url = f"{CVAT_URL}/api/requests/{int(rq_id)}"
    headers = _cvat_auth_headers()
    if not headers:
        raise RuntimeError("CVAT_TOKEN is not set; cannot poll import request status")

    start = time.time()
    last: dict[str, Any] | None = None
    while True:
        r = requests.get(url, headers=headers, timeout=30, verify=(not CVAT_INSECURE))
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


def _run_preannotation_for_task(task_id: int, local_paths: list[str], payload: dict[str, Any]) -> None:
    _set_status(task_id, {"status": "processing", "phase": "starting", "started_at": int(time.time())})
    opened_files: list[Any] = []
    try:
        logger.info(
            "task=%s preannotation start: files=%d task_type=%s use_qwen=%s use_clip=%s",
            task_id,
            len(local_paths),
            payload.get("task_type"),
            bool(payload.get("use_qwen", False)),
            bool(payload.get("use_clip", False)),
        )
        files = []
        for p in local_paths:
            fobj = open(p, "rb")
            opened_files.append(fobj)
            files.append(("images", (Path(p).name, fobj, "image/jpeg")))

        classes = payload.get("classes") or []
        if not isinstance(classes, list):
            classes = [str(classes)]
        classes = [str(c).strip() for c in classes if str(c).strip()]
        task_type = str(payload.get("task_type", "detection"))
        ml_payload = {
            "score_threshold": float(payload.get("score_threshold", 0.3)),
            "max_boxes": int(payload.get("max_boxes", 10)),
            "format": "coco",
            "task_type": task_type,
            "class_names": classes,
            "text_prompts": classes if task_type == "segmentation" else (payload.get("text_prompts") or []),
            "use_clip": bool(payload.get("use_clip", False)),
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
        _set_status(task_id, {"status": "processing", "phase": "calling_ml"})
        resp = requests.post(
            f"{ML_SERVICE_URL}/preannotate",
            data={"payload": json.dumps(ml_payload)},
            files=files,
            timeout=18000,
        )
        elapsed = round(time.time() - t0, 2)
        logger.info("task=%s ml /preannotate status=%s in %ss", task_id, resp.status_code, elapsed)
        if resp.status_code != 200:
            raise RuntimeError(f"ML service status {resp.status_code}: {resp.text[:2000]}")

        archive_path = PREANN_DIR / f"preann_task_{task_id}_{int(time.time())}.zip"
        with open(archive_path, "wb") as fh:
            fh.write(resp.content)

        _set_status(task_id, {"status": "processing", "phase": "validating_archive"})
        archive_stats = _validate_preannotation_archive(archive_path)
        logger.info(
            "task=%s archive valid: images=%s annotations=%s categories=%s",
            task_id,
            archive_stats["images_count"],
            archive_stats["annotations_count"],
            archive_stats["categories_count"],
        )

        _set_status(task_id, {"status": "processing", "phase": "importing_to_cvat"})
        rq_id = import_annotations_to_task(task_id, str(archive_path), format_name="COCO 1.0", timeout=600)
        _set_status(task_id, {"import_request_id": rq_id})

        cvat_rq_json: dict[str, Any] | None = None
        if rq_id is not None:
            _set_status(task_id, {"status": "processing", "phase": "waiting_cvat_import"})
            try:
                cvat_rq_json = _poll_cvat_request(int(rq_id), timeout_s=900, poll_s=2.0)
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

    local_paths: list[str] = []
    warnings: list[str] = []
    allowed_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    for f in files:
        name = Path(getattr(f, "filename", "")).name
        ext = Path(name).suffix.lower()
        if ext not in allowed_exts:
            warnings.append(f"Файл {name} пропущен (формат {ext} не поддерживается)")
            continue
        if not name:
            continue
        out_path = UPLOAD_DIR / f"{int(time.time() * 1000)}_{name}"
        data = await f.read()
        with open(out_path, "wb") as fh:
            fh.write(data)
        local_paths.append(str(out_path))

    if not local_paths:
        raise HTTPException(status_code=400, detail="No supported image files after filtering")

    class_names = payload.get("classes") or ["object"]
    if not isinstance(class_names, list):
        class_names = [str(class_names)]
    labels = [{"name": str(c), "description": str(payload.get("class_description", "") or "")} for c in class_names if str(c).strip()]
    if not labels:
        labels = [{"name": "object", "description": ""}]

    try:
        task_id = create_task_and_upload(local_paths, str(payload.get("task_name", "ml_preannot_task")), labels=labels)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create/upload CVAT task: {e}")

    run_preannot = bool(payload.get("run_preannot", False))
    if run_preannot:
        _set_status(task_id, {"status": "processing"})
        thread = threading.Thread(
            target=_run_preannotation_for_task,
            args=(task_id, local_paths, payload),
            daemon=True,
        )
        thread.start()
        preann: dict[str, Any] | None = {"status": "processing"}
    else:
        _set_status(task_id, {"status": "skipped"})
        preann = None

    return {
        "task_id": task_id,
        "task_name": str(payload.get("task_name", "ml_preannot_task")),
        "warnings": warnings,
        "preannotation": preann,
    }


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

