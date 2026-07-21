#api.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional
import io
import json
import shutil
import tempfile
import time
import zipfile

import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore[assignment]

try:
    from pycocotools import mask as mask_utils  # type: ignore
except Exception:
    mask_utils = None  # type: ignore[assignment]

from .qwen_llm import (
    qwen_suggest_prompts,
    ensure_qwen_loaded,
    QWEN_MODEL_ID,
    _ensure_detailed_prompts,
)
from .models import (
    MODEL_STORE,
    SAM_CHECKPOINT,
    SAM_BACKEND,
    SAM3_BACKENDS,
    SAM3_MODEL_ID,
    GND_DINO_CHECKPOINT,
    GND_DINO_CONFIG,
    MAX_MASKS_PER_IMAGE,
    SIGLIP2_MODEL_ID,
    SIGLIP_AVAILABLE,
    SIGLIP_LAST_ERROR,
    logger,
    siglip_classifier_available,
)
from .inference import (
    run_inference_grounding_dino,
    run_inference_siglip_classify,
    run_inference_clip_classify,
    run_inference_sam_auto,
    run_text_guided_segmentation,
    run_object_classification,
    run_image_classification,
    match_label_to_class,
)


app = FastAPI(title="Preannotation Service (auto: SAM/SAM3)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _is_sam3_backend() -> bool:
    return SAM_BACKEND in SAM3_BACKENDS


def _strategy_name(base: str) -> str:
    """
    Return strategy name adjusted for active SAM backend.
    Keeps old names for classic SAM and exposes explicit sam3 names for HF SAM3.
    """
    if not _is_sam3_backend():
        return base
    mapping = {
        "gnd+sam": "gnd+sam3",
        "gnd+sam-text": "gnd+sam3-text",
    }
    return mapping.get(base, base)


def create_coco_structure(images_info, annotations, categories):
    return {"images": images_info, "annotations": annotations, "categories": categories}


def _render_preview_with_masks(
    image: Image.Image,
    anns_for_image: List[Dict[str, Any]],
    categories: List[Dict[str, Any]],
    masks_dir: Path,
    *,
    default_outline: tuple[int, int, int, int] = (255, 0, 0, 220),
) -> Image.Image:
    """Render preview close to CVAT: semi-transparent masks, then bbox + labels on top."""
    preview = image.copy().convert("RGBA")
    base_arr = np.array(preview).astype(np.float32)
    overlay = np.zeros_like(base_arr, dtype=np.float32)
    alpha = np.zeros((base_arr.shape[0], base_arr.shape[1]), dtype=np.float32)
    from PIL import ImageDraw

    color_palette = [
        (255, 64, 64),
        (64, 160, 255),
        (64, 200, 120),
        (255, 196, 64),
        (180, 100, 255),
    ]
    cat_colors: Dict[int, tuple[int, int, int]] = {}

    for ann in anns_for_image:
        cat_id = int(ann.get("category_id", 1))
        if cat_id not in cat_colors:
            cat_colors[cat_id] = color_palette[(cat_id - 1) % len(color_palette)]
        rgb = cat_colors[cat_id]

        mask_rel = ann.get("mask_path")
        if isinstance(mask_rel, str) and mask_rel.strip():
            mp = masks_dir / Path(mask_rel).name
            if mp.exists():
                try:
                    m = (np.array(Image.open(mp).convert("L")) > 0).astype(np.uint8)
                    if m.shape == alpha.shape:
                        overlay[m > 0, 0] = rgb[0]
                        overlay[m > 0, 1] = rgb[1]
                        overlay[m > 0, 2] = rgb[2]
                        alpha[m > 0] = np.maximum(alpha[m > 0], 0.35)
                except Exception:
                    pass

    out = base_arr.copy()
    if alpha.max() > 0:
        for c in range(3):
            out[..., c] = out[..., c] * (1.0 - alpha) + overlay[..., c] * alpha
        out[..., 3] = base_arr[..., 3]
    preview = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGBA")

    draw = ImageDraw.Draw(preview)
    for ann in anns_for_image:
        cat_id = int(ann.get("category_id", 1))
        x, y, w_box, h_box = ann["bbox"]
        draw.rectangle([x, y, x + w_box, y + h_box], outline=default_outline, width=2)
        cat_name = next((c["name"] for c in categories if c["id"] == cat_id), "")
        if cat_name:
            draw.text((x + 3, y + 3), cat_name, fill=(255, 255, 255, 230))
    return preview

def _mask_to_coco_segmentation(mask: np.ndarray) -> tuple[dict[str, Any] | list, int]:
    """
    Convert binary mask (0/1) to COCO segmentation + area.
    Prefer polygons for CVAT compatibility, fallback to RLE if needed.
    """
    m = (mask > 0).astype(np.uint8)
    area = int(m.sum())
    if area <= 0:
        return [], 0
    if cv2 is not None:
        try:
            contours, _ = cv2.findContours(
                (m * 255).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            polygons: list[list[float]] = []
            for cnt in contours:
                if cnt is None or len(cnt) < 3:
                    continue
                if cv2.contourArea(cnt) < 8.0:
                    continue
                coords = cnt.reshape(-1, 2).astype(float).flatten().tolist()
                if len(coords) >= 6:
                    polygons.append(coords)
            if polygons:
                return polygons, area
        except Exception:
            pass
    if mask_utils is None:
        # Fallback: cannot encode RLE -> no segmentation (will be treated as bbox-only)
        return [], area
    rle = mask_utils.encode(np.asfortranarray(m))
    # pycocotools uses bytes for counts -> make JSON-serializable
    if isinstance(rle, dict) and isinstance(rle.get("counts"), (bytes, bytearray)):
        rle["counts"] = rle["counts"].decode("ascii")
    return rle, area


def _normalize_list(value: Any, fallback: List[str]) -> List[str]:
    if isinstance(value, list):
        out = [str(x).strip() for x in value if str(x).strip()]
        return out or fallback
    if isinstance(value, str):
        v = value.strip()
        if not v:
            return fallback
        try:
            parsed = json.loads(v)
            if isinstance(parsed, list):
                out = [str(x).strip() for x in parsed if str(x).strip()]
                return out or fallback
        except Exception:
            pass
        return [x.strip() for x in v.replace(",", "\n").splitlines() if x.strip()] or fallback
    return fallback


def _parse_class_names(data: Dict[str, Any]) -> List[str]:
    raw_class_names = data.get("class_names", None)
    if raw_class_names is None:
        single = data.get("class_name", None)
        if single is None:
            class_names = ["object"]
        elif isinstance(single, str):
            class_names = [single]
        elif isinstance(single, list):
            class_names = [str(x) for x in single if x]
        else:
            class_names = [str(single)]
    else:
        if isinstance(raw_class_names, str):
            try:
                parsed = json.loads(raw_class_names)
                if isinstance(parsed, list):
                    class_names = [str(x) for x in parsed if x]
                else:
                    class_names = [str(parsed)]
            except Exception:
                class_names = [
                    c.strip()
                    for c in raw_class_names.replace(",", "\n").splitlines()
                    if c.strip()
                ]
        elif isinstance(raw_class_names, list):
            class_names = [str(x) for x in raw_class_names if x]
        else:
            class_names = [str(raw_class_names)]

    if not class_names:
        class_names = ["object"]
    return class_names


def _parse_text_prompts(data: Dict[str, Any], fallback: Optional[List[str]] = None) -> List[str]:
    text_prompts = data.get("text_prompts", None)
    if isinstance(text_prompts, str):
        try:
            text_prompts = json.loads(text_prompts)
        except Exception:
            text_prompts = [
                t.strip() for t in text_prompts.replace(",", "\n").splitlines() if t.strip()
            ]
    if text_prompts is None:
        text_prompts = fallback or []
    if not isinstance(text_prompts, list):
        text_prompts = [str(text_prompts)]
    text_prompts = [str(x) for x in text_prompts if str(x).strip()]
    return text_prompts


def _norm_name(v: str) -> str:
    return " ".join(str(v).strip().lower().replace("_", " ").split())


def _match_label_to_class(label: str, class_names: List[str]) -> str:
    if not class_names:
        return "object"
    if not label:
        return class_names[0]
    label_n = _norm_name(label)
    by_norm = {_norm_name(c): c for c in class_names}
    if label_n in by_norm:
        return by_norm[label_n]
    for c in class_names:
        cn = _norm_name(c)
        if cn and (cn in label_n or label_n in cn):
            return c
    return class_names[0]


def _apply_qwen_suggestions(
    *,
    use_qwen: bool,
    image_paths: List[str],
    class_names: List[str],
    text_prompts: List[str],
    qwen_instruction: str,
    task_type: str,
) -> tuple[List[str], List[str]]:
    if not use_qwen or not image_paths:
        return class_names, text_prompts

    try:
        original_classes = _normalize_list(class_names, ["object"])
        original_prompts = _normalize_list(text_prompts, original_classes)

        qwen_result = qwen_suggest_prompts(
            image_paths[0],
            qwen_instruction or "Suggest annotation classes and text prompts for this image.",
            class_names=class_names,
            task_type=task_type or "auto",
        )
        suggested_classes = qwen_result.get("class_names") or class_names
        suggested_prompts = qwen_result.get("text_prompts") or text_prompts

        qwen_classes = _normalize_list(suggested_classes, original_classes)
        qwen_prompts = _normalize_list(
            suggested_prompts, qwen_classes if qwen_classes else original_prompts
        )

        # Keep task labels stable for CVAT import; use Qwen only to improve text prompts.
        class_names = original_classes
        best_prompts_by_class: Dict[str, str] = {}

        for cls, pr in zip(qwen_classes, qwen_prompts):
            mapped = _match_label_to_class(cls, class_names)
            cur = best_prompts_by_class.get(mapped, "")
            if not cur or len(str(pr)) > len(cur):
                best_prompts_by_class[mapped] = str(pr)

        # Ensure one prompt per class and avoid duplicates.
        text_prompts = [
            best_prompts_by_class.get(c, original_prompts[i] if i < len(original_prompts) else c)
            for i, c in enumerate(class_names)
        ]

        if str(qwen_result.get("source", "")).lower() == "fallback":
            text_prompts = _ensure_detailed_prompts(class_names, text_prompts)

        logger.info(
            "Qwen suggestions applied: classes=%s prompts=%s source=%s",
            class_names,
            text_prompts,
            qwen_result.get("source"),
        )
    except Exception as e:
        logger.warning("Qwen suggestion step failed, using original classes/prompts: %s", e)

    return class_names, text_prompts


@app.post("/preannotate")
async def preannotate(payload: str = Form(...), images: List[UploadFile] = File(...)):
    try:
        data = json.loads(payload)
    except Exception as e:
        return JSONResponse({"error": "Invalid payload JSON", "details": str(e)}, status_code=400)

    class_names = _parse_class_names(data)
    text_prompts = _parse_text_prompts(data)

    task_type = (data.get("task_type") or "").lower().strip()
    classification_mode = str(data.get("classification_mode", "object") or "object").strip().lower()
    if classification_mode not in ("object", "image"):
        classification_mode = "object"

    score_thr = float(
        data.get("score_threshold", 0.35 if task_type == "classification" else 0.2)
    )
    max_boxes = int(data.get("max_boxes", 20 if task_type == "classification" else 10))
    out_format = data.get("format", "coco")
    use_clip_flag = bool(data.get("use_clip", False) or data.get("use_siglip", False))

    use_qwen = bool(data.get("use_qwen", False) or data.get("use_qwen_flag", False))
    qwen_instruction = str(data.get("qwen_instruction", "") or "").strip()

    tmpdir = Path(tempfile.mkdtemp(prefix="preann_"))
    images_dir = tmpdir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = tmpdir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    previews_dir = tmpdir / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[str] = []
    for file in images:
        filename = Path(file.filename).name
        out_path = images_dir / filename
        with open(out_path, "wb") as fh:
            content = await file.read()
            fh.write(content)
        saved_paths.append(str(out_path))

    # Qwen applies before category creation so categories reflect suggestions.
    class_names, text_prompts = _apply_qwen_suggestions(
        use_qwen=use_qwen,
        image_paths=saved_paths,
        class_names=class_names,
        text_prompts=text_prompts,
        qwen_instruction=qwen_instruction,
        task_type=task_type or "auto",
    )

    if task_type == "segmentation":
        effective_seg_prompts: List[str] = [str(t) for t in (text_prompts or class_names)]
    else:
        effective_seg_prompts = [str(t) for t in text_prompts]

    has_gnd = MODEL_STORE.get("gnd_model") is not None
    has_sam = MODEL_STORE.get("sam_predictor") is not None
    has_sam_auto = MODEL_STORE.get("sam_automatic_generator") is not None
    has_siglip = siglip_classifier_available()

    if task_type == "detection":
        if text_prompts and (has_gnd or has_sam):
            strategy = _strategy_name("gnd+sam-text")
        elif has_gnd:
            strategy = "gnd-only"
        elif has_sam_auto:
            strategy = "sam-auto"
        elif has_sam:
            strategy = "sam-predictor-only"
        else:
            strategy = "stub"
    elif task_type == "segmentation":
        if has_gnd and has_sam:
            strategy = _strategy_name("gnd+sam-text")
        elif _is_sam3_backend() and has_sam:
            strategy = "sam3-text"
        elif effective_seg_prompts and (has_gnd or has_sam):
            strategy = _strategy_name("gnd+sam-text")
        elif has_sam_auto:
            strategy = "sam-auto"
        elif has_sam:
            strategy = "sam-predictor-only"
        else:
            strategy = "stub"
        use_clip_flag = False
    elif task_type == "classification":
        if classification_mode == "image":
            strategy = "siglip-image" if has_siglip else "stub"
        elif has_gnd:
            strategy = "gnd+siglip-classify" if (use_clip_flag and has_siglip) else "gnd-classify"
        else:
            strategy = "stub"
        if classification_mode == "object" and has_gnd and not use_clip_flag and has_siglip:
            logger.info("Classification object mode: SigLIP2 disabled; using GND phrase mapping only.")
    else:
        if has_gnd and has_sam:
            strategy = _strategy_name("gnd+sam")
        elif has_gnd:
            strategy = "gnd-only"
        elif has_sam_auto:
            strategy = "sam-auto"
        elif has_sam:
            strategy = "sam-predictor-only"
        else:
            strategy = "stub"

    use_siglip_for_classification = use_clip_flag and has_siglip

    logger.info(
        "Selected preannotation strategy: %s (task_type=%s) classes=%s text_prompts=%s "
        "use_qwen=%s classification_mode=%s use_siglip=%s",
        strategy,
        task_type or "auto",
        class_names,
        text_prompts,
        use_qwen,
        classification_mode if task_type == "classification" else "-",
        use_siglip_for_classification if task_type == "classification" else use_clip_flag,
    )

    annotations: List[Dict[str, Any]] = []
    images_info: List[Dict[str, Any]] = []

    # Keep categories strictly aligned with task labels for reliable CVAT import.
    categories = [{"id": i + 1, "name": name} for i, name in enumerate(class_names)]
    name_to_catid = {c["name"]: c["id"] for c in categories}

    ann_id = 1
    img_id = 1

    for p in saved_paths:
        p_path = Path(p)
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            logger.exception("Failed to open image %s", p)
            continue

        w, h = img.size
        images_info.append(
            {"id": img_id, "width": w, "height": h, "file_name": p_path.name}
        )

        if task_type == "classification":
            if classification_mode == "image":
                cls_results = run_image_classification(
                    p,
                    class_names,
                    use_siglip=use_siglip_for_classification or has_siglip,
                )
            else:
                cls_results = run_object_classification(
                    p,
                    class_names,
                    score_threshold=score_thr,
                    max_boxes=max_boxes,
                    use_siglip=use_siglip_for_classification,
                    class_size_hints=data.get("class_size_hints") or {},
                )
            for res in cls_results:
                x0, y0, x1, y1 = [int(v) for v in res["bbox"]]
                label = match_label_to_class(str(res.get("label", "")), class_names)
                cat_id = name_to_catid.get(label, name_to_catid.get(class_names[0], 1))
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cat_id,
                        "bbox": [x0, y0, x1 - x0, y1 - y0],
                        "score": float(res.get("score", 0.0)),
                        "segmentation": [],
                        "iscrowd": 0,
                    }
                )
                ann_id += 1
        else:
            use_text_guided = False
            prompts_for_this: List[str] = []
            if task_type == "segmentation" and effective_seg_prompts:
                use_text_guided = True
                prompts_for_this = effective_seg_prompts
            elif task_type in ("detection", "other") and text_prompts and (
                has_gnd or has_sam
            ):
                use_text_guided = True
                prompts_for_this = text_prompts

            if use_text_guided and prompts_for_this:
                tg_results = run_text_guided_segmentation(
                    p,
                    prompts_for_this,
                    score_threshold=score_thr,
                    max_boxes_per_prompt=max_boxes,
                    sam_multimask_k=3,
                )
                logger.info(
                    "Text-guided segmentation yielded %d masks for %s",
                    len(tg_results),
                    p_path.name,
                )
                for res in tg_results:
                    mask = res["mask"]
                    bbox = res["bbox"]
                    score = float(res["score"])
                    label = res.get("label", res.get("prompt", "object"))
                    mask_fname = f"{p_path.stem}_ann_{ann_id}.png"
                    mask_path = masks_dir / mask_fname
                    try:
                        Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
                    except Exception:
                        mimg = Image.fromarray((mask * 255).astype(np.uint8))
                        mimg.save(mask_path)
                    x0, y0, x1, y1 = bbox
                    fallback_label = class_names[0] if class_names else "object"
                    mapped_label = _match_label_to_class(str(label), class_names)
                    cat_id = name_to_catid.get(
                        mapped_label, name_to_catid.get(fallback_label, 1)
                    )
                    segmentation, area = _mask_to_coco_segmentation(mask)
                    ann = {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cat_id,
                        "bbox": [x0, y0, x1 - x0, y1 - y0],
                        "score": score,
                        "segmentation": segmentation,
                        "iscrowd": 0,
                        "area": area,
                        "mask_path": f"masks/{mask_fname}",
                    }
                    annotations.append(ann)
                    ann_id += 1
            else:
                if strategy in ("gnd-only", "gnd+sam", "gnd+sam3"):
                    boxes_all: List[Dict[str, Any]] = []
                    for cname in class_names:
                        try:
                            boxes = run_inference_grounding_dino(
                                p, cname, score_thr, max_boxes
                            )
                        except Exception as e:
                            logger.exception(
                                "GroundingDINO failed for class %s: %s", cname, e
                            )
                            boxes = []
                        for b in boxes:
                            b["pred_class"] = cname
                            boxes_all.append(b)
                    filtered = sorted(
                        boxes_all,
                        key=lambda x: -x.get("score", 0.0),
                    )[:max_boxes]
                    for b in filtered:
                        x0, y0, x1, y1 = [int(v) for v in b["bbox"]]
                        cat_id = name_to_catid.get(
                            b.get("pred_class", class_names[0]), 1
                        )
                        score_val = float(b.get("score", 0.0))

                        ann = {
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": cat_id,
                            "bbox": [x0, y0, x1 - x0, y1 - y0],
                            "score": score_val,
                            "segmentation": [],
                            "iscrowd": 0,
                        }
                        annotations.append(ann)
                        ann_id += 1
                elif strategy == "sam-auto":
                    proposals = run_inference_sam_auto(p, max_masks=MAX_MASKS_PER_IMAGE)
                    for prop in proposals:
                        mask = prop["mask"]
                        bbox = prop["bbox"]
                        score = float(prop.get("score", 0.0))
                        if score < score_thr:
                            continue
                        mask_fname = f"{p_path.stem}_ann_{ann_id}.png"
                        mask_path = masks_dir / mask_fname
                        Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
                        x0, y0, x1, y1 = bbox
                        segmentation, area = _mask_to_coco_segmentation(mask)
                        ann = {
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": name_to_catid.get(class_names[0], 1),
                            "bbox": [x0, y0, x1 - x0, y1 - y0],
                            "score": score,
                            "segmentation": segmentation,
                            "iscrowd": 0,
                            "area": area,
                            "mask_path": f"masks/{mask_fname}",
                        }
                        annotations.append(ann)
                        ann_id += 1
                else:
                    bbox = [0, 0, w, h]
                    mask = np.zeros((h, w), dtype=np.uint8)
                    mask[:, :] = 1
                    mask_fname = f"{p_path.stem}_ann_{ann_id}.png"
                    Image.fromarray((mask * 255).astype(np.uint8)).save(
                        masks_dir / mask_fname
                    )
                    segmentation, area = _mask_to_coco_segmentation(mask)
                    annotations.append(
                        {
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": name_to_catid.get(class_names[0], 1),
                            "bbox": [0, 0, w, h],
                            "score": 1.0,
                            "segmentation": segmentation,
                            "iscrowd": 0,
                            "area": area,
                            "mask_path": f"masks/{mask_fname}",
                        }
                    )
                    ann_id += 1

        try:
            anns_for_img = [a for a in annotations if a["image_id"] == img_id]
            preview = _render_preview_with_masks(
                img,
                anns_for_img,
                categories,
                masks_dir,
                default_outline=(255, 0, 0, 220),
            )
            preview_path = previews_dir / f"{p_path.stem}_preview.png"
            max_w = 800
            if preview.width > max_w:
                ratio = max_w / preview.width
                preview = preview.resize(
                    (int(preview.width * ratio), int(preview.height * ratio))
                )
            preview.save(preview_path)
        except Exception:
            logger.exception("Failed to create preview for %s", p)

        img_id += 1

    coco = create_coco_structure(images_info, annotations, categories)
    coco_path = tmpdir / "annotations_coco.json"
    with open(coco_path, "w", encoding="utf-8") as fh:
        json.dump(coco, fh, ensure_ascii=False)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(coco_path, arcname="annotations/annotations_coco.json")
        for p in previews_dir.glob("*.png"):
            zf.write(p, arcname=f"previews/{p.name}")
        for p in masks_dir.glob("*.png"):
            zf.write(p, arcname=f"masks/{p.name}")
        for p in images_dir.glob("*"):
            zf.write(p, arcname=f"images/{p.name}")

    try:
        shutil.rmtree(tmpdir)
    except Exception:
        pass

    zip_buffer.seek(0)
    headers = {
        "Content-Disposition": f"attachment; filename=preannotations_{int(time.time())}.zip"
    }
    return StreamingResponse(zip_buffer, media_type="application/zip", headers=headers)


@app.post("/segment_by_text")
async def segment_by_text(payload: str = Form(...), images: List[UploadFile] = File(...)):
    try:
        data = json.loads(payload)
    except Exception as e:
        return JSONResponse({"error": "Invalid payload JSON", "details": str(e)}, status_code=400)

    raw_prompts = data.get("text_prompts", [])
    if isinstance(raw_prompts, str):
        try:
            raw_prompts = json.loads(raw_prompts)
        except Exception:
            raw_prompts = [
                p.strip() for p in raw_prompts.replace(",", "\n").splitlines() if p.strip()
            ]
    text_prompts = [str(x) for x in raw_prompts if x] or ["object"]

    score_thr = float(data.get("score_threshold", 0.2))
    max_boxes = int(data.get("max_boxes", 10))
    task_type = (data.get("task_type") or "auto").lower().strip()
    use_qwen = bool(data.get("use_qwen", False) or data.get("use_qwen_flag", False))
    qwen_instruction = str(data.get("qwen_instruction", "") or "").strip()
    class_names = [str(x) for x in text_prompts if str(x).strip()]

    tmpdir = Path(tempfile.mkdtemp(prefix="textseg_"))
    images_dir = tmpdir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = tmpdir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    previews_dir = tmpdir / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    annotations: List[Dict[str, Any]] = []
    images_info: List[Dict[str, Any]] = []

    saved_paths: List[str] = []
    for file in images:
        filename = Path(file.filename).name
        out_path = images_dir / filename
        with open(out_path, "wb") as fh:
            content = await file.read()
            fh.write(content)
        saved_paths.append(str(out_path))

    # Qwen can refine text prompts before text-guided segmentation.
    class_names, text_prompts = _apply_qwen_suggestions(
        use_qwen=use_qwen,
        image_paths=saved_paths,
        class_names=class_names,
        text_prompts=text_prompts,
        qwen_instruction=qwen_instruction,
        task_type=task_type,
    )

    if not text_prompts:
        text_prompts = ["object"]

    categories = [{"id": i + 1, "name": name} for i, name in enumerate(text_prompts)]
    name_to_catid = {c["name"]: c["id"] for c in categories}

    ann_id = 1
    img_id = 1

    for p in saved_paths:
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            logger.exception("Failed to open image %s", p)
            continue

        w, h = img.size
        images_info.append(
            {"id": img_id, "width": w, "height": h, "file_name": Path(p).name}
        )

        tg_results = run_text_guided_segmentation(
            p,
            text_prompts,
            score_threshold=score_thr,
            max_boxes_per_prompt=max_boxes,
            sam_multimask_k=3,
        )
        logger.info("segment_by_text: %d results for %s", len(tg_results), Path(p).name)
        for res in tg_results:
            mask = res["mask"]
            bbox = res["bbox"]
            score = float(res["score"])
            label = res.get("label", res.get("prompt", text_prompts[0]))
            mask_fname = f"{Path(p).stem}_ann_{ann_id}.png"
            try:
                Image.fromarray((mask * 255).astype(np.uint8)).save(masks_dir / mask_fname)
            except Exception:
                Image.fromarray((mask * 255).astype(np.uint8)).convert("L").save(
                    masks_dir / mask_fname
                )
            x0, y0, x1, y1 = bbox
            cat_id = name_to_catid.get(label, 1)
            segmentation, area = _mask_to_coco_segmentation(mask)
            ann = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "score": score,
                "segmentation": segmentation,
                "iscrowd": 0,
                "area": area,
                "mask_path": f"masks/{mask_fname}",
            }
            annotations.append(ann)
            ann_id += 1

        try:
            anns_for_img = [a for a in annotations if a["image_id"] == img_id]
            preview = _render_preview_with_masks(
                img,
                anns_for_img,
                categories,
                masks_dir,
                default_outline=(0, 255, 0, 220),
            )
            preview_path = previews_dir / f"{Path(p).stem}_preview.png"
            max_w = 800
            if preview.width > max_w:
                ratio = max_w / preview.width
                preview = preview.resize(
                    (int(preview.width * ratio), int(preview.height * ratio))
                )
            preview.save(preview_path)
        except Exception:
            logger.exception("Failed to create preview for %s", p)

        img_id += 1

    coco = create_coco_structure(images_info, annotations, categories)
    coco_path = tmpdir / "annotations_coco.json"
    with open(coco_path, "w", encoding="utf-8") as fh:
        json.dump(coco, fh, ensure_ascii=False)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(coco_path, arcname="annotations/annotations_coco.json")
        for p in previews_dir.glob("*.png"):
            zf.write(p, arcname=f"previews/{p.name}")
        for p in masks_dir.glob("*.png"):
            zf.write(p, arcname=f"masks/{p.name}")
        for p in images_dir.glob("*"):
            zf.write(p, arcname=f"images/{p.name}")

    try:
        shutil.rmtree(tmpdir)
    except Exception:
        pass

    zip_buffer.seek(0)
    headers = {
        "Content-Disposition": f"attachment; filename=textseg_{int(time.time())}.zip"
    }
    return StreamingResponse(zip_buffer, media_type="application/zip", headers=headers)


@app.get("/health")
def health():
    has_gnd = MODEL_STORE.get("gnd_model") is not None
    has_sam = MODEL_STORE.get("sam_predictor") is not None
    has_sam_auto = MODEL_STORE.get("sam_automatic_generator") is not None

    using_local_sam_checkpoint = not _is_sam3_backend()
    sam_ck_exists = bool(SAM_CHECKPOINT and Path(SAM_CHECKPOINT).exists()) if using_local_sam_checkpoint else False
    gnd_ck_exists = bool(GND_DINO_CHECKPOINT and Path(GND_DINO_CHECKPOINT).exists())

    from .models import GND_DINO_CUDA_OPS, GND_DINO_LAST_ERROR

    siglip_loaded = siglip_classifier_available()
    siglip_device = MODEL_STORE.get("siglip_device")
    if siglip_device is None:
        try:
            import torch

            siglip_device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            siglip_device = "cpu"

    return {
        "status": "ok",
        "api_module_file": __file__,
        "models_module_file": __import__("ml_service.models", fromlist=["__file__"]).__file__,
        "strategy_suggested": (
            _strategy_name("gnd+sam")
            if has_gnd and has_sam
            else "sam3-text"
            if _is_sam3_backend() and has_sam
            else "gnd-only"
            if has_gnd
            else "sam-auto"
            if has_sam_auto
            else "sam-predictor-only"
            if has_sam
            else "stub"
        ),
        "groundingdino_available": has_gnd,
        "groundingdino_cuda_ops": GND_DINO_CUDA_OPS,
        "groundingdino_last_error": GND_DINO_LAST_ERROR,
        "sam_predictor_available": has_sam,
        "sam_automatic_generator_available": has_sam_auto,
        "sam_checkpoint_exists": sam_ck_exists,
        "sam_checkpoint_used": using_local_sam_checkpoint,
        "groundingdino_checkpoint_exists": gnd_ck_exists,
        "sam_checkpoint_path": SAM_CHECKPOINT if using_local_sam_checkpoint else None,
        "sam_backend": SAM_BACKEND,
        "sam3_load_mode": MODEL_STORE.get("sam_backend_mode"),
        "sam3_model_id": SAM3_MODEL_ID,
        "groundingdino_checkpoint_path": GND_DINO_CHECKPOINT,
        "groundingdino_config_path": GND_DINO_CONFIG,
        "siglip_available": SIGLIP_AVAILABLE,
        "siglip_model_id": SIGLIP2_MODEL_ID,
        "siglip_classifier_loaded": siglip_loaded,
        "siglip_device": siglip_device,
        "siglip_last_error": SIGLIP_LAST_ERROR,
        # backward-compatible health keys (UI may still read these)
        "clip_package_installed": SIGLIP_AVAILABLE,
        "clip_backend": "siglip2" if siglip_loaded else None,
        "clip_model_loaded": siglip_loaded,
        "clip_preprocess_present": siglip_loaded,
        "clip_device": siglip_device,
        "qwen_available": ensure_qwen_loaded(),
        "qwen_model_id": QWEN_MODEL_ID,
    }