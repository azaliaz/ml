from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any
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

from .models import (
    MODEL_STORE,
    SAM_CHECKPOINT,
    GND_DINO_CHECKPOINT,
    GND_DINO_CONFIG,
    MAX_MASKS_PER_IMAGE,
    logger,
)
from .inference import (
    run_inference_grounding_dino,
    run_inference_clip_classify,
    run_inference_sam_auto,
    run_text_guided_segmentation,
)


app = FastAPI(title="Preannotation Service (auto: GroundingDINO/SAM)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def create_coco_structure(images_info, annotations, categories):
    return {"images": images_info, "annotations": annotations, "categories": categories}


@app.post("/preannotate")
async def preannotate(payload: str = Form(...), images: List[UploadFile] = File(...)):
    try:
        data = json.loads(payload)
    except Exception as e:
        return JSONResponse({"error": "Invalid payload JSON", "details": str(e)}, status_code=400)

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

    text_prompts = data.get("text_prompts", None)
    if isinstance(text_prompts, str):
        try:
            text_prompts = json.loads(text_prompts)
        except Exception:
            text_prompts = [
                t.strip() for t in text_prompts.replace(",", "\n").splitlines() if t.strip()
            ]
    if text_prompts is None:
        text_prompts = []

    score_thr = float(data.get("score_threshold", 0.2))
    max_boxes = int(data.get("max_boxes", 10))
    out_format = data.get("format", "coco")
    use_clip_flag = bool(data.get("use_clip", False))
    task_type = (data.get("task_type") or "").lower().strip()

    if task_type == "segmentation":
        effective_seg_prompts: List[str] = [str(t) for t in (text_prompts or class_names)]
    else:
        effective_seg_prompts = [str(t) for t in text_prompts]

    has_gnd = MODEL_STORE.get("gnd_model") is not None
    has_sam = MODEL_STORE.get("sam_predictor") is not None
    has_sam_auto = MODEL_STORE.get("sam_automatic_generator") is not None
    has_clip = MODEL_STORE.get("clip_model") is not None and MODEL_STORE.get("clip_preprocess") is not None

    if task_type == "detection":
        strategy = "gnd-only"
    elif task_type == "segmentation":
        if effective_seg_prompts and (has_gnd or has_sam):
            strategy = "gnd+sam-text"
        elif has_sam_auto:
            strategy = "sam-auto"
        elif has_sam:
            strategy = "sam-predictor-only"
        else:
            strategy = "stub"
        use_clip_flag = False
    elif task_type == "classification":
        strategy = "gnd-only"
    else:
        if has_gnd and has_sam:
            strategy = "gnd+sam"
        elif has_gnd:
            strategy = "gnd-only"
        elif has_sam_auto:
            strategy = "sam-auto"
        elif has_sam:
            strategy = "sam-predictor-only"
        else:
            strategy = "stub"

    logger.info(
        "Selected preannotation strategy: %s (task_type=%s) classes=%s text_prompts=%s",
        strategy,
        task_type or "auto",
        class_names,
        text_prompts,
    )

    tmpdir = Path(tempfile.mkdtemp(prefix="preann_"))
    images_dir = tmpdir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = tmpdir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    previews_dir = tmpdir / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    annotations: List[Dict[str, Any]] = []
    images_info: List[Dict[str, Any]] = []

    categories = [{"id": i + 1, "name": name} for i, name in enumerate(class_names)]
    existing_cat_names = {c["name"] for c in categories}
    extra_prompts: List[str] = effective_seg_prompts if task_type == "segmentation" else text_prompts
    for tp in extra_prompts:
        if tp not in existing_cat_names:
            categories.append({"id": len(categories) + 1, "name": tp})
            existing_cat_names.add(tp)
    name_to_catid = {c["name"]: c["id"] for c in categories}

    ann_id = 1
    img_id = 1

    saved_paths: List[str] = []
    for file in images:
        filename = Path(file.filename).name
        out_path = images_dir / filename
        with open(out_path, "wb") as fh:
            content = await file.read()
            fh.write(content)
        saved_paths.append(str(out_path))

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

        use_text_guided = False
        prompts_for_this: List[str] = []
        if task_type == "segmentation" and effective_seg_prompts:
            use_text_guided = True
            prompts_for_this = effective_seg_prompts
        elif task_type != "segmentation" and text_prompts:
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
                fallback_label = prompts_for_this[0]
                cat_id = name_to_catid.get(
                    label, name_to_catid.get(fallback_label, 1)
                )
                ann = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "score": score,
                    "segmentation": [],
                    "iscrowd": 0,
                    "mask_path": f"masks/{mask_fname}",
                }
                annotations.append(ann)
                ann_id += 1
        else:
            if strategy in ("gnd-only", "gnd+sam"):
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

                    if (
                        task_type == "classification"
                        and MODEL_STORE.get("clip_model") is not None
                        and MODEL_STORE.get("clip_preprocess") is not None
                    ):
                        try:
                            from tempfile import NamedTemporaryFile

                            crop = img.crop((x0, y0, x1, y1))
                            with NamedTemporaryFile(suffix=".jpg", delete=False) as tmpf:
                                tmp_path = tmpf.name
                            crop.save(tmp_path)
                            clip_results = run_inference_clip_classify(
                                tmp_path, class_names
                            )
                            if clip_results:
                                best_clip = max(
                                    clip_results,
                                    key=lambda r: float(r.get("score", 0.0)),
                                )
                                pred_label = best_clip.get(
                                    "label", b.get("pred_class", class_names[0])
                                )
                                pred_score = float(best_clip.get("score", 0.0))
                            else:
                                pred_label = b.get(
                                    "pred_class", class_names[0]
                                )
                                pred_score = float(b.get("score", 0.0))
                        except Exception:
                            pred_label = b.get("pred_class", class_names[0])
                            pred_score = float(b.get("score", 0.0))
                        finally:
                            try:
                                import os

                                os.unlink(tmp_path)
                            except Exception:
                                pass
                        cat_id = name_to_catid.get(
                            pred_label, name_to_catid.get(class_names[0], 1)
                        )
                        score_val = pred_score
                    else:
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
                    ann = {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": name_to_catid.get(class_names[0], 1),
                        "bbox": [x0, y0, x1 - x0, y1 - y0],
                        "score": score,
                        "segmentation": [],
                        "iscrowd": 0,
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
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": name_to_catid.get(class_names[0], 1),
                        "bbox": [0, 0, w, h],
                        "score": 1.0,
                        "segmentation": [],
                        "iscrowd": 0,
                        "mask_path": f"masks/{mask_fname}",
                    }
                )
                ann_id += 1

        try:
            preview = img.copy().convert("RGBA")
            from PIL import ImageDraw

            draw = ImageDraw.Draw(preview)
            for a in annotations:
                if a["image_id"] == img_id:
                    x, y, w_box, h_box = a["bbox"]
                    draw.rectangle(
                        [x, y, x + w_box, y + h_box],
                        outline=(255, 0, 0, 200),
                        width=3,
                    )
                    cat_id = a.get("category_id", 1)
                    cat_name = next(
                        (c["name"] for c in categories if c["id"] == cat_id),
                        "",
                    )
                    draw.text(
                        (x + 3, y + 3), cat_name, fill=(255, 255, 255, 220)
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

    tmpdir = Path(tempfile.mkdtemp(prefix="textseg_"))
    images_dir = tmpdir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = tmpdir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    previews_dir = tmpdir / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    annotations: List[Dict[str, Any]] = []
    images_info: List[Dict[str, Any]] = []
    categories = [{"id": i + 1, "name": name} for i, name in enumerate(text_prompts)]
    name_to_catid = {c["name"]: c["id"] for c in categories}

    ann_id = 1
    img_id = 1

    saved_paths: List[str] = []
    for file in images:
        filename = Path(file.filename).name
        out_path = images_dir / filename
        with open(out_path, "wb") as fh:
            content = await file.read()
            fh.write(content)
        saved_paths.append(str(out_path))

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
            ann = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "score": score,
                "segmentation": [],
                "iscrowd": 0,
                "mask_path": f"masks/{mask_fname}",
            }
            annotations.append(ann)
            ann_id += 1

        try:
            preview = img.copy().convert("RGBA")
            from PIL import ImageDraw

            draw = ImageDraw.Draw(preview)
            for a in annotations:
                if a["image_id"] == img_id:
                    x, y, w_box, h_box = a["bbox"]
                    draw.rectangle(
                        [x, y, x + w_box, y + h_box],
                        outline=(0, 255, 0, 200),
                        width=2,
                    )
                    cat_id = a.get("category_id", 1)
                    cat_name = next(
                        (c["name"] for c in categories if c["id"] == cat_id),
                        "",
                    )
                    draw.text(
                        (x + 3, y + 3), cat_name, fill=(255, 255, 255, 220)
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

    sam_ck_exists = bool(SAM_CHECKPOINT and Path(SAM_CHECKPOINT).exists())
    gnd_ck_exists = bool(GND_DINO_CHECKPOINT and Path(GND_DINO_CHECKPOINT).exists())

    from .models import CLIP_AVAILABLE, CLIP_BACKEND, OWL_AVAILABLE

    clip_pkg_installed = bool("CLIP_AVAILABLE" in globals() and CLIP_AVAILABLE)
    clip_model_loaded = MODEL_STORE.get("clip_model") is not None
    clip_preprocess_present = MODEL_STORE.get("clip_preprocess") is not None
    import sys as _sys

    if MODEL_STORE.get("clip_device") is not None:
        clip_device = MODEL_STORE.get("clip_device")
    else:
        try:
            import torch

            clip_device = "cuda" if (("torch" in _sys.modules) and torch.cuda.is_available()) else "cpu"
        except Exception:
            clip_device = "cpu"

    return {
        "status": "ok",
        "strategy_suggested": (
            "gnd+sam"
            if has_gnd and has_sam
            else "gnd-only"
            if has_gnd
            else "sam-auto"
            if has_sam_auto
            else "sam-predictor-only"
            if has_sam
            else "stub"
        ),
        "groundingdino_available": has_gnd,
        "sam_predictor_available": has_sam,
        "sam_automatic_generator_available": has_sam_auto,
        "sam_checkpoint_exists": sam_ck_exists,
        "groundingdino_checkpoint_exists": gnd_ck_exists,
        "sam_checkpoint_path": SAM_CHECKPOINT,
        "groundingdino_checkpoint_path": GND_DINO_CHECKPOINT,
        "groundingdino_config_path": GND_DINO_CONFIG,
        "clip_package_installed": clip_pkg_installed,
        "clip_backend": CLIP_BACKEND,
        "clip_model_loaded": clip_model_loaded,
        "clip_preprocess_present": clip_preprocess_present,
        "clip_device": clip_device,
        "owl_text_detector_available": OWL_AVAILABLE,
    }

