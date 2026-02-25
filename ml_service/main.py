# ml_service/main.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import uvicorn
import tempfile
import shutil
import json
import time
import zipfile
import io
from PIL import Image
import numpy as np
import logging
import sys
import os
from pathlib import Path

SAM_AVAILABLE = False
SAM_AUTO_AVAILABLE = False
GND_DINO_AVAILABLE = False
CLIP_AVAILABLE = False

try:
    import open_clip
    CLIP_BACKEND = "open_clip"
    CLIP_AVAILABLE = True
except Exception:
    try:
        import clip  # OpenAI clip
        CLIP_BACKEND = "clip"
        CLIP_AVAILABLE = True
    except Exception:
        CLIP_AVAILABLE = False

try:
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
    SAM_AUTO_AVAILABLE = True
except Exception:
    try:
        from segment_anything import sam_model_registry, SamPredictor
        SAM_AVAILABLE = True
        SAM_AUTO_AVAILABLE = False
    except Exception:
        SAM_AVAILABLE = False
        SAM_AUTO_AVAILABLE = False

try:
    import groundingdino
    GND_DINO_AVAILABLE = True
except Exception:
    GND_DINO_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("preann_service")

app = FastAPI(title="Preannotation Service (auto: GroundingDINO/SAM)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SAM_CHECKPOINT = os.environ.get("SAM_CHECKPOINT", "").strip()
GND_DINO_CHECKPOINT = os.environ.get("GND_DINO_CHECKPOINT", "").strip()
GND_DINO_CONFIG = os.environ.get("GND_DINO_CONFIG", "").strip()

MAX_MASKS_PER_IMAGE = int(os.environ.get("MAX_MASKS_PER_IMAGE", "30"))

MODEL_STORE: Dict[str, Any] = {
    "sam_model": None,
    "sam_predictor": None,
    "sam_automatic_generator": None,
    "gnd_model": None,
    "gnd_inference_module": None,
    "clip_model": None,
    "clip_preprocess": None,
    "clip_device": None,
}

def _maybe_add_local_package_to_syspath(pkg_name: str) -> bool:
    cwd = Path(__file__).resolve().parent
    candidate = cwd / pkg_name
    if candidate.exists() and candidate.is_dir():
        parent = str(candidate.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
            logger.info("Added %s to sys.path (parent=%s) to try local import.", pkg_name, parent)
        return True
    candidate2 = Path.cwd() / pkg_name
    if candidate2.exists() and candidate2.is_dir():
        parent2 = str(candidate2.parent)
        if parent2 not in sys.path:
            sys.path.insert(0, parent2)
            logger.info("Added %s to sys.path (parent=%s) to try local import.", pkg_name, parent2)
        return True
    return False

def load_sam_model_if_available():
    global SAM_AVAILABLE, SAM_AUTO_AVAILABLE
    try:
        import segment_anything as _sa  # type: ignore
        SAM_AVAILABLE = True
        logger.info("segment_anything import OK (system).")
    except Exception as e:
        logger.info("segment_anything import failed: %s", e)
        if _maybe_add_local_package_to_syspath("segment_anything"):
            try:
                import segment_anything as _sa
                SAM_AVAILABLE = True
                logger.info("segment_anything import OK (local).")
            except Exception as e2:
                logger.warning("segment_anything import still failed after adding local path: %s", e2)
                SAM_AVAILABLE = False
        else:
            SAM_AVAILABLE = False

    if not SAM_AVAILABLE:
        logger.info("SAM not available -> skipping SAM loading.")
        return

    if not SAM_CHECKPOINT:
        logger.info("SAM_CHECKPOINT not set -> SAM disabled.")
        return

    ck_path = Path(SAM_CHECKPOINT)
    if not ck_path.exists():
        logger.warning("SAM_CHECKPOINT file not found: %s -> SAM disabled.", SAM_CHECKPOINT)
        return

    try:
        from segment_anything import sam_model_registry, SamPredictor
        try:
            from segment_anything import SamAutomaticMaskGenerator
            SAM_AUTO_AVAILABLE = True
        except Exception:
            SAM_AUTO_AVAILABLE = False

        model_type = "vit_h"
        ck = SAM_CHECKPOINT.lower()
        if "vit_l" in ck:
            model_type = "vit_l"
        elif "vit_b" in ck:
            model_type = "vit_b"

        sam = sam_model_registry[model_type](checkpoint=SAM_CHECKPOINT)
        predictor = SamPredictor(sam)
        MODEL_STORE["sam_model"] = sam
        MODEL_STORE["sam_predictor"] = predictor
        logger.info("SAM loaded: %s from %s", model_type, SAM_CHECKPOINT)

        if SAM_AUTO_AVAILABLE:
            try:
                mag = SamAutomaticMaskGenerator(sam)
                MODEL_STORE["sam_automatic_generator"] = mag
                logger.info("SamAutomaticMaskGenerator initialized.")
            except Exception as e:
                logger.warning("SamAutomaticMaskGenerator init failed: %s", e)
                MODEL_STORE["sam_automatic_generator"] = None
    except Exception as e:
        logger.exception("Failed to load SAM model: %s", e)
        MODEL_STORE["sam_model"] = None
        MODEL_STORE["sam_predictor"] = None
        MODEL_STORE["sam_automatic_generator"] = None
        SAM_AVAILABLE = False
        SAM_AUTO_AVAILABLE = False

def load_groundingdino_if_available():
    global GND_DINO_AVAILABLE
    try:
        import groundingdino as _gd  # type: ignore
        GND_DINO_AVAILABLE = True
        logger.info("groundingdino import OK (system).")
    except Exception as e:
        logger.info("groundingdino import failed: %s", e)
        if _maybe_add_local_package_to_syspath("groundingdino"):
            try:
                import groundingdino as _gd  # type: ignore
                GND_DINO_AVAILABLE = True
                logger.info("groundingdino import OK (local).")
            except Exception as e2:
                logger.warning("groundingdino import still failed after adding local path: %s", e2)
                GND_DINO_AVAILABLE = False
        else:
            GND_DINO_AVAILABLE = False

    if not GND_DINO_AVAILABLE:
        logger.info("GroundingDINO package not available -> GroundingDINO disabled.")
        return

    if not GND_DINO_CHECKPOINT:
        logger.info("GND_DINO_CHECKPOINT not set -> GroundingDINO disabled.")
        return

    ck_path = Path(GND_DINO_CHECKPOINT)
    if not ck_path.exists():
        logger.warning("GND_DINO_CHECKPOINT file not found: %s -> GroundingDINO disabled.", GND_DINO_CHECKPOINT)
        return

    config_path = GND_DINO_CONFIG or None
    default_config = Path("GrouningDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    if config_path is None and default_config.exists():
        config_path = str(default_config)

    try:
        from groundingdino.util import inference as gnd_inference  # type: ignore
        MODEL_STORE["gnd_inference_module"] = gnd_inference
    except Exception as e:
        logger.exception("Failed to import groundingdino.util.inference: %s", e)
        MODEL_STORE["gnd_model"] = None
        MODEL_STORE["gnd_inference_module"] = None
        GND_DINO_AVAILABLE = False
        return

    model_wrapper = None
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    try:
        import inspect
        load_fn = getattr(MODEL_STORE["gnd_inference_module"], "load_model", None)
        if load_fn:
            params = list(inspect.signature(load_fn).parameters.keys())
            logger.info("Found groundingdino.load_model signature params: %s", params)
            try:
                if "model_config_path" in params and "model_checkpoint_path" in params:
                    model_wrapper = load_fn(model_config_path=config_path, model_checkpoint_path=GND_DINO_CHECKPOINT, device=device)
                elif "config" in params and "checkpoint" in params:
                    model_wrapper = load_fn(config=config_path, checkpoint=GND_DINO_CHECKPOINT, device=device)
                elif len(params) >= 2:
                    model_wrapper = load_fn(config_path, GND_DINO_CHECKPOINT, device)
                elif len(params) == 1:
                    model_wrapper = load_fn(GND_DINO_CHECKPOINT)
                else:
                    model_wrapper = load_fn()
                logger.info("GroundingDINO loaded via load_model.")
            except Exception as e:
                logger.debug("load_model attempts failed: %s", e)
                model_wrapper = None

        if model_wrapper is None and hasattr(MODEL_STORE["gnd_inference_module"], "Model"):
            ModelClass = getattr(MODEL_STORE["gnd_inference_module"], "Model")
            try:
                model_wrapper = ModelClass(model_config_path=config_path, model_checkpoint_path=GND_DINO_CHECKPOINT, device=device)
                logger.info("GroundingDINO loaded via Model(...) with keywords.")
            except Exception:
                try:
                    model_wrapper = ModelClass(config_path, GND_DINO_CHECKPOINT, device)
                    logger.info("GroundingDINO loaded via Model(config, ckpt, device).")
                except Exception:
                    try:
                        model_wrapper = ModelClass(GND_DINO_CHECKPOINT)
                        logger.info("GroundingDINO loaded via Model(checkpoint).")
                    except Exception as e:
                        logger.exception("All attempts to instantiate GroundingDINO.Model failed: %s", e)
                        model_wrapper = None
    except Exception as e:
        logger.exception("Exception during GroundingDINO loader: %s", e)
        model_wrapper = None

    MODEL_STORE["gnd_model"] = model_wrapper
    if model_wrapper is not None:
        logger.info("GroundingDINO model loaded and stored (device=%s).", device)
    else:
        logger.info("GroundingDINO model not loaded.")


def load_clip_if_available(model_name: str = "ViT-B-32", pretrained: str = "openai"):
    if not CLIP_AVAILABLE:
        logger.info("CLIP not available in environment.")
        return

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    try:
        if CLIP_BACKEND == "open_clip":
            model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
            model.to(device)
            model.eval()
            MODEL_STORE["clip_model"] = model
            MODEL_STORE["clip_preprocess"] = preprocess
            MODEL_STORE["clip_device"] = device
            logger.info("open_clip loaded: %s (%s)", model_name, pretrained)
        else:
            model, preprocess = clip.load(model_name, device=device)
            model.to(device)
            model.eval()
            MODEL_STORE["clip_model"] = model
            MODEL_STORE["clip_preprocess"] = preprocess
            MODEL_STORE["clip_device"] = device
            logger.info("openai clip loaded: %s", model_name)
    except Exception as e:
        logger.exception("Failed to load CLIP model: %s", e)
        MODEL_STORE["clip_model"] = None
        MODEL_STORE["clip_preprocess"] = None
        MODEL_STORE["clip_device"] = None

# load models (best-effort)
load_sam_model_if_available()
load_groundingdino_if_available()
load_clip_if_available()

# -------------------------
# Inference helpers (unchanged)
# -------------------------
def run_inference_grounding_dino_stub(image_path: str, text_prompt: str, score_threshold: float = 0.3, max_boxes: int = 10):
    img = Image.open(image_path)
    w, h = img.size
    bbox = [int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75)]
    return [{"bbox": bbox, "score": 0.95, "label": text_prompt}]

def run_inference_grounding_dino_real(
    image_path: str,
    text_prompt: str,
    score_threshold: float = 0.3,
    max_boxes: int = 10
):
    gnd_model = MODEL_STORE.get("gnd_model")
    gnd_inference = MODEL_STORE.get("gnd_inference_module")

    if gnd_model is None or gnd_inference is None:
        logger.warning("GroundingDINO not loaded -> stub fallback")
        return run_inference_grounding_dino_stub(image_path, text_prompt, score_threshold, max_boxes)

    try:
        import torch
        device = "cpu"
        gnd_model = gnd_model.to(device)

        image_source, image = gnd_inference.load_image(image_path)

        boxes, logits, phrases = gnd_inference.predict(
            model=gnd_model,
            image=image,
            caption=text_prompt,
            box_threshold=score_threshold,
            text_threshold=0.25,
            device=device
        )

        if boxes is None or len(boxes) == 0:
            logger.warning("GroundingDINO returned zero boxes.")
            return []
        h, w = image_source.shape[:2]
        results = []
        for box, score, phrase in zip(boxes, logits, phrases):
            score = float(score)
            if score < score_threshold:
                continue

            cx, cy, bw, bh = box.tolist()

            x0 = int((cx - bw / 2) * w)
            y0 = int((cy - bh / 2) * h)
            x1 = int((cx + bw / 2) * w)
            y1 = int((cy + bh / 2) * h)

            results.append({
                "bbox": [x0, y0, x1, y1],
                "score": score,
                "label": phrase
            })

        results = sorted(results, key=lambda x: -x["score"])[:max_boxes]

        logger.info("GroundingDINO detected %d objects.", len(results))
        return results

    except Exception as e:
        logger.exception("GroundingDINO inference crashed: %s", e)
        return []

def run_inference_grounding_dino(image_path: str, text_prompt: str, score_threshold: float = 0.3, max_boxes: int = 10):
    if MODEL_STORE.get("gnd_model") is not None:
        return run_inference_grounding_dino_real(image_path, text_prompt, score_threshold, max_boxes)
    else:
        return run_inference_grounding_dino_stub(image_path, text_prompt, score_threshold, max_boxes)

def run_inference_clip_classify(image_path: str, labels: List[str]):
    if MODEL_STORE.get("clip_model") is None or MODEL_STORE.get("clip_preprocess") is None:
        return [{"label": l, "score": 0.0} for l in labels]

    try:
        import torch
        from PIL import Image
        model = MODEL_STORE["clip_model"]
        preprocess = MODEL_STORE["clip_preprocess"]
        device = MODEL_STORE["clip_device"] or ("cuda" if torch.cuda.is_available() else "cpu")

        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)

        if CLIP_BACKEND == "open_clip":
            text_tokens = open_clip.tokenize(labels).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                text_features = model.encode_text(text_tokens)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                logits = (100.0 * image_features @ text_features.T).squeeze(0)
                probs = logits.softmax(dim=0).cpu().numpy().tolist()
        else:
            text_tokens = clip.tokenize(labels).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                text_features = model.encode_text(text_tokens)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                logits = (100.0 * image_features @ text_features.T).squeeze(0)
                probs = logits.softmax(dim=0).cpu().numpy().tolist()

        results = [{"label": label, "score": float(prob)} for label, prob in zip(labels, probs)]
        results = sorted(results, key=lambda x: -x["score"])
        return results
    except Exception as e:
        logger.exception("CLIP classify failed: %s", e)
        return [{"label": l, "score": 0.0} for l in labels]

def run_inference_sam_stub(image_path: str, box: List[int]):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    mask = np.zeros((h, w), dtype=np.uint8)
    x0, y0, x1, y1 = box
    x0, x1 = max(0, int(x0)), min(w, int(x1))
    y0, y1 = max(0, int(y0)), min(h, int(y1))
    mask[y0:y1, x0:x1] = 1
    return mask

def run_inference_sam_predictor(image_path: str, box: List[int], multimask: bool = False):
    predictor = MODEL_STORE.get("sam_predictor")
    if predictor is None:
        return run_inference_sam_stub(image_path, box)
    try:
        image_np = np.array(Image.open(image_path).convert("RGB"))
        predictor.set_image(image_np)
        x0, y0, x1, y1 = [int(v) for v in box]
        masks, scores, logits = predictor.predict(box=np.array([x0, y0, x1, y1]), multimask_output=multimask)
        if isinstance(masks, np.ndarray) and masks.ndim == 3:
            best_idx = int(np.argmax(scores)) if len(scores) > 0 else 0
            chosen = masks[best_idx]
        else:
            chosen = masks
        return chosen.astype(np.uint8)
    except Exception as e:
        logger.exception("SAM predictor failed: %s", e)
        return run_inference_sam_stub(image_path, box)

def run_inference_sam_auto(image_path: str, max_masks: int = 30):
    mag = MODEL_STORE.get("sam_automatic_generator")
    if mag is None:
        logger.info("SamAutomaticMaskGenerator not available -> fallback to single rectangular mask.")
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        bbox = [int(w*0.25), int(h*0.25), int(w*0.75), int(h*0.75)]
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
        return [{"mask": mask, "bbox": bbox, "score": 0.9, "area": int(mask.sum())}]

    try:
        image_np = np.array(Image.open(image_path).convert("RGB"))
    except Exception as e:
        logger.exception("Failed to open image for SAM auto: %s", e)
        return []

    try:
        results_raw = mag.generate(image_np)
        proposals = []
        for r in results_raw[:max_masks]:
            mask_arr = r.get("segmentation", None)
            bbox_raw = r.get("bbox", None)
            area = int(r.get("area", 0))
            score = float(r.get("predicted_iou", 0.0) or r.get("score", 0.0) or 0.0)
            if mask_arr is None and "segmentation" in r:
                mask_arr = np.array(r["segmentation"], dtype=np.uint8)
            if mask_arr is None:
                continue
            if bbox_raw and len(bbox_raw) == 4:
                x, y, w_box, h_box = bbox_raw
                bbox = [int(x), int(y), int(x + w_box), int(y + h_box)]
            else:
                ys, xs = np.where(mask_arr)
                if len(xs) == 0 or len(ys) == 0:
                    continue
                x0, x1 = int(xs.min()), int(xs.max())
                y0, y1 = int(ys.min()), int(ys.max())
                bbox = [x0, y0, x1, y1]
            proposals.append({"mask": mask_arr.astype(np.uint8), "bbox": bbox, "score": score, "area": area})
        # sort proposals by score or area
        proposals = sorted(proposals, key=lambda x: -float(x.get("score", 0.0)))[:max_masks]
        return proposals
    except Exception as e:
        logger.exception("SamAutomaticMaskGenerator failed: %s", e)
        return []

def create_coco_structure(images_info, annotations, categories):
    return {"images": images_info, "annotations": annotations, "categories": categories}

# -------------------------
# Preannotate endpoint
# -------------------------
@app.post("/preannotate")
async def preannotate(payload: str = Form(...), images: List[UploadFile] = File(...)):
    try:
        data = json.loads(payload)
    except Exception as e:
        return JSONResponse({"error": "Invalid payload JSON", "details": str(e)}, status_code=400)

    # read inputs (note: class_name may be ignored for segmentation strategy below)
    class_name = data.get("class_name", "object")
    score_thr = float(data.get("score_threshold", 0.3))
    max_boxes = int(data.get("max_boxes", 10))
    out_format = data.get("format", "coco")
    use_clip_flag = bool(data.get("use_clip", False))
    task_type = (data.get("task_type") or "").lower().strip()

    # check model availability
    has_gnd = MODEL_STORE.get("gnd_model") is not None
    has_sam = MODEL_STORE.get("sam_predictor") is not None
    has_sam_auto = MODEL_STORE.get("sam_automatic_generator") is not None
    has_clip = MODEL_STORE.get("clip_model") is not None and MODEL_STORE.get("clip_preprocess") is not None

    # Choose strategy. If client explicitly requests segmentation -> force SAM-only strategies.
    if task_type == "detection":
        strategy = "gnd-only"
    elif task_type == "segmentation":
        # CHANGED: force SAM-only behavior for segmentation mode (ignore GroundingDINO/CLIP)
        if has_sam_auto:
            strategy = "sam-auto"
        elif has_sam:
            strategy = "sam-predictor-only"
        else:
            strategy = "stub"
        # ignore class input for segmentation
        class_name = "object"
        use_clip_flag = False
    elif task_type == "classification":
        strategy = "classification"
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

    logger.info("Selected preannotation strategy: %s (task_type=%s)", strategy, task_type or "auto")

    tmpdir = Path(tempfile.mkdtemp(prefix="preann_"))
    images_dir = tmpdir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = tmpdir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    previews_dir = tmpdir / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    annotations = []
    images_info = []
    categories = [{"id": 1, "name": class_name}]

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
        images_info.append({"id": img_id, "width": w, "height": h, "file_name": p_path.name})

        # -------------------------------
        # SAM-only segmentation modes
        # -------------------------------
        if strategy == "sam-auto":
            # SAM AutomaticMaskGenerator → proposals
            proposals = run_inference_sam_auto(p, max_masks=MAX_MASKS_PER_IMAGE)
            logger.info("SAM auto produced %d proposals for %s", len(proposals), p_path.name)
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
                    "category_id": 1,
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "score": score,
                    "segmentation": [],
                    "iscrowd": 0,
                    "mask_path": f"masks/{mask_fname}"
                }
                annotations.append(ann)
                ann_id += 1

        elif strategy == "sam-predictor-only":
            # use a central box and SAM predictor (no GroundingDINO)
            img_w, img_h = w, h
            bbox = [int(img_w * 0.25), int(img_h * 0.25), int(img_w * 0.75), int(img_h * 0.75)]
            mask = run_inference_sam_predictor(p, bbox, multimask=False)
            if mask is None:
                mask = np.zeros((h, w), dtype=np.uint8)
            mask_fname = f"{p_path.stem}_ann_{ann_id}.png"
            mask_path = masks_dir / mask_fname
            Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
            x0, y0, x1, y1 = bbox
            ann = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "score": 1.0,
                "segmentation": [],
                "iscrowd": 0,
                "mask_path": f"masks/{mask_fname}"
            }
            annotations.append(ann)
            ann_id += 1

        # -------------------------------
        # Detection-only (GroundingDINO)
        # -------------------------------
        elif strategy == "gnd-only":
            boxes = run_inference_grounding_dino(p, class_name, score_thr, max_boxes)
            boxes = sorted(boxes, key=lambda x: -x.get("score", 0.0))[:max_boxes]
            logger.info("GroundingDINO produced %d boxes for %s (gnd-only)", len(boxes), p_path.name)
            for b in boxes:
                score = float(b.get("score", 0.0))
                if score < score_thr:
                    continue
                x0, y0, x1, y1 = [int(coord) for coord in b["bbox"]]
                ann = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "score": score,
                    "segmentation": [],
                    "iscrowd": 0
                }
                annotations.append(ann)
                ann_id += 1

        # -------------------------------
        # gnd+sam (if chosen / auto)
        # -------------------------------
        elif strategy == "gnd+sam":
            boxes = run_inference_grounding_dino(p, class_name, score_thr, max_boxes)
            boxes = sorted(boxes, key=lambda x: -x.get("score", 0.0))[:max_boxes]
            logger.info("GroundingDINO produced %d boxes for %s", len(boxes), p_path.name)

            CLIP_VERIFY = bool(use_clip_flag) and has_clip
            clip_threshold = 0.2
            filtered_boxes = []

            for b in boxes:
                x0, y0, x1, y1 = [int(v) for v in b["bbox"]]
                tmpf = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                try:
                    with Image.open(p).convert("RGB") as _im:
                        crop = _im.crop((x0, y0, x1, y1))
                        crop.save(tmpf.name)
                    if CLIP_VERIFY:
                        scores = run_inference_clip_classify(tmpf.name, [class_name])
                        score_val = scores[0]["score"] if scores else 0.0
                        if score_val >= clip_threshold:
                            filtered_boxes.append(b)
                        else:
                            logger.info("Box filtered by CLIP (score=%.4f) for %s", float(score_val), p_path.name)
                    else:
                        filtered_boxes.append(b)
                except Exception as e:
                    logger.exception("Error while cropping/CLIP-checking box: %s", e)
                finally:
                    try:
                        tmpf.close()
                        os.unlink(tmpf.name)
                    except Exception:
                        pass

            logger.info("Filtered boxes: %d -> %d (after CLIP)", len(boxes), len(filtered_boxes))

            # SAM for each filtered box
            for b in filtered_boxes:
                score = float(b.get("score", 0.0))
                if score < score_thr:
                    continue
                x0, y0, x1, y1 = [int(coord) for coord in b["bbox"]]
                mask = run_inference_sam_predictor(p, [x0, y0, x1, y1], multimask=False)
                if mask is None:
                    mask = np.zeros((h, w), dtype=np.uint8)

                mask_fname = f"{p_path.stem}_ann_{ann_id}.png"
                mask_path = masks_dir / mask_fname
                Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)

                ann = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "score": score,
                    "segmentation": [],
                    "iscrowd": 0,
                    "mask_path": f"masks/{mask_fname}"
                }
                annotations.append(ann)
                ann_id += 1

        # -------------------------------
        # classification flow (gnd -> clip -> sam)
        # -------------------------------
        elif strategy == "classification":
            boxes = run_inference_grounding_dino(p, class_name, score_thr, max_boxes)
            boxes = sorted(boxes, key=lambda x: -x.get("score", 0.0))[:max_boxes]
            logger.info("GroundingDINO produced %d boxes for %s (classification flow)", len(boxes), p_path.name)

            CLIP_VERIFY = bool(use_clip_flag) and has_clip
            clip_threshold = 0.2
            filtered_boxes = []

            for b in boxes:
                x0, y0, x1, y1 = [int(v) for v in b["bbox"]]
                tmpf = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                try:
                    with Image.open(p).convert("RGB") as _im:
                        crop = _im.crop((x0, y0, x1, y1))
                        crop.save(tmpf.name)
                    if CLIP_VERIFY:
                        scores = run_inference_clip_classify(tmpf.name, [class_name])
                        score_val = scores[0]["score"] if scores else 0.0
                        if score_val >= clip_threshold:
                            filtered_boxes.append(b)
                        else:
                            logger.info("Box filtered by CLIP (score=%.4f) for %s (classification)", float(score_val), p_path.name)
                    else:
                        filtered_boxes.append(b)
                except Exception as e:
                    logger.exception("Error while cropping/CLIP-checking box (classification flow): %s", e)
                finally:
                    try:
                        tmpf.close()
                        os.unlink(tmpf.name)
                    except Exception:
                        pass

            logger.info("Filtered boxes (classification): %d -> %d (after CLIP)", len(boxes), len(filtered_boxes))

            # if SAM not available, produce bboxes only
            if not has_sam and not has_sam_auto:
                logger.info("SAM not available; classification flow will produce bboxes only.")
                for b in filtered_boxes:
                    score = float(b.get("score", 0.0))
                    if score < score_thr:
                        continue
                    x0, y0, x1, y1 = [int(coord) for coord in b["bbox"]]
                    ann = {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": 1,
                        "bbox": [x0, y0, x1 - x0, y1 - y0],
                        "score": score,
                        "segmentation": [],
                        "iscrowd": 0
                    }
                    annotations.append(ann)
                    ann_id += 1
            else:
                for b in filtered_boxes:
                    score = float(b.get("score", 0.0))
                    if score < score_thr:
                        continue
                    x0, y0, x1, y1 = [int(coord) for coord in b["bbox"]]

                    if has_sam:
                        mask = run_inference_sam_predictor(p, [x0, y0, x1, y1], multimask=False)
                    else:
                        proposals = run_inference_sam_auto(p, max_masks=MAX_MASKS_PER_IMAGE)
                        mask = None
                        best_prop = None
                        bx0, by0, bx1, by1 = x0, y0, x1, y1
                        area_b = max(1, (bx1 - bx0) * (by1 - by0))
                        best_score = -1.0
                        for prop in proposals:
                            px0, py0, px1, py1 = prop["bbox"]
                            inter_x0 = max(bx0, px0); inter_y0 = max(by0, py0)
                            inter_x1 = min(bx1, px1); inter_y1 = min(by1, py1)
                            if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
                                continue
                            inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
                            union = (px1-px0)*(py1-py0) + area_b - inter_area
                            iou = inter_area / union if union > 0 else 0.0
                            if iou > best_score:
                                best_score = iou
                                best_prop = prop
                        if best_prop is not None:
                            mask = best_prop["mask"]
                        else:
                            mask = np.zeros((h, w), dtype=np.uint8)

                    if mask is None:
                        mask = np.zeros((h, w), dtype=np.uint8)

                    mask_fname = f"{p_path.stem}_ann_{ann_id}.png"
                    mask_path = masks_dir / mask_fname
                    Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)

                    ann = {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": 1,
                        "bbox": [x0, y0, x1 - x0, y1 - y0],
                        "score": score,
                        "segmentation": [],
                        "iscrowd": 0,
                        "mask_path": f"masks/{mask_fname}"
                    }
                    annotations.append(ann)
                    ann_id += 1

        # stub fallback
        else:
            bbox = [0, 0, w, h]
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[0:h, 0:w] = 1
            mask_fname = f"{p_path.stem}_ann_{ann_id}.png"
            mask_path = masks_dir / mask_fname
            Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
            ann = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [0, 0, w, h],
                "score": 1.0,
                "segmentation": [],
                "iscrowd": 0,
                "mask_path": f"masks/{mask_fname}"
            }
            annotations.append(ann)
            ann_id += 1

        # preview drawing
        try:
            preview = img.copy().convert("RGBA")
            from PIL import ImageDraw
            draw = ImageDraw.Draw(preview)
            for a in annotations:
                if a["image_id"] == img_id:
                    x, y, w_box, h_box = a["bbox"]
                    draw.rectangle([x, y, x + w_box, y + h_box], outline=(255, 0, 0, 200), width=3)
            preview_path = previews_dir / f"{p_path.stem}_preview.png"
            max_w = 800
            if preview.width > max_w:
                ratio = max_w / preview.width
                preview = preview.resize((int(preview.width * ratio), int(preview.height * ratio)))
            preview.save(preview_path)
        except Exception:
            logger.exception("Failed to create preview for %s", p)

        img_id += 1

    # save COCO and package
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
    headers = {"Content-Disposition": f"attachment; filename=preannotations_{int(time.time())}.zip"}
    return StreamingResponse(zip_buffer, media_type="application/zip", headers=headers)

# health endpoint
@app.get("/health")
def health():
    has_gnd = MODEL_STORE.get("gnd_model") is not None
    has_sam = MODEL_STORE.get("sam_predictor") is not None
    has_sam_auto = MODEL_STORE.get("sam_automatic_generator") is not None

    sam_ck_exists = bool(SAM_CHECKPOINT and Path(SAM_CHECKPOINT).exists())
    gnd_ck_exists = bool(GND_DINO_CHECKPOINT and Path(GND_DINO_CHECKPOINT).exists())

    # CLIP status
    clip_pkg_installed = bool("CLIP_AVAILABLE" in globals() and CLIP_AVAILABLE)
    clip_model_loaded = MODEL_STORE.get("clip_model") is not None
    clip_preprocess_present = MODEL_STORE.get("clip_preprocess") is not None
    clip_device = MODEL_STORE.get("clip_device") if MODEL_STORE.get("clip_device") is not None else ("cuda" if (("torch" in sys.modules) and __import__("torch").cuda.is_available()) else "cpu")
    clip_backend = globals().get("CLIP_BACKEND", None)

    return {
        "status": "ok",
        "strategy_suggested": (
            "gnd+sam" if has_gnd and has_sam else
            "gnd-only" if has_gnd else
            "sam-auto" if has_sam_auto else
            "sam-predictor-only" if has_sam else
            "stub"
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
        "clip_backend": clip_backend,
        "clip_model_loaded": clip_model_loaded,
        "clip_preprocess_present": clip_preprocess_present,
        "clip_device": clip_device,
    }

if __name__ == "__main__":
    module_app = "ml_service.main:app" if Path(__file__).resolve().parent.name == "ml_service" else "main:app"
    uvicorn.run(module_app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)