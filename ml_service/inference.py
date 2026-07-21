# inference.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any
import os
import tempfile
import logging

import numpy as np
from PIL import Image

try:
    import cv2
except Exception:
    cv2 = None  # type: ignore[assignment]

from .models import (
    MODEL_STORE,
    MAX_MASKS_PER_IMAGE,
    logger,
    siglip_classifier_available,
)

# Ensure logger configured (user can set PREANN_DEBUG=1 to enable DEBUG)
logger = logging.getLogger("preann_service")


def _is_sam3_backend() -> bool:
    return str(os.environ.get("SAM_BACKEND", "sam")).strip().lower() in {
        "sam3",
        "sam3_hf",
        "sam3_native",
        "sam3-official",
        "native",
        "hf",
        "huggingface",
    }


def _is_sam3_native_backend() -> bool:
    if MODEL_STORE.get("sam_backend_mode") == "native":
        return True
    return str(os.environ.get("SAM_BACKEND", "sam")).strip().lower() in {
        "sam3_native",
        "sam3-official",
        "native",
    }


def _clip_bbox_to_shape(box: List[int], h: int, w: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [int(v) for v in box]
    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))
    x1 = max(0, min(w, x1))
    y1 = max(0, min(h, y1))
    return x0, y0, x1, y1


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = (a > 0)
    bb = (b > 0)
    inter = float(np.logical_and(aa, bb).sum())
    union = float(np.logical_or(aa, bb).sum())
    if union <= 0.0:
        return 0.0
    return inter / union


def _postprocess_mask(mask: np.ndarray, bbox: List[int], *, is_rope_like: bool) -> np.ndarray:
    m = (np.array(mask) > 0).astype(np.uint8)
    if m.ndim != 2 or m.size == 0:
        return np.zeros_like(np.array(mask), dtype=np.uint8)
    h, w = m.shape
    x0, y0, x1, y1 = _clip_bbox_to_shape(bbox, h, w)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((h, w), dtype=np.uint8)

    box_area = max(1, (x1 - x0) * (y1 - y0))
    constrained = np.zeros_like(m, dtype=np.uint8)
    constrained[y0:y1, x0:x1] = m[y0:y1, x0:x1]

    if cv2 is None:
        return constrained.astype(np.uint8)

    mask_u8 = constrained * 255
    if is_rope_like:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 1))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    if not is_rope_like:
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)

    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    final_mask = np.zeros_like(mask_u8)
    min_component_px = max(6, min(48, int(box_area * 0.002)))
    for i in range(1, nb_components):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_component_px:
            final_mask[output == i] = 255

    if not is_rope_like and int(final_mask.sum()) > 0:
        # Fill tiny holes to get cleaner closed boundaries on compact objects.
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return (final_mask > 0).astype(np.uint8)


def _bbox_iou(a: List[int], b: List[int]) -> float:
    ax0, ay0, ax1, ay1 = [int(v) for v in a]
    bx0, by0, bx1, by1 = [int(v) for v in b]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = float(iw * ih)
    if inter <= 0.0:
        return 0.0
    area_a = float(max(0, ax1 - ax0) * max(0, ay1 - ay0))
    area_b = float(max(0, bx1 - bx0) * max(0, by1 - by0))
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _dedupe_overlapping_boxes(
    boxes: List[Dict[str, Any]],
    *,
    iou_thr: float = 0.5,
) -> List[Dict[str, Any]]:
    """Drop duplicate detections; keep nested small PPE boxes inside large person boxes."""
    if not boxes:
        return []

    def _area(b: List[int]) -> float:
        x0, y0, x1, y1 = [int(v) for v in b]
        return float(max(0, x1 - x0) * max(0, y1 - y0))

    ordered = sorted(boxes, key=lambda x: -float(x.get("score", 0.0)))
    kept: List[Dict[str, Any]] = []
    for cand in ordered:
        cand_bbox = cand["bbox"]
        cand_area = _area(cand_bbox)
        drop = False
        for prev in kept:
            iou = _bbox_iou(cand_bbox, prev["bbox"])
            if iou < iou_thr:
                continue
            prev_area = _area(prev["bbox"])
            # Small object inside a larger one — keep both (e.g. helmet on person).
            if cand_area < prev_area * 0.40 or prev_area < cand_area * 0.40:
                continue
            drop = True
            break
        if not drop:
            kept.append(cand)
    return kept


def _padded_crop(img: Image.Image, box: List[int], *, pad_ratio: float = 0.12) -> Image.Image:
    w, h = img.size
    x0, y0, x1, y1 = _clip_bbox_to_shape(box, h, w)
    if x1 <= x0 or y1 <= y0:
        return img.crop((x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)))
    bw, bh = x1 - x0, y1 - y0
    px, py = int(bw * pad_ratio), int(bh * pad_ratio)
    cx0, cy0 = max(0, x0 - px), max(0, y0 - py)
    cx1, cy1 = min(w, x1 + px), min(h, y1 + py)
    return img.crop((cx0, cy0, cx1, cy1))


def _siglip_class_prompts(class_names: List[str]) -> List[str]:
    """Generic SigLIP prompts from class names (open vocabulary)."""
    return [f"a photo of a {name}" for name in class_names]


def _dedupe_overlapping_masks(
    masks: List[Dict[str, Any]],
    *,
    iou_thr: float = 0.85,
) -> List[Dict[str, Any]]:
    if not masks:
        return []
    ordered = sorted(masks, key=lambda x: -float(x.get("score", 0.0)))
    kept: List[Dict[str, Any]] = []
    for cand in ordered:
        cm = cand.get("mask")
        if cm is None:
            continue
        drop = False
        for k in kept:
            if _mask_iou(cm, k["mask"]) >= iou_thr:
                drop = True
                break
        if not drop:
            kept.append(cand)
    return kept

def run_inference_grounding_dino_stub(
    image_path: str,
    text_prompt: str,
    score_threshold: float = 0.3,
    max_boxes: int = 10,
) -> List[Dict[str, Any]]:
    img = Image.open(image_path)
    w, h = img.size
    bbox = [int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75)]
    logger.debug("GroundingDINO stub: returning one bbox %s for prompt '%s'", bbox, text_prompt)
    return [{"bbox": bbox, "score": 0.95, "label": text_prompt}]


def run_inference_grounding_dino_real(
    image_path: str,
    text_prompt: str,
    score_threshold: float = 0.3,
    max_boxes: int = 10,
) -> List[Dict[str, Any]]:
    gnd_model = MODEL_STORE.get("gnd_model")
    gnd_inference = MODEL_STORE.get("gnd_inference_module")

    if gnd_model is None or gnd_inference is None:
        logger.warning("GroundingDINO not loaded -> stub fallback")
        return run_inference_grounding_dino_stub(image_path, text_prompt, score_threshold, max_boxes)

    from .models import GND_DINO_CUDA_OPS

    if not GND_DINO_CUDA_OPS:
        logger.warning(
            "GroundingDINO CUDA ops missing (_C) — skip inference for '%s'.",
            text_prompt,
        )
        return []

    try:
        import torch
    except Exception:
        torch = None  # type: ignore[assignment]

    if torch is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"

    try:
        gnd_model = gnd_model.to(device)

        image_source, image = gnd_inference.load_image(image_path)

        boxes, logits, phrases = gnd_inference.predict(
            model=gnd_model,
            image=image,
            caption=text_prompt,
            box_threshold=score_threshold,
            text_threshold=0.25,
            device=device,
        )

        if boxes is None or len(boxes) == 0:
            logger.warning("GroundingDINO returned zero boxes for prompt '%s'.", text_prompt)
            return []
        h, w = image_source.shape[:2]
        results: List[Dict[str, Any]] = []
        for box, score, phrase in zip(boxes, logits, phrases):
            score = float(score)
            if score < score_threshold:
                continue

            cx, cy, bw, bh = box.tolist()

            x0 = int((cx - bw / 2) * w)
            y0 = int((cy - bh / 2) * h)
            x1 = int((cx + bw / 2) * w)
            y1 = int((cy + bh / 2) * h)

            results.append(
                {
                    "bbox": [x0, y0, x1, y1],
                    "score": score,
                    "label": phrase,
                }
            )

        results = sorted(results, key=lambda x: -x["score"])[:max_boxes]

        logger.info("GroundingDINO detected %d objects for prompt '%s'.", len(results), text_prompt)
        logger.debug("GroundingDINO boxes: %s", [r["bbox"] for r in results])
        return results

    except Exception as e:
        logger.exception("GroundingDINO inference crashed: %s", e)
        return []


def run_inference_grounding_dino(
    image_path: str, text_prompt: str, score_threshold: float = 0.3, max_boxes: int = 10
) -> List[Dict[str, Any]]:
    if MODEL_STORE.get("gnd_model") is not None:
        return run_inference_grounding_dino_real(image_path, text_prompt, score_threshold, max_boxes)
    else:
        return run_inference_grounding_dino_stub(image_path, text_prompt, score_threshold, max_boxes)


def run_inference_siglip_classify(
    image_path: str,
    labels: List[str],
    *,
    text_prompts: List[str] | None = None,
) -> List[Dict[str, Any]]:
    clf = MODEL_STORE.get("siglip_classifier")
    if clf is None:
        logger.debug("SigLIP2 classifier not loaded, returning zero scores.")
        return [{"label": l, "score": 0.0} for l in labels]

    if not labels:
        return []

    prompts = text_prompts if text_prompts is not None else labels
    if len(prompts) != len(labels):
        prompts = labels

    try:
        import torch

        image = Image.open(image_path).convert("RGB")
        model = clf.model
        tokenizer = clf.tokenizer
        image_processor = getattr(clf, "image_processor", None) or getattr(
            clf, "feature_extractor", None
        )
        if image_processor is None:
            raise RuntimeError("SigLIP2 pipeline has no image processor")

        device = model.device
        text_inputs = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        image_inputs = image_processor(images=image, return_tensors="pt")
        inputs = {**text_inputs, **image_inputs}
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)

        logits = outputs.logits_per_image
        if logits.dim() == 2:
            logits = logits[0]
        probs = torch.softmax(logits, dim=-1)

        results = [
            {"label": str(labels[i]), "score": float(probs[i])}
            for i in range(len(labels))
        ]
        results = sorted(results, key=lambda x: -x["score"])
        logger.info("SigLIP2 classification top: %s", results[0] if results else None)
        return results
    except Exception as e:
        logger.exception("SigLIP2 classify failed: %s", e)
        return [{"label": l, "score": 0.0} for l in labels]


def run_inference_clip_classify(image_path: str, labels: List[str]) -> List[Dict[str, Any]]:
    """Backward-compatible alias for SigLIP2 zero-shot classification."""
    return run_inference_siglip_classify(image_path, labels)


def _norm_class_name(value: str) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").split())


# Size categories (generic — not tied to specific CVAT labels).
# Override per class via API payload: class_size_hints={"helmet": "tiny", "bus": "large"}
_SIZE_CATEGORY_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "tiny": {
        "threshold_boost": -0.03,
        "min_area_ratio": 0.00015,
        "max_area_ratio": 0.10,
        "max_boxes_per_class": 10,
        "reserve_in_final": 4,
        "search_on_person": True,
        "per_person_max": 3,
    },
    "small": {
        "threshold_boost": -0.02,
        "min_area_ratio": 0.0003,
        "max_area_ratio": 0.20,
        "max_boxes_per_class": 6,
        "reserve_in_final": 2,
        "search_on_person": True,
        "per_person_max": 2,
    },
    "medium": {
        "threshold_boost": 0.03,
        "min_area_ratio": 0.003,
        "max_area_ratio": 0.50,
        "max_boxes_per_class": 5,
        "reserve_in_final": 2,
        "search_on_person": False,
        "per_person_max": 1,
    },
    "large": {
        "threshold_boost": -0.02,
        "min_area_ratio": 0.035,
        "max_area_ratio": 0.98,
        "max_boxes_per_class": 3,
        "reserve_in_final": 1,
        "search_on_person": False,
        "per_person_max": 1,
        "requires_height": True,
    },
}

_VALID_SIZE_CATEGORIES = frozenset(_SIZE_CATEGORY_DEFAULTS.keys())


def _infer_size_category(class_name: str) -> str:
    """Weak default when user did not pass class_size_hints."""
    n = _norm_class_name(class_name)
    if any(k in n for k in ("person", "people", "worker", "human", "pedestrian", "man", "woman")):
        return "medium"
    if any(k in n for k in ("crane", "tower", "building", "vehicle", "truck", "bus", "excavator")):
        return "large"
    if any(k in n for k in ("helmet", "glove", "hat", "mask", "glasses", "boot", "vest")):
        return "tiny"
    return "small"


def _build_class_settings(
    class_names: List[str],
    class_size_hints: Dict[str, str] | None = None,
) -> Dict[str, Dict[str, Any]]:
    hints = {_norm_class_name(k): str(v).strip().lower() for k, v in (class_size_hints or {}).items()}
    settings: Dict[str, Dict[str, Any]] = {}
    for class_name in class_names:
        norm = _norm_class_name(class_name)
        category = hints.get(norm) or _infer_size_category(class_name)
        if category not in _VALID_SIZE_CATEGORIES:
            category = "small"
        settings[norm] = {
            "class_name": class_name,
            "size_category": category,
            **_SIZE_CATEGORY_DEFAULTS[category],
        }
    return settings


def _class_cfg(class_name: str, class_settings: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    cfg = class_settings.get(_norm_class_name(class_name))
    if cfg:
        return cfg
    return {"class_name": class_name, "size_category": "small", **_SIZE_CATEGORY_DEFAULTS["small"]}


def _bbox_area_ratio(bbox: List[int], img_w: int, img_h: int) -> float:
    x0, y0, x1, y1 = [int(v) for v in bbox]
    box_area = float(max(0, x1 - x0) * max(0, y1 - y0))
    return box_area / float(max(1, img_w * img_h))


def _gnd_prompt(class_name: str) -> str:
    name = str(class_name).strip()
    if not name:
        return "object ."
    if " . " in name or name.endswith(" ."):
        return name
    return f"{name} ."


def _gnd_settings_for_class(
    class_name: str,
    base_threshold: float,
    class_settings: Dict[str, Dict[str, Any]],
) -> tuple[str, float, float, float]:
    cfg = _class_cfg(class_name, class_settings)
    prompt = _gnd_prompt(str(cfg.get("class_name", class_name)))
    threshold = max(0.22, min(0.65, base_threshold + float(cfg.get("threshold_boost", 0.0))))
    min_area = float(cfg.get("min_area_ratio", 0.0002))
    max_area = float(cfg.get("max_area_ratio", 0.90))
    return prompt, threshold, min_area, max_area


def _passes_bbox_size_filter(
    class_name: str,
    area_ratio: float,
    class_settings: Dict[str, Dict[str, Any]],
    bbox: List[int] | None = None,
    img_w: int = 0,
    img_h: int = 0,
) -> bool:
    cfg = _class_cfg(class_name, class_settings)
    min_area = float(cfg.get("min_area_ratio", 0.0002))
    max_area = float(cfg.get("max_area_ratio", 0.90))
    if area_ratio < min_area or area_ratio > max_area:
        return False

    if not cfg.get("requires_height") or not bbox or img_w <= 0 or img_h <= 0:
        return True

    x0, y0, x1, y1 = [int(v) for v in bbox]
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    height_ratio = bh / float(img_h)
    if area_ratio < 0.045:
        return False
    if height_ratio >= 0.14 and area_ratio >= 0.03:
        return True
    if area_ratio >= 0.06:
        return True
    if bw > bh * 2.5 and area_ratio < 0.12:
        return False
    return height_ratio >= 0.10 and area_ratio >= 0.035


def _person_crop_region(
    person_bbox: List[int],
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = _clip_bbox_to_shape(person_bbox, img_h, img_w)
    ph = max(1, y1 - y0)
    pw = max(1, x1 - x0)
    pad_x = int(pw * 0.08)
    pad_y = int(ph * 0.05)
    return (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(img_w, x1 + pad_x),
        min(img_h, y1 + pad_y),
    )


def _gnd_detect_on_crop(
    img: Image.Image,
    crop_xyxy: tuple[int, int, int, int],
    pred_class: str,
    score_threshold: float,
    max_boxes: int,
    class_settings: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Run GND on a crop and map boxes back to full-image coordinates."""
    import os
    import tempfile

    img_w, img_h = img.size
    x0, y0, x1, y1 = crop_xyxy
    if x1 - x0 < 8 or y1 - y0 < 8:
        return []

    crop = img.crop((x0, y0, x1, y1))
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmpf:
            tmp_path = tmpf.name
        crop.save(tmp_path)
        prompt, class_thr, _, _ = _gnd_settings_for_class(
            pred_class, score_threshold, class_settings
        )
        class_thr = max(0.20, class_thr - 0.05)
        raw = run_inference_grounding_dino(
            tmp_path, prompt, score_threshold=class_thr, max_boxes=max_boxes
        )
        results: List[Dict[str, Any]] = []
        for b in raw:
            bx0, by0, bx1, by1 = [int(v) for v in b["bbox"]]
            full = [bx0 + x0, by0 + y0, bx1 + x0, by1 + y0]
            full = list(_clip_bbox_to_shape(full, img_h, img_w))
            area_ratio = _bbox_area_ratio(full, img_w, img_h)
            if not _passes_bbox_size_filter(
                pred_class, area_ratio, class_settings, full, img_w, img_h
            ):
                continue
            results.append(
                {
                    "bbox": full,
                    "score": float(b.get("score", 0.0)),
                    "label": str(b.get("label", "")),
                    "pred_class": pred_class,
                    "gnd_phrase": str(b.get("label", "")),
                    "area_ratio": area_ratio,
                    "source": "person_crop",
                }
            )
        return results
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def _detect_nested_on_persons(
    img: Image.Image,
    person_boxes: List[Dict[str, Any]],
    class_names: List[str],
    score_threshold: float,
    class_settings: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Second pass: search tiny/small classes inside each medium (person) bbox."""
    nested_classes = [
        c for c in class_names if _class_cfg(c, class_settings).get("search_on_person")
    ]
    if not nested_classes or not person_boxes:
        return []

    img_w, img_h = img.size
    extra: List[Dict[str, Any]] = []
    for person in person_boxes:
        region = _person_crop_region(person["bbox"], img_w, img_h)
        for nested_class in nested_classes:
            per_person_max = int(_class_cfg(nested_class, class_settings).get("per_person_max", 2))
            found = _gnd_detect_on_crop(
                img,
                region,
                nested_class,
                score_threshold,
                per_person_max,
                class_settings,
            )
            extra.extend(found)
    logger.info(
        "Person-crop nested pass: +%d boxes (persons=%d classes=%s)",
        len(extra),
        len(person_boxes),
        [_norm_class_name(c) for c in nested_classes],
    )
    return extra


def _max_boxes_for_class(
    class_name: str,
    max_boxes: int,
    n_classes: int,
    class_settings: Dict[str, Dict[str, Any]],
) -> int:
    cfg = _class_cfg(class_name, class_settings)
    per_class = int(cfg.get("max_boxes_per_class", max(3, max_boxes // max(1, n_classes))))
    return max(per_class, max_boxes // max(1, n_classes))


def _select_final_classification_boxes(
    boxes: List[Dict[str, Any]],
    max_boxes: int,
    class_names: List[str],
    class_settings: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Keep top boxes; reserve slots per size category (many tiny, few large)."""
    if not boxes:
        return []

    def _norm_pred(b: Dict[str, Any]) -> str:
        return _norm_class_name(str(b.get("pred_class", "")))

    ordered = sorted(boxes, key=lambda x: -float(x.get("score", 0.0)))
    kept: List[Dict[str, Any]] = []
    kept_ids: set[int] = set()

    def _add(box: Dict[str, Any]) -> None:
        bid = id(box)
        if bid in kept_ids:
            return
        kept.append(box)
        kept_ids.add(bid)

    for class_name in class_names:
        norm = _norm_class_name(class_name)
        target = int(_class_cfg(class_name, class_settings).get("reserve_in_final", 1))
        class_boxes = [b for b in ordered if _norm_pred(b) == norm]
        for b in class_boxes[:target]:
            _add(b)

    for b in ordered:
        if len(kept) >= max_boxes:
            break
        _add(b)

    if len(kept) > max_boxes:
        kept = sorted(kept, key=lambda x: -float(x.get("score", 0.0)))[:max_boxes]
    return kept


def _siglip_candidates_for_area(
    class_names: List[str],
    area_ratio: float,
    class_settings: Dict[str, Dict[str, Any]],
) -> List[str]:
    """On small crops only compare tiny/small classes (avoid large-structure confusion)."""
    if area_ratio >= 0.10:
        return list(class_names)
    return [
        c
        for c in class_names
        if _class_cfg(c, class_settings).get("size_category") in ("tiny", "small")
    ] or list(class_names)


def match_label_to_class(label: str, class_names: List[str]) -> str:
    if not class_names:
        return "object"
    if not label:
        return class_names[0]
    label_n = _norm_class_name(label)
    by_norm = {_norm_class_name(c): c for c in class_names}
    if label_n in by_norm:
        return by_norm[label_n]
    for c in class_names:
        cn = _norm_class_name(c)
        if cn and (cn in label_n or label_n in cn):
            return c
    return class_names[0]


def build_gnd_caption(class_names: List[str]) -> str:
    parts = [str(c).strip() for c in class_names if str(c).strip()]
    if not parts:
        return "object ."
    return " . ".join(parts) + " ."


def _siglip_best_label(image_path: str, class_names: List[str]) -> tuple[str, float] | None:
    prompts = _siglip_class_prompts(class_names)
    ranked = run_inference_siglip_classify(
        image_path, class_names, text_prompts=prompts
    )
    if not ranked:
        return None
    best = max(ranked, key=lambda x: float(x.get("score", 0.0)))
    score = float(best.get("score", 0.0))
    if score <= 0.0:
        return None
    return str(best.get("label", class_names[0])), score


def _refine_class_with_siglip(
    crop_path: str,
    gnd_class: str,
    class_names: List[str],
    class_settings: Dict[str, Dict[str, Any]],
    *,
    area_ratio: float,
) -> tuple[str, float, str]:
    """
    Verify GND class with SigLIP among size-appropriate candidates only.
    """
    candidates = _siglip_candidates_for_area(class_names, area_ratio, class_settings)
    gnd_norm = _norm_class_name(gnd_class)
    if gnd_norm not in {_norm_class_name(c) for c in candidates}:
        candidates = [gnd_class] + [c for c in candidates if _norm_class_name(c) != gnd_norm]

    prompts = _siglip_class_prompts(candidates)
    ranked = run_inference_siglip_classify(
        crop_path, candidates, text_prompts=prompts
    )
    if not ranked:
        return gnd_class, 0.0, ""

    best = ranked[0]
    second_score = float(ranked[1]["score"]) if len(ranked) > 1 else 0.0
    best_score = float(best.get("score", 0.0))
    margin = best_score - second_score
    siglip_class = match_label_to_class(str(best.get("label", "")), candidates)

    if siglip_class not in candidates:
        return gnd_class, best_score, siglip_class

    if best_score < 0.24:
        return gnd_class, best_score, siglip_class

    if siglip_class == gnd_class:
        return gnd_class, best_score, siglip_class

    # Override only with high confidence within the same size bucket.
    if best_score >= 0.32 and margin >= 0.08:
        logger.debug(
            "SigLIP overrides GND %s -> %s (score=%.3f margin=%.3f area=%.3f)",
            gnd_class,
            siglip_class,
            best_score,
            margin,
            area_ratio,
        )
        return siglip_class, best_score, siglip_class

    return gnd_class, best_score, siglip_class


def run_image_classification(
    image_path: str,
    class_names: List[str],
    *,
    use_siglip: bool = True,
) -> List[Dict[str, Any]]:
    """Whole-image classification: one label per image (full-frame bbox)."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if not class_names:
        class_names = ["object"]

    if use_siglip and siglip_classifier_available():
        siglip_result = _siglip_best_label(image_path, class_names)
        if siglip_result is not None:
            label, score = siglip_result
        else:
            label, score = class_names[0], 0.0
    else:
        label, score = class_names[0], 0.0

    return [
        {
            "bbox": [0, 0, w, h],
            "label": match_label_to_class(label, class_names),
            "score": score,
            "meta": {"mode": "image", "siglip": use_siglip},
        }
    ]


def run_object_classification(
    image_path: str,
    class_names: List[str],
    score_threshold: float = 0.3,
    max_boxes: int = 10,
    *,
    use_siglip: bool = True,
    class_size_hints: Dict[str, str] | None = None,
) -> List[Dict[str, Any]]:
    """
    Object classification: GND per class → NMS → optional SigLIP2 refine on crops.
    Class list comes from CVAT/user; optional class_size_hints override size category.
    """
    import os
    import tempfile

    if not class_names:
        class_names = ["object"]

    class_settings = _build_class_settings(class_names, class_size_hints)
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size
    n_classes = max(1, len(class_names))

    boxes_all: List[Dict[str, Any]] = []
    for cname in class_names:
        prompt, class_thr, _, _ = _gnd_settings_for_class(
            cname, score_threshold, class_settings
        )
        class_max = _max_boxes_for_class(cname, max_boxes, n_classes, class_settings)
        try:
            boxes = run_inference_grounding_dino(
                image_path,
                prompt,
                score_threshold=class_thr,
                max_boxes=class_max,
            )
        except Exception as exc:
            logger.exception("GroundingDINO failed for class %s: %s", cname, exc)
            boxes = []
        for b in boxes:
            bbox = [int(v) for v in b["bbox"]]
            area_ratio = _bbox_area_ratio(bbox, img_w, img_h)
            if not _passes_bbox_size_filter(
                cname, area_ratio, class_settings, bbox, img_w, img_h
            ):
                logger.debug(
                    "Skip GND box for %s: area_ratio=%.4f phrase=%r bbox=%s",
                    cname,
                    area_ratio,
                    b.get("label"),
                    bbox,
                )
                continue
            entry = dict(b)
            entry["bbox"] = bbox
            entry["pred_class"] = cname
            entry["gnd_phrase"] = str(b.get("label", ""))
            entry["area_ratio"] = area_ratio
            boxes_all.append(entry)

    if not boxes_all:
        logger.info("Object classification: GND found 0 boxes for %s", Path(image_path).name)
        return []

    person_boxes = [
        b
        for b in boxes_all
        if _class_cfg(str(b.get("pred_class", "")), class_settings).get("size_category")
        == "medium"
    ]
    if person_boxes and any(
        _class_cfg(c, class_settings).get("search_on_person") for c in class_names
    ):
        boxes_all.extend(
            _detect_nested_on_persons(
                img, person_boxes, class_names, score_threshold, class_settings
            )
        )

    boxes_all = _dedupe_overlapping_boxes(boxes_all, iou_thr=0.5)
    has_tiny = any(
        _class_cfg(c, class_settings).get("size_category") == "tiny" for c in class_names
    )
    effective_max = max(max_boxes, 18 if has_tiny else max_boxes)
    boxes_all = _select_final_classification_boxes(
        boxes_all, effective_max, class_names, class_settings
    )

    results: List[Dict[str, Any]] = []
    for b in boxes_all:
        x0, y0, x1, y1 = [int(v) for v in b["bbox"]]
        gnd_phrase = str(b.get("gnd_phrase", ""))
        pred_class = match_label_to_class(str(b.get("pred_class", "")), class_names)
        pred_score = float(b.get("score", 0.0))
        area_ratio = float(b.get("area_ratio", _bbox_area_ratio(b["bbox"], img_w, img_h)))
        siglip_top = ""

        if use_siglip and siglip_classifier_available():
            tmp_path: str | None = None
            try:
                crop = _padded_crop(img, [x0, y0, x1, y1])
                if crop.size[0] > 0 and crop.size[1] > 0:
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmpf:
                        tmp_path = tmpf.name
                    crop.save(tmp_path)
                    pred_class, pred_score, siglip_top = _refine_class_with_siglip(
                        tmp_path,
                        pred_class,
                        class_names,
                        class_settings,
                        area_ratio=area_ratio,
                    )
            except Exception as exc:
                logger.debug("SigLIP2 crop classify failed: %s", exc)
            finally:
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

        results.append(
            {
                "bbox": [x0, y0, x1, y1],
                "label": pred_class,
                "score": pred_score,
                "meta": {
                    "mode": "object",
                    "gnd_phrase": gnd_phrase,
                    "gnd_class": match_label_to_class(str(b.get("pred_class", "")), class_names),
                    "area_ratio": area_ratio,
                    "siglip_top": siglip_top,
                    "siglip": use_siglip,
                },
            }
        )

    logger.info(
        "Object classification: %d boxes for %s (filtered per-class GND siglip=%s)",
        len(results),
        Path(image_path).name,
        use_siglip,
    )
    return results


def run_inference_sam_predictor(
    image_path: str, box: List[int], multimask: bool = False
) -> tuple[np.ndarray, float]:
    predictor = MODEL_STORE.get("sam_predictor")
    if predictor is None:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        mask = np.zeros((h, w), dtype=np.uint8)
        x0, y0, x1, y1 = box
        x0, x1 = max(0, int(x0)), min(w, int(x1))
        y0, y1 = max(0, int(y0)), min(h, int(y1))
        mask[y0:y1, x0:x1] = 1
        logger.debug("SAM predictor missing -> returning bbox-as-mask for box %s", box)
        return mask, 1.0
    try:
        image_np = np.array(Image.open(image_path).convert("RGB"))
        predictor.set_image(image_np)
        x0, y0, x1, y1 = [int(v) for v in box]
        masks, scores, logits = predictor.predict(
            box=np.array([x0, y0, x1, y1]),
            multimask_output=multimask,
        )
        # log info about masks returned (counts & top score)
        if isinstance(masks, np.ndarray) and masks.ndim == 3:
            n = masks.shape[0]
            best_idx = int(np.argmax(scores)) if len(scores) > 0 else 0
            chosen = masks[best_idx]
            score = float(scores[best_idx]) if len(scores) > 0 else 1.0
            logger.info("SAM predictor returned %d masks for box %s, best_score=%.3f", n, box, score)
            logger.debug("SAM predictor scores: %s", scores.tolist() if hasattr(scores, "tolist") else scores)
            return chosen.astype(np.uint8), float(score)
        else:
            # single mask
            mask = masks
            logger.info("SAM predictor returned single mask for box %s", box)
            return mask.astype(np.uint8), 1.0
    except Exception as e:
        logger.exception("SAM predictor failed: %s", e)
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        mask = np.zeros((h, w), dtype=np.uint8)
        x0, y0, x1, y1 = box
        x0, x1 = max(0, int(x0)), min(w, int(x1))
        y0, y1 = max(0, int(y0)), min(h, int(y1))
        mask[y0:y1, x0:x1] = 1
        return mask, 0.0


def run_inference_sam_auto(
    image_path: str,
    max_masks: int = 30,
    text_prompt: str | None = None,
) -> List[Dict[str, Any]]:
    mag = MODEL_STORE.get("sam_automatic_generator")
    if mag is None:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        bbox = [int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75)]
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[bbox[1] : bbox[3], bbox[0] : bbox[2]] = 1
        logger.debug("SAM auto not available -> returning single proposal")
        return [{"mask": mask, "bbox": bbox, "score": 0.9, "area": int(mask.sum())}]
    try:
        image_np = np.array(Image.open(image_path).convert("RGB"))
    except Exception as e:
        logger.exception("Failed to open image for SAM auto: %s", e)
        return []

    try:
        # HF SAM3 adapter supports text_prompt; classic SAM generator ignores prompt.
        if text_prompt:
            try:
                results_raw = mag.generate(image_np, text_prompt=text_prompt)
            except TypeError:
                results_raw = mag.generate(image_np)
        else:
            results_raw = mag.generate(image_np)
        proposals: List[Dict[str, Any]] = []
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
            proposals.append(
                {"mask": mask_arr.astype(np.uint8), "bbox": bbox, "score": score, "area": area}
            )
        proposals = sorted(proposals, key=lambda x: -float(x.get("score", 0.0)))[:max_masks]
        logger.info("SAM Auto generated %d proposals for image %s", len(proposals), image_path)
        return proposals
    except Exception as e:
        logger.exception("SamAutomaticMaskGenerator failed: %s", e)
        return []


def run_inference_sam_multimask(
    image_path: str, box: List[int], max_masks: int = 3
) -> List[Dict[str, Any]]:
    predictor = MODEL_STORE.get("sam_predictor")
    if predictor is None:
        mask, score = run_inference_sam_predictor(image_path, box, multimask=False)
        logger.debug("SAM multimask fallback to predictor single mask for box %s", box)
        return [{"mask": mask, "score": score}]
    try:
        image_np = np.array(Image.open(image_path).convert("RGB"))
        x0, y0, x1, y1 = [int(v) for v in box]
        h, w = image_np.shape[:2]
        x0 = max(0, min(w - 1, x0))
        x1 = max(0, min(w, x1))
        y0 = max(0, min(h - 1, y0))
        y1 = max(0, min(h, y1))
        if x1 <= x0 or y1 <= y0:
            logger.debug("SAM multimask: invalid box after clipping %s", box)
            return []

        # For HF SAM3 backend, box prompting via generic pipeline may be unstable.
        # Use crop-local auto generation as primary path and remap to full image coords.
        if _is_sam3_backend() and not _is_sam3_native_backend() and MODEL_STORE.get("sam_automatic_generator") is not None:
            try:
                mag = MODEL_STORE.get("sam_automatic_generator")
                crop_np = image_np[y0:y1, x0:x1]
                raw = mag.generate(crop_np)
                remapped: List[Dict[str, Any]] = []
                for r in raw[: max(1, max_masks * 2)]:
                    m = r.get("segmentation")
                    if m is None:
                        continue
                    m_crop = (np.array(m) > 0).astype(np.uint8)
                    if m_crop.size == 0 or int(m_crop.sum()) == 0:
                        continue
                    if m_crop.shape != (y1 - y0, x1 - x0):
                        m_crop = np.array(
                            Image.fromarray(m_crop * 255).resize((x1 - x0, y1 - y0), Image.NEAREST)
                        )
                        m_crop = (m_crop > 0).astype(np.uint8)
                    full = np.zeros((h, w), dtype=np.uint8)
                    full[y0:y1, x0:x1] = m_crop
                    score = float(r.get("predicted_iou", r.get("score", 0.0)) or 0.0)
                    remapped.append({"mask": full, "score": score})
                if remapped:
                    remapped = sorted(remapped, key=lambda x: -float(x.get("score", 0.0)))[:max_masks]
                    logger.info(
                        "SAM3 crop-multimask produced %d candidates for box %s (image=%s).",
                        len(remapped),
                        box,
                        image_path,
                    )
                    return remapped
            except Exception as e:
                logger.debug("SAM3 crop-multimask failed for box %s: %s", box, e)

        predictor.set_image(image_np)
        masks, scores, logits = predictor.predict(
            box=np.array([x0, y0, x1, y1]),
            multimask_output=True,
        )
        results: List[Dict[str, Any]] = []

        def _clip_mask_to_box(mask_arr: np.ndarray) -> np.ndarray:
            m = (np.array(mask_arr) > 0).astype(np.uint8)
            if m.shape != (h, w):
                return np.zeros((h, w), dtype=np.uint8)
            out = np.zeros_like(m, dtype=np.uint8)
            out[y0:y1, x0:x1] = m[y0:y1, x0:x1]
            return out

        seen_signatures: set[tuple[int, int, int, int]] = set()
        if isinstance(masks, np.ndarray) and masks.ndim == 3:
            for i in range(min(len(masks), max_masks)):
                m = _clip_mask_to_box(masks[i])
                area = int(m.sum())
                if area == 0:
                    continue
                ys, xs = np.where(m > 0)
                sig = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
                if sig in seen_signatures:
                    continue
                seen_signatures.add(sig)
                s = float(scores[i]) if (scores is not None and len(scores) > i) else 1.0
                results.append({"mask": m, "score": s})
            logger.info("SAM multimask produced %d candidates for box %s (image=%s).", len(results), box, image_path)
            logger.debug("SAM multimask scores: %s", scores.tolist() if hasattr(scores, "tolist") else scores)
        else:
            m = _clip_mask_to_box(np.array(masks))
            if int(m.sum()) > 0:
                s = float(scores) if scores is not None else 1.0
                results.append({"mask": m, "score": s})
                logger.info("SAM multimask returned single mask for box %s (score=%.3f)", box, s)
            else:
                logger.info("SAM multimask single mask empty after clipping for box %s", box)
        if not results:
            # Fallback path when HF SAM3 collapses outside bbox:
            # 1) try edge-based mask constrained by bbox, 2) fallback to bbox rectangle.
            try:
                edge_mask = extract_thin_mask_edges(image_path, [x0, y0, x1, y1])
                if int(edge_mask.sum()) > 0:
                    results.append({"mask": edge_mask.astype(np.uint8), "score": 0.15})
                    logger.warning(
                        "SAM multimask produced 0 usable masks for box %s -> fallback edge mask used",
                        box,
                    )
                else:
                    raise RuntimeError("edge mask empty")
            except Exception:
                fallback = np.zeros((h, w), dtype=np.uint8)
                fallback[y0:y1, x0:x1] = 1
                results.append({"mask": fallback, "score": 0.05})
                logger.warning(
                    "SAM multimask produced 0 usable masks for box %s -> fallback bbox mask used",
                    box,
                )
        return results
    except Exception as e:
        logger.exception("SAM multimask failed: %s", e)
        mask, score = run_inference_sam_predictor(image_path, box, multimask=False)
        return [{"mask": mask, "score": score}]


def extract_thin_mask_edges(
    image_path: str,
    box: List[int],
    edge_thresh1: int = 50,
    edge_thresh2: int = 150,
) -> np.ndarray:
    """Return a binary mask (uint8) constructed from Canny edges + morphology inside the box."""
    if cv2 is None:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        m = np.zeros((h, w), dtype=np.uint8)
        x0, y0, x1, y1 = box
        x0, x1 = max(0, int(x0)), min(w, int(x1))
        y0, y1 = max(0, int(y0)), min(h, int(y1))
        m[y0:y1, x0:x1] = 1
        return m

    img = (
        cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if isinstance(image_path, (str, Path))
        else None
    )
    if img is None:
        pil = Image.open(image_path).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    x0, y0, x1, y1 = [int(v) for v in box]
    H, W = img.shape[:2]
    x0, x1 = max(0, x0), min(W - 1, x1)
    y0, y1 = max(0, y0), min(H - 1, y1)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((H, W), dtype=np.uint8)

    crop = img[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    except Exception:
        pass

    edges = cv2.Canny(gray, edge_thresh1, edge_thresh2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_closed = cv2.dilate(edges, kernel, iterations=2)
    edges_closed = cv2.morphologyEx(edges_closed, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_crop = np.zeros_like(gray, dtype=np.uint8)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5:
            continue
        cv2.drawContours(mask_crop, [cnt], -1, 255, thickness=cv2.FILLED)

    mask_crop = cv2.dilate(
        mask_crop, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 1)), iterations=1
    )
    mask_crop = (mask_crop > 0).astype(np.uint8)

    full_mask = np.zeros((H, W), dtype=np.uint8)
    full_mask[y0:y1, x0:x1] = mask_crop
    logger.debug("Extracted thin-edge mask for box %s (sum=%d)", box, int(full_mask.sum()))
    return full_mask


def compute_edge_alignment_score(image_path: str, mask: np.ndarray) -> float:
    if cv2 is None:
        return 0.0
    pil = Image.open(image_path).convert("RGB")
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    eroded = cv2.erode(mask_uint8, kernel, iterations=1)
    boundary = cv2.subtract(mask_uint8, eroded)
    if boundary.sum() == 0:
        inside_edges = (edges > 0) & (mask_uint8 > 0)
        total_edges = (edges > 0).sum()
        if total_edges == 0:
            return 0.0
        return float(inside_edges.sum()) / float(total_edges)
    else:
        boundary_bool = boundary > 0
        overlap = (edges > 0) & boundary_bool
        boundary_count = boundary_bool.sum()
        if boundary_count == 0:
            return 0.0
        return float(overlap.sum()) / float(boundary_count)


def run_text_guided_segmentation_native(
    image_path: str,
    text_prompts: List[str],
    score_threshold: float = 0.3,
) -> List[Dict[str, Any]]:
    """Text segmentation via official Meta SAM3 (set_text_prompt), without HF pipeline."""
    mag = MODEL_STORE.get("sam_automatic_generator")
    if mag is None:
        logger.warning("Official SAM3 generator not loaded.")
        return []

    try:
        image_np = np.array(Image.open(image_path).convert("RGB"))
    except Exception as e:
        logger.exception("Failed to open image for native SAM3: %s", e)
        return []

    results: List[Dict[str, Any]] = []
    for prompt in text_prompts:
        prompt_l = str(prompt).lower()
        is_rope_like = any(k in prompt_l for k in ["rope", "hawser", "cable", "wire"])
        try:
            proposals = mag.generate(image_np, text_prompt=str(prompt))
        except Exception as e:
            logger.exception("Official SAM3 text prompt failed for '%s': %s", prompt, e)
            proposals = []

        for prop in proposals:
            score = float(prop.get("score", 0.0))
            if score < score_threshold:
                continue
            mask = prop.get("mask") or prop.get("segmentation")
            bbox = prop.get("bbox")
            if mask is None or bbox is None:
                continue
            try:
                mask_pp = _postprocess_mask(mask, bbox, is_rope_like=is_rope_like)
            except Exception:
                mask_pp = (np.asarray(mask) > 0).astype(np.uint8)
            if int(mask_pp.sum()) == 0:
                continue
            results.append(
                {
                    "prompt": prompt,
                    "bbox": [int(v) for v in bbox],
                    "score": score,
                    "mask": mask_pp,
                    "label": prompt,
                    "meta": {"sam_backend": "native"},
                }
            )

    results = _dedupe_overlapping_masks(results, iou_thr=0.85)
    logger.info(
        "Official SAM3 text segmentation: %d masks for %s",
        len(results),
        Path(image_path).name,
    )
    return results


def run_text_guided_segmentation(
    image_path: str,
    text_prompts: List[str],
    score_threshold: float = 0.3,
    max_boxes_per_prompt: int = 10,
    sam_multimask_k: int = 3,
) -> List[Dict[str, Any]]:
    prompts = [str(p).strip() for p in (text_prompts or []) if str(p).strip()]
    if _is_sam3_native_backend() and MODEL_STORE.get("sam_automatic_generator") is not None:
        native_results = run_text_guided_segmentation_native(
            image_path,
            prompts,
            score_threshold=score_threshold,
        )
        if native_results:
            return native_results
        logger.warning(
            "Official SAM3 returned 0 masks for %s; falling back to GND+SAM path.",
            Path(image_path).name,
        )

    results: List[Dict[str, Any]] = []

    has_siglip = siglip_classifier_available()

    for prompt in prompts:
        prompt_l = str(prompt).lower()
        is_rope_like = any(k in prompt_l for k in ["rope", "hawser", "cable", "wire"])
        try:
            boxes = run_inference_grounding_dino(
                image_path, prompt, score_threshold, max_boxes_per_prompt
            )
        except Exception as e:
            logger.exception("Grounding for prompt '%s' failed: %s", prompt, e)
            boxes = []

        if not boxes:
            if MODEL_STORE.get("sam_automatic_generator") is not None:
                proposals = run_inference_sam_auto(
                    image_path,
                    max_masks=min(MAX_MASKS_PER_IMAGE, 8),
                    text_prompt=prompt,
                )
                if proposals:
                    best_prop = max(proposals, key=lambda x: float(x.get("score", 0.0)))
                    p_score = float(best_prop.get("score", 0.0))
                    if p_score >= max(0.15, score_threshold * 0.7):
                        p_mask = _postprocess_mask(best_prop["mask"], best_prop["bbox"], is_rope_like=is_rope_like)
                        if int(p_mask.sum()) > 0:
                            results.append(
                                {
                                    "prompt": prompt,
                                    "bbox": best_prop["bbox"],
                                    "score": p_score,
                                    "mask": p_mask,
                                    "label": prompt,
                                }
                            )
                logger.info("Prompt '%s': used SAM auto fallback (top-1) from %d proposals", prompt, len(proposals))
            else:
                logger.info("Prompt '%s': no boxes found and SAM auto not available", prompt)
            continue

        for b in boxes:
            bbox = [int(v) for v in b["bbox"]]
            gnd_score = float(b.get("score", 0.0))
            logger.debug("Processing bbox %s (gnd_score=%.3f) for prompt '%s'", bbox, gnd_score, prompt)
            sam_candidates = run_inference_sam_multimask(
                image_path, bbox, max_masks=sam_multimask_k
            )
            logger.debug("SAM candidates count=%d for bbox=%s", len(sam_candidates), bbox)

            best_mask = None
            best_score = -9999.0
            best_meta: Dict[str, Any] | None = None

            for cand in sam_candidates:
                mask = cand.get("mask")
                sam_score = float(cand.get("score", 0.0) or 0.0)
                siglip_score = 0.0
                if has_siglip:
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmpf:
                            tmp_path = tmpf.name
                        pil = Image.open(image_path).convert("RGB")
                        arr = np.array(pil)
                        m = (mask > 0).astype(np.uint8)
                        x0, y0, x1, y1 = bbox
                        x0, x1 = max(0, x0), min(arr.shape[1], x1)
                        y0, y1 = max(0, y0), min(arr.shape[0], y1)
                        crop = arr[y0:y1, x0:x1].copy()
                        if crop.size == 0:
                            raise RuntimeError("Empty crop")
                        mask_crop = (
                            m[y0:y1, x0:x1]
                            if m.shape == arr[..., 0].shape
                            else (m[y0:y1, x0:x1] if m.ndim == 2 else m)
                        )
                        if mask_crop.size == 0:
                            raise RuntimeError("Empty mask crop")
                        crop_masked = (crop * mask_crop[..., None]).astype(np.uint8)
                        Image.fromarray(crop_masked).save(tmp_path)
                        scores = run_inference_siglip_classify(tmp_path, [prompt])
                        siglip_score = float(scores[0]["score"]) if scores else 0.0
                        logger.debug(
                            "SigLIP2 score=%.3f for prompt '%s' bbox=%s",
                            siglip_score,
                            prompt,
                            bbox,
                        )
                    except Exception as e:
                        logger.debug("SigLIP2-check failed for candidate: %s", e)
                        siglip_score = 0.0
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass

                try:
                    edge_score = compute_edge_alignment_score(image_path, mask)
                    logger.debug("Edge alignment score=%.3f for bbox=%s", edge_score, bbox)
                except Exception:
                    edge_score = 0.0

                if is_rope_like:
                    combined = (
                        0.35 * sam_score + 0.25 * siglip_score + 0.3 * edge_score + 0.1 * gnd_score
                    )
                else:
                    combined = (
                        0.5 * sam_score + 0.35 * siglip_score + 0.15 * edge_score + 0.1 * gnd_score
                    )
                w_box = bbox[2] - bbox[0]
                h_box = bbox[3] - bbox[1]
                if h_box == 0:
                    ar = 1.0
                else:
                    ar = max(w_box / (h_box + 1e-6), h_box / (w_box + 1e-6))
                if ar > 3.0:
                    if is_rope_like:
                        combined += 0.3 * edge_score
                    else:
                        combined += 0.15 * edge_score

                logger.debug(
                    "Candidate combined score=%.4f (sam=%.3f siglip=%.3f edge=%.3f gnd=%.3f) for bbox=%s",
                    combined,
                    sam_score,
                    siglip_score,
                    edge_score,
                    gnd_score,
                    bbox,
                )

                if combined > best_score:
                    best_score = combined
                    best_mask = mask
                    best_meta = {
                        "sam_score": sam_score,
                        "siglip_score": siglip_score,
                        "clip_score": siglip_score,
                        "edge_score": edge_score,
                        "gnd_score": gnd_score,
                    }

            if best_mask is None or (
                best_meta
                and best_meta.get("sam_score", 0.0) < 0.2
                and best_meta.get("siglip_score", best_meta.get("clip_score", 0.0)) < 0.2
            ):
                thin_mask = extract_thin_mask_edges(image_path, bbox)
                thin_edge_score = compute_edge_alignment_score(image_path, thin_mask)
                edge_thr = 0.15 if is_rope_like else 0.2
                logger.debug("Thin-edge fallback score=%.3f for bbox=%s", thin_edge_score, bbox)
                if thin_edge_score > edge_thr and thin_mask.sum() > 0:
                    best_mask = thin_mask
                    best_score = max(best_score, 0.2 + 0.5 * thin_edge_score)
                    best_meta = {
                        "sam_score": 0.0,
                        "siglip_score": 0.0,
                        "clip_score": 0.0,
                        "edge_score": thin_edge_score,
                        "gnd_score": gnd_score,
                    }

            if best_mask is None:
                logger.debug("No mask selected for bbox=%s (prompt=%s)", bbox, prompt)
                continue

            try:
                best_mask = _postprocess_mask(best_mask, bbox, is_rope_like=is_rope_like)
            except Exception:
                best_mask = (best_mask > 0).astype(np.uint8)

            if int(best_mask.sum()) == 0:
                logger.debug("Postprocessed mask became empty for bbox=%s (prompt=%s)", bbox, prompt)
                continue

            results.append(
                {
                    "prompt": prompt,
                    "bbox": bbox,
                    "score": float(best_score),
                    "mask": best_mask,
                    "label": prompt,
                    "meta": best_meta or {},
                }
            )
            logger.info("Selected mask for prompt '%s' bbox=%s score=%.3f area=%d meta=%s",
                        prompt, bbox, float(best_score), int(best_mask.sum()), best_meta or {})

    results = _dedupe_overlapping_masks(results, iou_thr=0.85)
    logger.info("Text-guided segmentation yielded %d masks for %s", len(results), Path(image_path).name)
    return results
