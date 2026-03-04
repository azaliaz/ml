from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any
import os
import tempfile

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
)


def run_inference_grounding_dino_stub(
    image_path: str,
    text_prompt: str,
    score_threshold: float = 0.3,
    max_boxes: int = 10,
) -> List[Dict[str, Any]]:
    img = Image.open(image_path)
    w, h = img.size
    bbox = [int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75)]
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
            logger.warning("GroundingDINO returned zero boxes.")
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

        logger.info("GroundingDINO detected %d objects.", len(results))
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


def run_inference_clip_classify(image_path: str, labels: List[str]) -> List[Dict[str, Any]]:
    if MODEL_STORE.get("clip_model") is None or MODEL_STORE.get("clip_preprocess") is None:
        return [{"label": l, "score": 0.0} for l in labels]

    try:
        import torch

        model = MODEL_STORE["clip_model"]
        preprocess = MODEL_STORE["clip_preprocess"]
        device = MODEL_STORE["clip_device"] or ("cuda" if torch.cuda.is_available() else "cpu")

        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)

        from .models import CLIP_BACKEND  # local import to avoid cycle

        if CLIP_BACKEND == "open_clip":
            import open_clip  # type: ignore

            text_tokens = open_clip.tokenize(labels).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                text_features = model.encode_text(text_tokens)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                logits = (100.0 * image_features @ text_features.T).squeeze(0)
                probs = logits.softmax(dim=0).cpu().numpy().tolist()
        else:
            import clip  # type: ignore

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
        return mask, 1.0
    try:
        image_np = np.array(Image.open(image_path).convert("RGB"))
        predictor.set_image(image_np)
        x0, y0, x1, y1 = [int(v) for v in box]
        masks, scores, logits = predictor.predict(
            box=np.array([x0, y0, x1, y1]),
            multimask_output=multimask,
        )
        if isinstance(masks, np.ndarray) and masks.ndim == 3:
            best_idx = int(np.argmax(scores)) if len(scores) > 0 else 0
            chosen = masks[best_idx]
            score = float(scores[best_idx]) if len(scores) > 0 else 1.0
            return chosen.astype(np.uint8), float(score)
        else:
            mask = masks
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


def run_inference_sam_auto(image_path: str, max_masks: int = 30) -> List[Dict[str, Any]]:
    mag = MODEL_STORE.get("sam_automatic_generator")
    if mag is None:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        bbox = [int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75)]
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[bbox[1] : bbox[3], bbox[0] : bbox[2]] = 1
        return [{"mask": mask, "bbox": bbox, "score": 0.9, "area": int(mask.sum())}]
    try:
        image_np = np.array(Image.open(image_path).convert("RGB"))
    except Exception as e:
        logger.exception("Failed to open image for SAM auto: %s", e)
        return []

    try:
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
        return [{"mask": mask, "score": score}]
    try:
        image_np = np.array(Image.open(image_path).convert("RGB"))
        predictor.set_image(image_np)
        x0, y0, x1, y1 = [int(v) for v in box]
        masks, scores, logits = predictor.predict(
            box=np.array([x0, y0, x1, y1]),
            multimask_output=True,
        )
        results: List[Dict[str, Any]] = []
        if isinstance(masks, np.ndarray) and masks.ndim == 3:
            for i in range(min(len(masks), max_masks)):
                m = masks[i].astype(np.uint8)
                s = float(scores[i]) if (scores is not None and len(scores) > i) else 1.0
                results.append({"mask": m, "score": s})
        else:
            m = np.array(masks).astype(np.uint8)
            results.append({"mask": m, "score": 1.0})
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


def run_text_guided_segmentation(
    image_path: str,
    text_prompts: List[str],
    score_threshold: float = 0.3,
    max_boxes_per_prompt: int = 10,
    sam_multimask_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    For each text prompt:
      - run GroundingDINO to get boxes
      - for each box run SAM (multimask)
      - optionally run edge-based thin extraction
      - score masks by weighted sum (SAM_score, CLIP_score, edge_score)
      - choose best mask(s), post-process, return list of annotations with mask paths temporarily omitted
    Returns list of dicts: {"prompt":..., "bbox":[x0,y0,x1,y1], "score":..., "mask": np.ndarray, "label": chosen_label}
    """
    results: List[Dict[str, Any]] = []

    has_clip = MODEL_STORE.get("clip_model") is not None and MODEL_STORE.get("clip_preprocess") is not None

    for prompt in text_prompts:
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
                proposals = run_inference_sam_auto(image_path, max_masks=MAX_MASKS_PER_IMAGE)
                for p in proposals:
                    results.append(
                        {
                            "prompt": prompt,
                            "bbox": p["bbox"],
                            "score": float(p.get("score", 0.0)),
                            "mask": p["mask"],
                            "label": prompt,
                        }
                    )
            continue

        for b in boxes:
            bbox = [int(v) for v in b["bbox"]]
            gnd_score = float(b.get("score", 0.0))
            sam_candidates = run_inference_sam_multimask(
                image_path, bbox, max_masks=sam_multimask_k
            )
            best_mask = None
            best_score = -9999.0
            best_meta: Dict[str, Any] | None = None

            for cand in sam_candidates:
                mask = cand.get("mask")
                sam_score = float(cand.get("score", 0.0) or 0.0)
                clip_score = 0.0
                if has_clip:
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
                        scores = run_inference_clip_classify(tmp_path, [prompt])
                        clip_score = float(scores[0]["score"]) if scores else 0.0
                    except Exception as e:
                        logger.debug("CLIP-check failed for candidate: %s", e)
                        clip_score = 0.0
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass

                try:
                    edge_score = compute_edge_alignment_score(image_path, mask)
                except Exception:
                    edge_score = 0.0

                if is_rope_like:
                    combined = (
                        0.35 * sam_score + 0.25 * clip_score + 0.3 * edge_score + 0.1 * gnd_score
                    )
                else:
                    combined = (
                        0.5 * sam_score + 0.35 * clip_score + 0.15 * edge_score + 0.1 * gnd_score
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

                if combined > best_score:
                    best_score = combined
                    best_mask = mask
                    best_meta = {
                        "sam_score": sam_score,
                        "clip_score": clip_score,
                        "edge_score": edge_score,
                        "gnd_score": gnd_score,
                    }

            if best_mask is None or (
                best_meta
                and best_meta.get("sam_score", 0.0) < 0.2
                and best_meta.get("clip_score", 0.0) < 0.2
            ):
                thin_mask = extract_thin_mask_edges(image_path, bbox)
                thin_edge_score = compute_edge_alignment_score(image_path, thin_mask)
                edge_thr = 0.15 if is_rope_like else 0.2
                if thin_edge_score > edge_thr and thin_mask.sum() > 0:
                    best_mask = thin_mask
                    best_score = max(best_score, 0.2 + 0.5 * thin_edge_score)
                    best_meta = {
                        "sam_score": 0.0,
                        "clip_score": 0.0,
                        "edge_score": thin_edge_score,
                        "gnd_score": gnd_score,
                    }

            if best_mask is None:
                continue

            try:
                if cv2 is not None:
                    mask_u8 = (best_mask > 0).astype(np.uint8) * 255
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
                    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
                        mask_u8, connectivity=8
                    )
                    final_mask = np.zeros_like(mask_u8)
                    for i in range(1, nb_components):
                        area = stats[i, cv2.CC_STAT_AREA]
                        if area >= 20:
                            final_mask[output == i] = 255
                    best_mask = (final_mask > 0).astype(np.uint8)
                else:
                    best_mask = (best_mask > 0).astype(np.uint8)
            except Exception:
                best_mask = (best_mask > 0).astype(np.uint8)

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

    return results

