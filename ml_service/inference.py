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

from .models import MODEL_STORE, MAX_MASKS_PER_IMAGE, logger

# Ensure logger configured (user can set PREANN_DEBUG=1 to enable DEBUG)
logger = logging.getLogger("preann_service")

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


def run_inference_owlvit(
    image_path: str,
    text_prompt: str,
    score_threshold: float = 0.3,
    max_boxes: int = 10,
) -> List[Dict[str, Any]]:
    """
    Optional extra text-guided detector based on OWL-ViT / Owlv2.
    Returns list of dicts with same 'bbox' / 'score' / 'label' keys as GroundingDINO.
    """
    owl_model = MODEL_STORE.get("owl_model")
    owl_processor = MODEL_STORE.get("owl_processor")
    device = MODEL_STORE.get("owl_device")
    if owl_model is None or owl_processor is None or device is None:
        logger.debug("OWL-ViT not available in MODEL_STORE.")
        return []

    try:
        import torch
    except Exception:
        return []

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        logger.exception("OWL-ViT: failed to open image %s: %s", image_path, e)
        return []

    try:
        # text format: batch of lists of phrases
        inputs = owl_processor(
            text=[[text_prompt]],
            images=[image],
            return_tensors="pt",
        )
        # move tensors to device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            outputs = owl_model(**inputs)

        target_sizes = [image.size[::-1]]  # (height, width)
        postprocess = getattr(
            owl_processor,
            "post_process_object_detection",
            getattr(owl_processor, "post_process", None),
        )
        if postprocess is None:
            logger.warning("OWL-ViT processor has no post_process_object_detection")
            return []

        results = postprocess(
            outputs=outputs,
            target_sizes=target_sizes,
        )[0]

        boxes = results.get("boxes", None)
        scores = results.get("scores", None)
        if boxes is None or len(boxes) == 0:
            logger.info("OWL-ViT returned 0 boxes for prompt '%s'.", text_prompt)
            return []

        boxes = boxes.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy() if scores is not None else np.zeros((len(boxes),))
        # results["labels"] indexes into provided text queries; we have only one

        out: List[Dict[str, Any]] = []
        for box, score in zip(boxes, scores):
            score_f = float(score)
            if score_f < score_threshold:
                continue
            x0, y0, x1, y1 = [int(v) for v in box.tolist()]
            out.append(
                {
                    "bbox": [x0, y0, x1, y1],
                    "score": score_f,
                    "label": text_prompt,
                    "source": "owlvit",
                }
            )
        out = sorted(out, key=lambda x: -x["score"])[:max_boxes]
        logger.info("OWL-ViT detected %d boxes for prompt '%s'.", len(out), text_prompt)
        logger.debug("OWL-ViT boxes: %s", [o["bbox"] for o in out])
        return out
    except Exception as e:
        logger.exception("OWL-ViT inference crashed: %s", e)
        return []


def run_inference_clip_classify(image_path: str, labels: List[str]) -> List[Dict[str, Any]]:
    if MODEL_STORE.get("clip_model") is None or MODEL_STORE.get("clip_preprocess") is None:
        logger.debug("CLIP model or preprocess missing, returning zero scores.")
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

        logger.debug("CLIP classify for labels %s -> probs %s", labels, probs)
        results = [{"label": label, "score": float(prob)} for label, prob in zip(labels, probs)]
        results = sorted(results, key=lambda x: -x["score"])
        logger.info("CLIP classification top: %s", results[0] if results else None)
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


def run_inference_sam_auto(image_path: str, max_masks: int = 30) -> List[Dict[str, Any]]:
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
            logger.info("SAM multimask produced %d candidates for box %s (image=%s).", len(results), box, image_path)
            logger.debug("SAM multimask scores: %s", scores.tolist() if hasattr(scores, "tolist") else scores)
        else:
            m = np.array(masks).astype(np.uint8)
            s = float(scores) if scores is not None else 1.0
            results.append({"mask": m, "score": s})
            logger.info("SAM multimask returned single mask for box %s (score=%.3f)", box, s)
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

        try:
            owl_boxes = run_inference_owlvit(
                image_path, prompt, score_threshold, max_boxes_per_prompt
            )
        except Exception as e:
            logger.exception("OWL-ViT for prompt '%s' failed: %s", prompt, e)
            owl_boxes = []

        # Log counts right away
        logger.info("Prompt '%s': groundingdino returned %d boxes, owl returned %d boxes", prompt, len(boxes), len(owl_boxes))

        if boxes and owl_boxes:
            # объединяем и немного ограничиваем общее кол-во
            combined_boxes = boxes + owl_boxes
            combined_boxes = sorted(
                combined_boxes, key=lambda x: -float(x.get("score", 0.0))
            )[: max_boxes_per_prompt * 2]
            boxes = combined_boxes
            logger.debug("Combined boxes count after merging GND+OWL: %d", len(boxes))
        elif not boxes and owl_boxes:
            boxes = owl_boxes

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
                logger.info("Prompt '%s': used SAM auto proposals, added %d proposals", prompt, len(proposals))
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
                        logger.debug("CLIP score=%.3f for prompt '%s' bbox=%s", clip_score, prompt, bbox)
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
                    logger.debug("Edge alignment score=%.3f for bbox=%s", edge_score, bbox)
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

                logger.debug("Candidate combined score=%.4f (sam=%.3f clip=%.3f edge=%.3f gnd=%.3f) for bbox=%s",
                             combined, sam_score, clip_score, edge_score, gnd_score, bbox)

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
                logger.debug("Thin-edge fallback score=%.3f for bbox=%s", thin_edge_score, bbox)
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
                logger.debug("No mask selected for bbox=%s (prompt=%s)", bbox, prompt)
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
            logger.info("Selected mask for prompt '%s' bbox=%s score=%.3f area=%d meta=%s",
                        prompt, bbox, float(best_score), int(best_mask.sum()), best_meta or {})

    logger.info("Text-guided segmentation yielded %d masks for %s", len(results), Path(image_path).name)
    return results