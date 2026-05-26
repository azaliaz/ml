# models.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import logging
import os
import sys
import inspect

logger = logging.getLogger("preann_service")

# Ensure preann_service logs are visible under any server (uvicorn/fastapi-cli/etc).
# Some runtimes install their own logging config and may not show non-uvicorn loggers
# unless they have an explicit handler.
if not logger.handlers:
    _h = logging.StreamHandler(stream=sys.stdout)
    _h.setLevel(logging.INFO)
    _h.setFormatter(logging.Formatter("%(levelname)s: %(name)s:%(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)
    logger.propagate = False
# Allow enabling DEBUG via env var PREANN_DEBUG=1
if os.environ.get("PREANN_DEBUG", "") in ("1", "true", "True"):
    logger.setLevel(logging.DEBUG)
    for _hh in list(logger.handlers):
        try:
            _hh.setLevel(logging.DEBUG)
        except Exception:
            pass

# Global feature flags
SAM_AVAILABLE = False
SAM_AUTO_AVAILABLE = False
GND_DINO_AVAILABLE = False
GND_DINO_CUDA_OPS = False
GND_DINO_LAST_ERROR: str = ""
CLIP_AVAILABLE = False
CLIP_BACKEND: str | None = None

SAM_CHECKPOINT = os.environ.get("SAM_CHECKPOINT", "").strip()
SAM_BACKEND = os.environ.get("SAM_BACKEND", "sam").strip().lower()
SAM3_HF_BACKENDS = {"sam3", "sam3_hf", "hf", "huggingface"}
SAM3_NATIVE_BACKENDS = {"sam3_native", "sam3-official", "native"}
SAM3_BACKENDS = SAM3_HF_BACKENDS | SAM3_NATIVE_BACKENDS
SAM3_MODEL_ID = os.environ.get("SAM3_MODEL_ID", "").strip()
SAM3_CHECKPOINT_PATH = os.environ.get("SAM3_CHECKPOINT_PATH", "").strip()
SAM3_HF_CKPT_VERSION = os.environ.get("SAM3_HF_CKPT_VERSION", "sam3").strip() or "sam3"
SAM3_CONFIDENCE_THRESHOLD = float(os.environ.get("SAM3_CONFIDENCE_THRESHOLD", "0.5"))
SAM3_INFERENCE_DTYPE = os.environ.get("SAM3_INFERENCE_DTYPE", "float32").strip().lower()
HUGGINGFACE_HUB_TOKEN = (
    os.environ.get("HUGGINGFACE_HUB_TOKEN", "").strip()
    or os.environ.get("HF_TOKEN", "").strip()
)
SAM3_TRUST_REMOTE_CODE = os.environ.get("SAM3_TRUST_REMOTE_CODE", "0").strip() in (
    "1",
    "true",
    "True",
)
GND_DINO_CHECKPOINT = os.environ.get("GND_DINO_CHECKPOINT", "").strip()
GND_DINO_CONFIG = os.environ.get("GND_DINO_CONFIG", "").strip()

MAX_MASKS_PER_IMAGE = int(os.environ.get("MAX_MASKS_PER_IMAGE", "30"))

MODEL_STORE: Dict[str, Any] = {
    "sam_model": None,
    "sam_predictor": None,
    "sam_automatic_generator": None,
    "sam_backend_mode": None,
    "gnd_model": None,
    "gnd_inference_module": None,
    "clip_model": None,
    "clip_preprocess": None,
    "clip_device": None,
}


def _masks_scores_boxes_from_sam3_state(state: Dict[str, Any]) -> tuple[Any, Any, Any]:
    import numpy as np
    import torch

    masks = state.get("masks")
    boxes = state.get("boxes")
    scores = state.get("scores")
    if masks is None:
        return None, None, None
    if isinstance(masks, torch.Tensor):
        masks_np = masks.detach().cpu().numpy()
    else:
        masks_np = np.asarray(masks)
    if isinstance(scores, torch.Tensor):
        scores_np = scores.detach().cpu().numpy()
    else:
        scores_np = np.asarray(scores) if scores is not None else np.array([])
    if isinstance(boxes, torch.Tensor):
        boxes_np = boxes.detach().cpu().numpy()
    else:
        boxes_np = np.asarray(boxes) if boxes is not None else np.array([])
    return masks_np, scores_np, boxes_np


def _iter_mask_rows(masks_np: Any, scores_np: Any, boxes_np: Any) -> list[tuple[Any, float, list[int] | None]]:
    import numpy as np

    if masks_np is None or masks_np.size == 0:
        return []
    if masks_np.ndim == 2:
        masks_np = masks_np[None, ...]
    n = int(masks_np.shape[0])
    rows: list[tuple[Any, float, list[int] | None]] = []
    for i in range(n):
        m = masks_np[i]
        while m.ndim > 2:
            m = m[0]
        score = float(scores_np[i]) if scores_np is not None and len(scores_np) > i else 1.0
        bbox = None
        if boxes_np is not None and len(boxes_np) > i:
            b = boxes_np[i].tolist() if hasattr(boxes_np[i], "tolist") else list(boxes_np[i])
            if len(b) >= 4:
                x0, y0, x1, y1 = [int(round(v)) for v in b[:4]]
                bbox = [x0, y0, x1, y1]
        rows.append((m, score, bbox))
    return rows


class _OfficialSam3PredictorAdapter:
    """Adapter for Meta SAM3 Sam3Processor (box / multimask via geometric prompts)."""

    def __init__(self, processor: Any):
        self._processor = processor
        self._image = None

    def set_image(self, image_np: Any) -> None:
        self._image = image_np

    def predict(
        self,
        box: Any,
        multimask_output: bool = False,
        point_coords: Any = None,
        point_labels: Any = None,
    ) -> tuple[Any, Any, Any]:
        import numpy as np
        from PIL import Image

        if self._image is None:
            raise RuntimeError("Image is not set. Call set_image() before predict().")

        image_pil = Image.fromarray(self._image).convert("RGB")
        h, w = self._image.shape[:2]
        state = self._processor.set_image(image_pil)

        x0, y0, x1, y1 = [int(v) for v in box]
        x0 = max(0, min(w - 1, x0))
        x1 = max(0, min(w, x1))
        y0 = max(0, min(h - 1, y0))
        y1 = max(0, min(h, y1))
        if x1 <= x0 or y1 <= y0:
            m = np.zeros((h, w), dtype=np.uint8)
            return m[None, ...], np.array([0.0], dtype=np.float32), np.zeros((1, h, w), dtype=np.float32)

        cx = ((x0 + x1) / 2.0) / float(w)
        cy = ((y0 + y1) / 2.0) / float(h)
        bw = (x1 - x0) / float(w)
        bh = (y1 - y0) / float(h)
        norm_box = [cx, cy, bw, bh]

        if point_coords is not None and point_labels is not None:
            logger.debug("Official SAM3: point prompts are not wired yet; using box only.")

        state = self._processor.add_geometric_prompt(box=norm_box, label=True, state=state)
        masks_np, scores_np, boxes_np = _masks_scores_boxes_from_sam3_state(state)
        rows = _iter_mask_rows(masks_np, scores_np, boxes_np)

        if not rows:
            m = np.zeros((h, w), dtype=np.uint8)
            return m[None, ...], np.array([0.0], dtype=np.float32), np.zeros((1, h, w), dtype=np.float32)

        masks_out = []
        scores_out = []
        for m, score, _ in rows:
            masks_out.append((np.asarray(m) > 0).astype(np.uint8))
            scores_out.append(score)

        if not multimask_output and len(masks_out) > 1:
            best_i = int(np.argmax(scores_out))
            masks_out = [masks_out[best_i]]
            scores_out = [scores_out[best_i]]

        masks_stack = np.stack(masks_out, axis=0)
        scores_arr = np.asarray(scores_out, dtype=np.float32)
        logits = np.zeros_like(masks_stack, dtype=np.float32)
        return masks_stack, scores_arr, logits


class _OfficialSam3AutoGeneratorAdapter:
    """Adapter for Meta SAM3 text-prompt segmentation."""

    def __init__(self, processor: Any):
        self._processor = processor

    def generate(self, image_np: Any, text_prompt: str | None = None) -> list[dict[str, Any]]:
        import numpy as np
        from PIL import Image

        if not text_prompt or not str(text_prompt).strip():
            return []

        image_pil = Image.fromarray(image_np).convert("RGB")
        state = self._processor.set_image(image_pil)
        state = self._processor.set_text_prompt(prompt=str(text_prompt).strip(), state=state)

        masks_np, scores_np, boxes_np = _masks_scores_boxes_from_sam3_state(state)
        rows = _iter_mask_rows(masks_np, scores_np, boxes_np)

        proposals: list[dict[str, Any]] = []
        for m, score, bbox in rows:
            mask_u8 = (np.asarray(m) > 0).astype(np.uint8)
            if mask_u8.size == 0 or int(mask_u8.sum()) == 0:
                continue
            if bbox is None:
                ys, xs = np.where(mask_u8 > 0)
                if len(xs) == 0:
                    continue
                bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            proposals.append(
                {
                    "segmentation": mask_u8,
                    "bbox": bbox,
                    "score": float(score),
                    "predicted_iou": float(score),
                    "area": int(mask_u8.sum()),
                }
            )
        proposals.sort(key=lambda x: -float(x.get("score", 0.0)))
        return proposals


class _HFSamPredictorAdapter:
    """Adapter that mimics SamPredictor API for HF mask-generation models."""

    def __init__(self, mask_pipe: Any):
        self._pipe = mask_pipe
        self._image = None

    def set_image(self, image_np: Any) -> None:
        self._image = image_np

    def _invoke(self, image_pil: Any, **kwargs: Any) -> Any:
        sig = inspect.signature(self._pipe.__call__)
        params = sig.parameters
        # If pipeline __call__ accepts **kwargs, pass all candidates through.
        # Filtering in this case drops box/prompt args and causes identical masks.
        has_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        if has_var_kwargs:
            return self._pipe(image_pil, **kwargs)
        accepted = set(params.keys())
        filtered = {k: v for k, v in kwargs.items() if k in accepted}
        return self._pipe(image_pil, **filtered)

    def predict(
        self,
        box: Any,
        multimask_output: bool = False,
        point_coords: Any = None,
        point_labels: Any = None,
    ) -> tuple[Any, Any, Any]:
        if self._image is None:
            raise RuntimeError("Image is not set. Call set_image() before predict().")
        from PIL import Image
        import numpy as np

        image_pil = Image.fromarray(self._image)
        x0, y0, x1, y1 = [int(v) for v in box]
        h, w = self._image.shape[:2]
        x0 = max(0, min(w - 1, x0))
        x1 = max(0, min(w, x1))
        y0 = max(0, min(h - 1, y0))
        y1 = max(0, min(h, y1))
        if x1 <= x0 or y1 <= y0:
            m = np.zeros((h, w), dtype=np.uint8)
            return m[None, ...], np.array([0.0], dtype=np.float32), np.zeros((1, h, w), dtype=np.float32)

        out = None
        last_err = None
        point_coords_list = None
        point_labels_list = None
        if point_coords is not None and point_labels is not None:
            try:
                point_coords_list = np.array(point_coords).astype(int).tolist()
                point_labels_list = np.array(point_labels).astype(int).tolist()
            except Exception:
                point_coords_list = None
                point_labels_list = None

        base_candidates = [
            {"input_boxes": [[[x0, y0, x1, y1]]], "multimask_output": multimask_output},
            {"boxes": [[x0, y0, x1, y1]], "multimask_output": multimask_output},
            {"bbox": [x0, y0, x1, y1], "multimask_output": multimask_output},
            {},
        ]
        candidate_kwargs = []
        if point_coords_list is not None and point_labels_list is not None:
            for kw in base_candidates:
                with_points = dict(kw)
                with_points.update(
                    {
                        "input_points": [point_coords_list],
                        "input_labels": [point_labels_list],
                        "points": [point_coords_list],
                        "labels": [point_labels_list],
                        "point_coords": point_coords_list,
                        "point_labels": point_labels_list,
                    }
                )
                candidate_kwargs.append(with_points)
        candidate_kwargs.extend(base_candidates)
        for kw in candidate_kwargs:
            try:
                out = self._invoke(image_pil, **kw)
                break
            except Exception as e:
                last_err = e
                continue
        if out is None:
            raise RuntimeError(f"SAM3 HF predictor call failed: {last_err}")

        masks, scores = _extract_masks_and_scores_from_hf_output(out)
        # Some HF SAM pipelines ignore box kwargs and return image-level masks.
        # If produced masks do not intersect with bbox at all, fallback to crop-based prediction.
        need_crop_fallback = False
        try:
            if masks.ndim == 2:
                masks_check = masks[None, ...]
            else:
                masks_check = masks
            intersects = False
            for m in masks_check:
                if m.shape[-2:] != (h, w):
                    continue
                if (m[y0:y1, x0:x1] > 0).any():
                    intersects = True
                    break
            if not intersects:
                need_crop_fallback = True
        except Exception:
            need_crop_fallback = False

        if need_crop_fallback:
            crop = self._image[y0:y1, x0:x1]
            crop_pil = Image.fromarray(crop)
            crop_out = None
            crop_last_err = None
            # First try prompting the crop with local box, then plain crop call.
            crop_w = max(1, x1 - x0)
            crop_h = max(1, y1 - y0)
            crop_kwargs = [
                {"input_boxes": [[[0, 0, crop_w, crop_h]]], "multimask_output": multimask_output},
                {"boxes": [[0, 0, crop_w, crop_h]], "multimask_output": multimask_output},
                {"bbox": [0, 0, crop_w, crop_h], "multimask_output": multimask_output},
                {},
            ]
            for kw in crop_kwargs:
                try:
                    crop_out = self._invoke(crop_pil, **kw)
                    break
                except Exception as e:
                    crop_last_err = e
                    continue
            if crop_out is None:
                raise RuntimeError(
                    f"SAM3 HF crop fallback failed for box {[x0, y0, x1, y1]}: {crop_last_err}"
                )
            crop_masks, crop_scores = _extract_masks_and_scores_from_hf_output(crop_out)
            if crop_masks.ndim == 2:
                crop_masks = crop_masks[None, ...]
            full_masks = np.zeros((crop_masks.shape[0], h, w), dtype=np.uint8)
            for i, cm in enumerate(crop_masks):
                cm_u8 = (np.array(cm) > 0).astype(np.uint8)
                if cm_u8.shape != (crop_h, crop_w):
                    cm_u8 = np.array(Image.fromarray(cm_u8 * 255).resize((crop_w, crop_h), Image.NEAREST))
                    cm_u8 = (cm_u8 > 0).astype(np.uint8)
                full_masks[i, y0:y1, x0:x1] = cm_u8
            masks = full_masks
            scores = crop_scores

        if masks.ndim == 2:
            masks = masks[None, ...]
        if scores is None or len(scores) == 0:
            scores = np.ones((masks.shape[0],), dtype=np.float32)
        logits = np.zeros_like(masks, dtype=np.float32)
        return masks.astype(np.uint8), scores.astype(np.float32), logits


class _HFSamAutoGeneratorAdapter:
    """Adapter that mimics SamAutomaticMaskGenerator.generate API."""

    def __init__(self, mask_pipe: Any):
        self._pipe = mask_pipe

    def _invoke(self, image_pil: Any, **kwargs: Any) -> Any:
        sig = inspect.signature(self._pipe.__call__)
        params = sig.parameters
        has_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        if has_var_kwargs:
            return self._pipe(image_pil, **kwargs)
        accepted = set(params.keys())
        filtered = {k: v for k, v in kwargs.items() if k in accepted}
        return self._pipe(image_pil, **filtered)

    def generate(self, image_np: Any, text_prompt: str | None = None) -> list[dict[str, Any]]:
        from PIL import Image
        import numpy as np

        image_pil = Image.fromarray(image_np)
        out = None
        last_err = None
        if text_prompt:
            candidate_kwargs = [
                {"text": text_prompt},
                {"prompt": text_prompt},
                {"labels": [text_prompt]},
                {"candidate_labels": [text_prompt]},
            ]
            for kw in candidate_kwargs:
                try:
                    out = self._invoke(image_pil, **kw)
                    break
                except Exception as e:
                    last_err = e
                    continue
        if out is None:
            try:
                out = self._invoke(image_pil)
            except Exception as e:
                raise RuntimeError(
                    f"SAM3 HF auto-generator call failed (text_prompt={text_prompt!r}): {e or last_err}"
                )
        masks, scores = _extract_masks_and_scores_from_hf_output(out)
        if masks.ndim == 2:
            masks = masks[None, ...]
        if scores is None or len(scores) == 0:
            scores = np.ones((masks.shape[0],), dtype=np.float32)

        proposals = []
        for idx in range(masks.shape[0]):
            m = (masks[idx] > 0).astype(np.uint8)
            ys, xs = np.where(m > 0)
            if len(xs) == 0 or len(ys) == 0:
                continue
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            proposals.append(
                {
                    "segmentation": m,
                    "bbox": [x0, y0, max(1, x1 - x0), max(1, y1 - y0)],
                    "predicted_iou": float(scores[idx]),
                    "area": int(m.sum()),
                }
            )
        return proposals


def _extract_masks_and_scores_from_hf_output(out: Any) -> tuple[Any, Any]:
    import numpy as np

    masks = None
    scores = None

    def _first_present(mapping: dict[str, Any], keys: list[str]) -> Any:
        for k in keys:
            if k in mapping and mapping.get(k) is not None:
                return mapping.get(k)
        return None

    if isinstance(out, dict):
        if "masks" in out:
            masks = out.get("masks")
        elif "mask" in out:
            masks = out.get("mask")
        elif "segmentation" in out:
            masks = out.get("segmentation")
        # Do not use `or` here: tensors cannot be cast to bool when they contain many values.
        scores = _first_present(out, ["scores", "iou_scores", "predicted_iou"])
    elif isinstance(out, list):
        if len(out) > 0 and isinstance(out[0], dict):
            mask_items = []
            score_items = []
            for item in out:
                m = item.get("mask", item.get("masks", item.get("segmentation", None)))
                if m is None:
                    continue
                if isinstance(m, list):
                    m = np.array(m, dtype=np.uint8)
                elif hasattr(m, "detach"):
                    m = m.detach().cpu().numpy()
                else:
                    m = np.array(m)
                if m.ndim == 3:
                    for mm in m:
                        mask_items.append((mm > 0).astype(np.uint8))
                        score_items.append(
                            float(item.get("score", item.get("iou_score", item.get("predicted_iou", 1.0))))
                        )
                else:
                    mask_items.append((m > 0).astype(np.uint8))
                    score_items.append(
                        float(item.get("score", item.get("iou_score", item.get("predicted_iou", 1.0))))
                    )
            if mask_items:
                masks = np.stack(mask_items, axis=0)
                scores = np.array(score_items, dtype=np.float32)
        else:
            masks = np.array(out)
    else:
        masks = np.array(out)

    if hasattr(masks, "detach"):
        masks = masks.detach().cpu().numpy()
    if masks is None:
        raise RuntimeError("HF output does not contain masks.")
    masks = np.array(masks)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0, ...]
    if scores is not None and hasattr(scores, "detach"):
        scores = scores.detach().cpu().numpy()
    if scores is not None:
        scores = np.array(scores).reshape(-1)
    return (masks > 0).astype(np.uint8), scores


def _ensure_official_sam3_importable() -> None:
    """
    Make `import sam3` work when the Meta repo is checked out at ml/sam3/ (editable or not).
    Set SAM3_LOCAL_DIR to override the repo root.
    """
    explicit = os.environ.get("SAM3_LOCAL_DIR", "").strip()
    if explicit:
        repo_root = Path(explicit).expanduser().resolve()
    else:
        repo_root = Path(__file__).resolve().parent.parent / "sam3"

    if not repo_root.is_dir():
        return

    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
        logger.info("Added official SAM3 repo root to sys.path: %s", repo_str)

    inner_pkg = repo_root / "sam3"
    if inner_pkg.is_dir():
        inner_str = str(inner_pkg)
        if inner_str not in sys.path:
            sys.path.insert(0, inner_str)
            logger.info("Added official SAM3 package dir to sys.path: %s", inner_str)


def _sam3_fp32_addmm_act(activation: Any, linear: Any, mat1: Any, _orig: Any) -> Any:
    import torch
    import torch.nn.functional as F

    if linear.weight.dtype == torch.float32:
        x = F.linear(mat1, linear.weight, linear.bias)
        if activation in (F.relu, torch.nn.ReLU):
            return F.relu(x)
        if activation in (F.gelu, torch.nn.GELU):
            return F.gelu(x, approximate="tanh")
        raise ValueError(f"Unexpected activation {activation}")
    return _orig(activation, linear, mat1)


def _patch_sam3_fused_ops_for_fp32() -> bool:
    """
    Meta SAM3 Mlp.fc1 uses perflib.addmm_act, which hardcodes bfloat16 matmuls.
    vitdet does `from sam3.perflib.fused import addmm_act` — must rebind after patch.
    """
    if SAM3_INFERENCE_DTYPE in ("bfloat16", "bf16"):
        return False
    try:
        from sam3.perflib import fused as sam3_fused
    except ImportError:
        logger.debug("sam3.perflib.fused not importable; skip addmm_act patch")
        return False

    if getattr(sam3_fused.addmm_act, "__name__", "") == "addmm_act_dtype_safe":
        return True

    _orig_addmm_act = sam3_fused.addmm_act

    def addmm_act_dtype_safe(activation: Any, linear: Any, mat1: Any) -> Any:
        return _sam3_fp32_addmm_act(activation, linear, mat1, _orig_addmm_act)

    sam3_fused.addmm_act = addmm_act_dtype_safe
    return True


def _rebind_sam3_vitdet_addmm_act() -> bool:
    """Rebind vitdet's imported addmm_act after fused module patch."""
    try:
        from sam3.perflib import fused as sam3_fused
        import sam3.model.vitdet as vitdet
    except ImportError:
        return False

    vitdet.addmm_act = sam3_fused.addmm_act
    return True


def _patch_sam3_vitdet_mlp_for_fp32() -> bool:
    """Replace vitdet.Mlp.forward so float32 weights skip hardcoded bf16 fused ops."""
    if SAM3_INFERENCE_DTYPE in ("bfloat16", "bf16"):
        return False
    try:
        import torch
        import sam3.model.vitdet as vitdet
    except ImportError:
        return False

    if getattr(vitdet.Mlp.forward, "__name__", "") == "mlp_forward_fp32_safe":
        return True

    _orig_forward = vitdet.Mlp.forward

    def mlp_forward_fp32_safe(self: Any, x: Any) -> Any:
        w = getattr(self.fc1, "weight", None)
        if w is not None and w.dtype == torch.float32:
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop1(x)
            x = self.norm(x)
            x = self.fc2(x)
            x = self.drop2(x)
            return x
        return _orig_forward(self, x)

    vitdet.Mlp.forward = mlp_forward_fp32_safe
    return True


def _apply_sam3_fp32_inference_patches(*, after_model_build: bool = False) -> None:
    fused_ok = _patch_sam3_fused_ops_for_fp32()
    rebind_ok = _rebind_sam3_vitdet_addmm_act() if after_model_build else False
    mlp_ok = _patch_sam3_vitdet_mlp_for_fp32() if after_model_build else False
    if fused_ok or rebind_ok or mlp_ok:
        logger.info(
            "SAM3 fp32 inference patches (after_build=%s): fused=%s vitdet_addmm=%s vitdet_mlp=%s",
            after_model_build,
            fused_ok,
            rebind_ok,
            mlp_ok,
        )


def _probe_groundingdino_cuda_ops() -> bool:
    try:
        from groundingdino import _C as gnd_c  # type: ignore

        return hasattr(gnd_c, "ms_deform_attn_forward")
    except Exception:
        return False


def _maybe_add_local_package_to_syspath(pkg_name: str) -> bool:
    """Try to add a local checkout of a package to sys.path (for dev setups)."""
    if pkg_name == "sam3":
        _ensure_official_sam3_importable()
        try:
            from importlib import import_module

            import_module("sam3")
            return True
        except Exception:
            pass

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


def load_sam_model_if_available() -> None:
    """Best-effort loader for SAM + optional SamAutomaticMaskGenerator."""
    global SAM_AVAILABLE, SAM_AUTO_AVAILABLE
    SAM_AVAILABLE = False
    SAM_AUTO_AVAILABLE = False
    logger.info(
        "SAM loader config: backend=%s sam3_model_id=%s checkpoint=%s",
        SAM_BACKEND,
        SAM3_MODEL_ID or "<empty>",
        SAM_CHECKPOINT or "<empty>",
    )

    if SAM_BACKEND in SAM3_NATIVE_BACKENDS:
        _load_sam3_native()
        return
    if SAM_BACKEND in SAM3_HF_BACKENDS:
        _load_sam3_from_huggingface()
        return

    from importlib import import_module

    try:
        import_module("segment_anything")  # type: ignore
        SAM_AVAILABLE = True
        logger.info("segment_anything import OK (system).")
    except Exception as e:
        logger.info("segment_anything import failed: %s", e)
        if _maybe_add_local_package_to_syspath("segment_anything"):
            try:
                import_module("segment_anything")  # type: ignore
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
        from segment_anything import sam_model_registry, SamPredictor  # type: ignore
        try:
            from segment_anything import SamAutomaticMaskGenerator  # type: ignore

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
                mag = SamAutomaticMaskGenerator(sam)  # type: ignore
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


def _load_sam3_native() -> None:
    """Load official Meta SAM3 (sam3 package + Sam3Processor)."""
    global SAM_AVAILABLE, SAM_AUTO_AVAILABLE

    if HUGGINGFACE_HUB_TOKEN:
        os.environ.setdefault("HF_TOKEN", HUGGINGFACE_HUB_TOKEN)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", HUGGINGFACE_HUB_TOKEN)

    _ensure_official_sam3_importable()
    _apply_sam3_fp32_inference_patches(after_model_build=False)

    try:
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = SAM3_CHECKPOINT_PATH or None
        model = build_sam3_image_model(
            device=device,
            checkpoint_path=ckpt,
            load_from_HF=ckpt is None,
        )
        _apply_sam3_fp32_inference_patches(after_model_build=True)
        # Sam3Processor feeds float32 images; unify model weights.
        if SAM3_INFERENCE_DTYPE in ("float32", "fp32", "f32"):
            model = model.float()
        elif SAM3_INFERENCE_DTYPE in ("bfloat16", "bf16") and device == "cuda":
            model = model.to(dtype=torch.bfloat16)
        else:
            model = model.float()
        processor = Sam3Processor(
            model,
            device=device,
            confidence_threshold=SAM3_CONFIDENCE_THRESHOLD,
        )
        MODEL_STORE["sam_model"] = model
        MODEL_STORE["sam_predictor"] = _OfficialSam3PredictorAdapter(processor)
        MODEL_STORE["sam_automatic_generator"] = _OfficialSam3AutoGeneratorAdapter(processor)
        MODEL_STORE["sam_backend_mode"] = "native"
        SAM_AVAILABLE = True
        SAM_AUTO_AVAILABLE = True
        logger.info(
            "Official SAM3 loaded (Sam3Processor) device=%s dtype=%s ckpt=%s conf=%.2f",
            device,
            SAM3_INFERENCE_DTYPE,
            ckpt or f"hf:{SAM3_HF_CKPT_VERSION}",
            SAM3_CONFIDENCE_THRESHOLD,
        )
    except Exception as e:
        logger.exception("Failed to load official SAM3: %s", e)
        MODEL_STORE["sam_model"] = None
        MODEL_STORE["sam_predictor"] = None
        MODEL_STORE["sam_automatic_generator"] = None
        MODEL_STORE["sam_backend_mode"] = None
        SAM_AVAILABLE = False
        SAM_AUTO_AVAILABLE = False


def _load_sam3_from_huggingface() -> None:
    """Load SAM3 from Hugging Face transformers pipeline (legacy adapter)."""
    global SAM_AVAILABLE, SAM_AUTO_AVAILABLE

    if not SAM3_MODEL_ID:
        logger.warning("SAM_BACKEND=%s but SAM3_MODEL_ID is empty -> SAM disabled.", SAM_BACKEND)
        return

    try:
        import torch
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        kwargs: Dict[str, Any] = {
            "task": "mask-generation",
            "model": SAM3_MODEL_ID,
            "device": device,
        }
        if HUGGINGFACE_HUB_TOKEN:
            kwargs["token"] = HUGGINGFACE_HUB_TOKEN
        if SAM3_TRUST_REMOTE_CODE:
            kwargs["trust_remote_code"] = True
        mask_pipe = pipeline(**kwargs)

        MODEL_STORE["sam_model"] = mask_pipe.model if hasattr(mask_pipe, "model") else mask_pipe
        MODEL_STORE["sam_predictor"] = _HFSamPredictorAdapter(mask_pipe)
        MODEL_STORE["sam_automatic_generator"] = _HFSamAutoGeneratorAdapter(mask_pipe)
        MODEL_STORE["sam_backend_mode"] = "hf"
        SAM_AVAILABLE = True
        SAM_AUTO_AVAILABLE = True
        logger.info(
            "SAM3 loaded from Hugging Face: model_id=%s device=%s", SAM3_MODEL_ID, "cuda" if device == 0 else "cpu"
        )
    except Exception as e:
        logger.exception("Failed to load SAM3 from Hugging Face (model_id=%s): %s", SAM3_MODEL_ID, e)
        MODEL_STORE["sam_model"] = None
        MODEL_STORE["sam_predictor"] = None
        MODEL_STORE["sam_automatic_generator"] = None
        SAM_AVAILABLE = False
        SAM_AUTO_AVAILABLE = False


def load_groundingdino_if_available() -> None:
    """Best-effort loader for GroundingDINO."""
    global GND_DINO_AVAILABLE, GND_DINO_CUDA_OPS, GND_DINO_LAST_ERROR
    GND_DINO_LAST_ERROR = ""
    GND_DINO_CUDA_OPS = False
    project_root = Path(__file__).resolve().parent.parent

    try:
        import groundingdino as _gd  # type: ignore  # noqa: F401

        GND_DINO_AVAILABLE = True
        logger.info("groundingdino import OK (system).")
    except Exception as e:
        logger.info("groundingdino import failed: %s", e)
        GND_DINO_LAST_ERROR = f"groundingdino import failed: {e}"
        if _maybe_add_local_package_to_syspath("groundingdino"):
            try:
                import groundingdino as _gd  # type: ignore  # noqa: F401

                GND_DINO_AVAILABLE = True
                logger.info("groundingdino import OK (local).")
            except Exception as e2:
                logger.warning("groundingdino import still failed after adding local path: %s", e2)
                GND_DINO_AVAILABLE = False
                GND_DINO_LAST_ERROR = (
                    f"groundingdino import failed after local path: {e2}"
                )
        else:
            GND_DINO_AVAILABLE = False

    if not GND_DINO_AVAILABLE:
        logger.info("GroundingDINO package not available -> GroundingDINO disabled.")
        if not GND_DINO_LAST_ERROR:
            GND_DINO_LAST_ERROR = "GroundingDINO package not available"
        return

    if not GND_DINO_CHECKPOINT:
        logger.info("GND_DINO_CHECKPOINT not set -> GroundingDINO disabled.")
        GND_DINO_LAST_ERROR = "GND_DINO_CHECKPOINT is empty"
        return

    ck_path = Path(GND_DINO_CHECKPOINT)
    if not ck_path.is_absolute():
        ck_path = (project_root / ck_path).resolve()
    if not ck_path.exists():
        logger.warning(
            "GND_DINO_CHECKPOINT file not found: %s -> GroundingDINO disabled.", str(ck_path)
        )
        GND_DINO_LAST_ERROR = f"GND_DINO_CHECKPOINT file not found: {str(ck_path)}"
        return

    config_path = GND_DINO_CONFIG or None
    default_config = project_root / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    if config_path is None and default_config.exists():
        config_path = str(default_config)
    if config_path:
        cfg_path = Path(config_path)
        if not cfg_path.is_absolute():
            cfg_path = (project_root / cfg_path).resolve()
        config_path = str(cfg_path)
    if not config_path or not Path(config_path).exists():
        GND_DINO_LAST_ERROR = f"GroundingDINO config file not found: {config_path}"
        logger.warning("GroundingDINO config file not found: %s", config_path)
        return

    # Compatibility patch for newer transformers versions where BertModel
    # may not expose get_head_mask directly (GroundingDINO expects it).
    try:
        from transformers.modeling_utils import ModuleUtilsMixin  # type: ignore
        from transformers.models.bert.modeling_bert import BertModel  # type: ignore

        if not hasattr(BertModel, "get_head_mask"):
            BertModel.get_head_mask = ModuleUtilsMixin.get_head_mask  # type: ignore[attr-defined]
            logger.info("Applied transformers compatibility patch: BertModel.get_head_mask")
    except Exception as e:
        logger.debug("Transformers compatibility patch skipped: %s", e)

    try:
        from groundingdino.util import inference as gnd_inference  # type: ignore
    except Exception as e:
        logger.exception("Failed to import groundingdino.util.inference: %s", e)
        MODEL_STORE["gnd_model"] = None
        MODEL_STORE["gnd_inference_module"] = None
        GND_DINO_AVAILABLE = False
        GND_DINO_LAST_ERROR = f"Failed to import groundingdino.util.inference: {e}"
        return

    MODEL_STORE["gnd_inference_module"] = gnd_inference

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
                    model_wrapper = load_fn(
                        model_config_path=config_path,
                        model_checkpoint_path=str(ck_path),
                        device=device,
                    )
                elif "config" in params and "checkpoint" in params:
                    model_wrapper = load_fn(
                        config=config_path, checkpoint=str(ck_path), device=device
                    )
                elif len(params) >= 2:
                    model_wrapper = load_fn(config_path, str(ck_path), device)
                elif len(params) == 1:
                    model_wrapper = load_fn(str(ck_path))
                else:
                    model_wrapper = load_fn()
                logger.info("GroundingDINO loaded via load_model.")
            except Exception as e:
                logger.debug("load_model attempts failed: %s", e)
                GND_DINO_LAST_ERROR = f"load_model attempts failed: {e}"
                model_wrapper = None
                # Do not continue with constructor fallbacks when the root cause
                # is a known transformers/runtime compatibility issue.
                if "get_head_mask" in str(e):
                    raise RuntimeError(str(e))

        if model_wrapper is None and hasattr(MODEL_STORE["gnd_inference_module"], "Model"):
            ModelClass = getattr(MODEL_STORE["gnd_inference_module"], "Model")
            try:
                model_wrapper = ModelClass(
                    model_config_path=config_path,
                    model_checkpoint_path=str(ck_path),
                    device=device,
                )
                logger.info("GroundingDINO loaded via Model(...) with keywords.")
            except Exception:
                try:
                    model_wrapper = ModelClass(config_path, str(ck_path), device)
                    logger.info("GroundingDINO loaded via Model(config, ckpt, device).")
                except Exception:
                    try:
                        model_wrapper = ModelClass(config_path, str(ck_path))
                        logger.info("GroundingDINO loaded via Model(config, ckpt).")
                    except Exception:
                        try:
                            model_wrapper = ModelClass(str(ck_path))
                            logger.info("GroundingDINO loaded via Model(checkpoint).")
                        except Exception as e:
                            logger.exception(
                                "All attempts to instantiate GroundingDINO.Model failed: %s", e
                            )
                            GND_DINO_LAST_ERROR = (
                                f"All attempts to instantiate GroundingDINO.Model failed: {e}"
                            )
                            model_wrapper = None
    except Exception as e:
        logger.exception("Exception during GroundingDINO loader: %s", e)
        GND_DINO_LAST_ERROR = f"Exception during GroundingDINO loader: {e}"
        model_wrapper = None

    MODEL_STORE["gnd_model"] = model_wrapper
    if model_wrapper is not None:
        GND_DINO_CUDA_OPS = _probe_groundingdino_cuda_ops()
        if not GND_DINO_CUDA_OPS:
            logger.warning(
                "GroundingDINO CUDA ops (_C) not built — inference will be skipped. "
                "Rebuild: cd GroundingDINO && pip install -e . --no-build-isolation"
            )
            GND_DINO_LAST_ERROR = "GroundingDINO CUDA extension _C not built"
        logger.info(
            "GroundingDINO model loaded (device=%s, cuda_ops=%s).",
            device,
            GND_DINO_CUDA_OPS,
        )
        GND_DINO_LAST_ERROR = "" if GND_DINO_CUDA_OPS else GND_DINO_LAST_ERROR
    else:
        logger.info("GroundingDINO model not loaded.")
        if not GND_DINO_LAST_ERROR:
            GND_DINO_LAST_ERROR = "GroundingDINO model not loaded (unknown reason)"


def load_clip_if_available(model_name: str = "ViT-B-32", pretrained: str = "openai") -> None:
    """Best-effort loader for CLIP / open_clip."""
    global CLIP_AVAILABLE, CLIP_BACKEND

    try:
        import open_clip  # type: ignore

        CLIP_BACKEND = "open_clip"
        CLIP_AVAILABLE = True
    except Exception:
        try:
            import clip  # type: ignore  # noqa: F401

            CLIP_BACKEND = "clip"
            CLIP_AVAILABLE = True
        except Exception:
            CLIP_AVAILABLE = False

    if not CLIP_AVAILABLE:
        logger.info("CLIP not available in environment.")
        return

    try:
        import torch
    except Exception:
        torch = None  # type: ignore[assignment]

    if torch is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"

    try:
        if CLIP_BACKEND == "open_clip":
            import open_clip  # type: ignore

            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            model.to(device)
            model.eval()
            MODEL_STORE["clip_model"] = model
            MODEL_STORE["clip_preprocess"] = preprocess
            MODEL_STORE["clip_device"] = device
            logger.info("open_clip loaded: %s (%s)", model_name, pretrained)
        else:
            import clip  # type: ignore

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


#
# Backward-compatibility guard:
# if this file is partially merged and SAM3 env vars are missing,
# export safe defaults so api import does not crash.
#
if "SAM_BACKEND" not in globals():
    SAM_BACKEND = os.environ.get("SAM_BACKEND", "sam").strip().lower()
if "SAM3_HF_BACKENDS" not in globals():
    SAM3_HF_BACKENDS = {"sam3", "sam3_hf", "hf", "huggingface"}
if "SAM3_NATIVE_BACKENDS" not in globals():
    SAM3_NATIVE_BACKENDS = {"sam3_native", "sam3-official", "native"}
if "SAM3_BACKENDS" not in globals():
    SAM3_BACKENDS = SAM3_HF_BACKENDS | SAM3_NATIVE_BACKENDS
if "SAM3_MODEL_ID" not in globals():
    SAM3_MODEL_ID = os.environ.get("SAM3_MODEL_ID", "").strip()
if "HUGGINGFACE_HUB_TOKEN" not in globals():
    HUGGINGFACE_HUB_TOKEN = (
        os.environ.get("HUGGINGFACE_HUB_TOKEN", "").strip()
        or os.environ.get("HF_TOKEN", "").strip()
    )
if "SAM3_TRUST_REMOTE_CODE" not in globals():
    SAM3_TRUST_REMOTE_CODE = os.environ.get("SAM3_TRUST_REMOTE_CODE", "0").strip() in (
        "1",
        "true",
        "True",
    )

load_sam_model_if_available()
load_groundingdino_if_available()
load_clip_if_available()