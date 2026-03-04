# models.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import logging
import os
import sys

logger = logging.getLogger("preann_service")

# Configure basic logging only if no handlers attached (keeps behavior stable on re-import)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)
# Allow enabling DEBUG via env var PREANN_DEBUG=1
if os.environ.get("PREANN_DEBUG", "") in ("1", "true", "True"):
    logger.setLevel(logging.DEBUG)

# Global feature flags
SAM_AVAILABLE = False
SAM_AUTO_AVAILABLE = False
GND_DINO_AVAILABLE = False
CLIP_AVAILABLE = False
CLIP_BACKEND: str | None = None

OWL_AVAILABLE = False

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
    "owl_model": None,
    "owl_processor": None,
    "owl_device": None,
}


def _maybe_add_local_package_to_syspath(pkg_name: str) -> bool:
    """Try to add a local checkout of a package to sys.path (for dev setups)."""
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


def load_groundingdino_if_available() -> None:
    """Best-effort loader for GroundingDINO."""
    global GND_DINO_AVAILABLE

    try:
        import groundingdino as _gd  # type: ignore  # noqa: F401

        GND_DINO_AVAILABLE = True
        logger.info("groundingdino import OK (system).")
    except Exception as e:
        logger.info("groundingdino import failed: %s", e)
        if _maybe_add_local_package_to_syspath("groundingdino"):
            try:
                import groundingdino as _gd  # type: ignore  # noqa: F401

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
        logger.warning(
            "GND_DINO_CHECKPOINT file not found: %s -> GroundingDINO disabled.", GND_DINO_CHECKPOINT
        )
        return

    config_path = GND_DINO_CONFIG or None
    default_config = Path("GrouningDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    if config_path is None and default_config.exists():
        config_path = str(default_config)

    try:
        from groundingdino.util import inference as gnd_inference  # type: ignore
    except Exception as e:
        logger.exception("Failed to import groundingdino.util.inference: %s", e)
        MODEL_STORE["gnd_model"] = None
        MODEL_STORE["gnd_inference_module"] = None
        GND_DINO_AVAILABLE = False
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
                        model_checkpoint_path=GND_DINO_CHECKPOINT,
                        device=device,
                    )
                elif "config" in params and "checkpoint" in params:
                    model_wrapper = load_fn(
                        config=config_path, checkpoint=GND_DINO_CHECKPOINT, device=device
                    )
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
                model_wrapper = ModelClass(
                    model_config_path=config_path,
                    model_checkpoint_path=GND_DINO_CHECKPOINT,
                    device=device,
                )
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
                        logger.exception(
                            "All attempts to instantiate GroundingDINO.Model failed: %s", e
                        )
                        model_wrapper = None
    except Exception as e:
        logger.exception("Exception during GroundingDINO loader: %s", e)
        model_wrapper = None

    MODEL_STORE["gnd_model"] = model_wrapper
    if model_wrapper is not None:
        logger.info("GroundingDINO model loaded and stored (device=%s).", device)
    else:
        logger.info("GroundingDINO model not loaded.")


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


def load_owlvit_if_available() -> None:
    """
    Optional loader for a text-guided detector (OWL-ViT or Owlv2).
    Used as дополнительный источник боксов для run_text_guided_segmentation.
    """
    global OWL_AVAILABLE

    try:
        # Prefer newer Owlv2, fall back to OwlViT if not available
        try:
            from transformers import Owlv2Processor, Owlv2ForObjectDetection  # type: ignore

            model_cls = Owlv2ForObjectDetection
            processor_cls = Owlv2Processor
            checkpoint = os.environ.get("OWL_CHECKPOINT", "google/owlv2-base-patch16")
            model_type = "owlv2"
        except Exception:
            from transformers import OwlViTProcessor, OwlViTForObjectDetection  # type: ignore

            model_cls = OwlViTForObjectDetection
            processor_cls = OwlViTProcessor
            checkpoint = os.environ.get("OWL_CHECKPOINT", "google/owlvit-base-patch32")
            model_type = "owlvit"
    except Exception as e:
        logger.info("transformers / OWL-ViT not available: %s", e)
        OWL_AVAILABLE = False
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
        logger.info("Attempting to load %s from checkpoint %s on %s", model_type, checkpoint, device)
        processor = processor_cls.from_pretrained(checkpoint)
        model = model_cls.from_pretrained(checkpoint)
        model.to(device)
        model.eval()
        MODEL_STORE["owl_model"] = model
        MODEL_STORE["owl_processor"] = processor
        MODEL_STORE["owl_device"] = device
        OWL_AVAILABLE = True
        logger.info("Loaded %s text detector from %s (device=%s).", model_type, checkpoint, device)
    except Exception as e:
        logger.exception("Failed to load OWL text detector: %s", e)
        MODEL_STORE["owl_model"] = None
        MODEL_STORE["owl_processor"] = None
        MODEL_STORE["owl_device"] = None
        OWL_AVAILABLE = False

load_sam_model_if_available()
load_groundingdino_if_available()
load_clip_if_available()
load_owlvit_if_available()