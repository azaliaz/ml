# qwen_llm
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

logger = logging.getLogger("preann_service")

QWEN_MODEL_ID = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
QWEN_MAX_NEW_TOKENS = int(os.environ.get("QWEN_MAX_NEW_TOKENS", "256"))

_QWEN_MODEL = None
_QWEN_PROCESSOR = None
_QWEN_LOADED = False


def _extract_json_object(text: str) -> Optional[dict]:
    text = (text or "").strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # убираем code fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)

    # берём первый JSON-объект из текста
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None

    chunk = m.group(0)
    try:
        obj = json.loads(chunk)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None

    return None


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


def ensure_qwen_loaded() -> bool:
    global _QWEN_LOADED, _QWEN_MODEL, _QWEN_PROCESSOR

    if _QWEN_LOADED:
        return _QWEN_MODEL is not None and _QWEN_PROCESSOR is not None

    _QWEN_LOADED = True

    try:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except Exception as e:
        logger.info("Qwen imports unavailable: %s", e)
        return False

    try:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        _QWEN_MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL_ID,
            torch_dtype="auto",
            device_map="auto",
        )
        _QWEN_PROCESSOR = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
        logger.info("Qwen loaded successfully: %s", QWEN_MODEL_ID)
        return True
    except Exception as e:
        logger.exception("Failed to load Qwen: %s", e)
        _QWEN_MODEL = None
        _QWEN_PROCESSOR = None
        return False


def qwen_suggest_prompts(
    image_path: str,
    user_instruction: str,
    class_names: Optional[List[str]] = None,
    task_type: str = "segmentation",
) -> Dict[str, Any]:
    """
    Returns:
        {
            "class_names": [...],
            "text_prompts": [...],
            "raw_text": "...",
            "source": "qwen" | "fallback"
        }
    """
    fallback_classes = class_names or ["object"]
    fallback_prompts = fallback_classes

    if not ensure_qwen_loaded():
        return {
            "class_names": fallback_classes,
            "text_prompts": fallback_prompts,
            "raw_text": "",
            "source": "fallback",
        }

    try:
        import torch
    except Exception as e:
        logger.warning("Torch unavailable for Qwen inference: %s", e)
        return {
            "class_names": fallback_classes,
            "text_prompts": fallback_prompts,
            "raw_text": "",
            "source": "fallback",
        }

    assert _QWEN_MODEL is not None
    assert _QWEN_PROCESSOR is not None

    image = Image.open(image_path).convert("RGB")
    instruction = (user_instruction or "").strip()
    if not instruction:
        instruction = (
            "Extract useful annotation classes and concise text prompts for CVAT preannotation."
        )

    system_prompt = (
        "You are helping prepare CVAT preannotation. "
        "Return ONLY valid JSON with keys: class_names, text_prompts. "
        "Do not add markdown, explanations, or extra keys."
    )

    existing = ", ".join(class_names or []) or "none"
    user_text = (
        f"Task type: {task_type}\n"
        f"Existing classes: {existing}\n"
        f"User instruction: {instruction}\n"
        f"Need concise classes and prompts suitable for object annotation."
    )

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ],
        },
    ]

    try:
        inputs = _QWEN_PROCESSOR.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(_QWEN_MODEL.device)

        with torch.no_grad():
            generated_ids = _QWEN_MODEL.generate(
                **inputs,
                max_new_tokens=QWEN_MAX_NEW_TOKENS,
                do_sample=False,
            )

        trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        raw_text = _QWEN_PROCESSOR.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        parsed = _extract_json_object(raw_text) or {}
        suggested_classes = _normalize_list(parsed.get("class_names"), fallback_classes)
        suggested_prompts = _normalize_list(parsed.get("text_prompts"), suggested_classes)

        return {
            "class_names": suggested_classes,
            "text_prompts": suggested_prompts,
            "raw_text": raw_text,
            "source": "qwen",
        }

    except Exception as e:
        logger.exception("Qwen prompt suggestion failed: %s", e)
        return {
            "class_names": fallback_classes,
            "text_prompts": fallback_prompts,
            "raw_text": "",
            "source": "fallback",
        }