from __future__ import annotations

import json
import logging
import random
import time

from app.config import Settings
from app.pony_pipeline import PipelineError, run_inference
from app.schemas import GenerateResponse, LoraSpec
from app.storage import upload_image

logger = logging.getLogger(__name__)


class InferenceError(Exception):
    def __init__(self, message: str, *, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def generate_image(
    settings: Settings,
    *,
    prompt: str,
    model: str | None = None,
    negative_prompt: str | None = None,
    scheduler: str | None = None,
    steps: int | None = None,
    cfg: float | None = None,
    width: int | None = None,
    height: int | None = None,
    seed: int | None = None,
    loras: list[LoraSpec] | None = None,
) -> GenerateResponse:
    normalized_loras = [(lora.path, lora.scale) for lora in (loras or [])]

    resolved_model = model or settings.default_model
    resolved_negative = negative_prompt or settings.default_negative_prompt
    resolved_scheduler = scheduler or settings.default_scheduler
    resolved_steps = steps if steps is not None else settings.default_steps
    resolved_cfg = cfg if cfg is not None else settings.default_cfg
    resolved_width = width if width is not None else settings.default_width
    resolved_height = height if height is not None else settings.default_height
    resolved_seed = seed if seed is not None and seed >= 0 else random.randint(0, 2**63 - 1)

    started_at = time.perf_counter()
    try:
        image_bytes, used_seed = run_inference(
            settings,
            prompt=prompt,
            model=resolved_model,
            negative_prompt=resolved_negative,
            scheduler_name=resolved_scheduler,
            steps=resolved_steps,
            cfg=resolved_cfg,
            width=resolved_width,
            height=resolved_height,
            seed=resolved_seed,
            loras=normalized_loras,
        )
    except PipelineError as exc:
        raise InferenceError(str(exc), status_code=exc.status_code) from exc

    try:
        url, key = upload_image(settings, image_bytes)
    except Exception as exc:
        raise InferenceError(f"S3 upload failed: {exc}", status_code=500) from exc

    duration_seconds = round(time.perf_counter() - started_at, 2)
    logger.info(
        "[inference] %s",
        json.dumps(
            {
                "model": resolved_model,
                "prompt_chars": len(prompt),
                "prompt_preview": prompt[:120],
                "scheduler": resolved_scheduler,
                "steps": resolved_steps,
                "cfg": resolved_cfg,
                "width": resolved_width,
                "height": resolved_height,
                "seed": used_seed,
                "loras": [{"path": path, "scale": scale} for path, scale in normalized_loras],
                "duration_seconds": duration_seconds,
                "key": key,
            },
            ensure_ascii=False,
        ),
    )
    return GenerateResponse(
        url=url,
        key=key,
        seed=used_seed,
        duration_seconds=duration_seconds,
        cost=0.0,
    )


def generate_from_payload(settings: Settings, payload: dict) -> dict:
    if not isinstance(payload, dict):
        return {"error": "JSON body must be an object"}

    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return {"error": "prompt (non-empty string) required"}

    loras_raw = payload.get("loras") or []
    if not isinstance(loras_raw, list):
        return {"error": "loras must be an array"}

    loras: list[LoraSpec] = []
    for item in loras_raw:
        if not isinstance(item, dict):
            return {"error": "each lora must be an object with path and optional scale"}
        path = item.get("path")
        if not isinstance(path, str) or not path.strip():
            return {"error": "each lora must include path (string)"}
        scale = item.get("scale", 1.0)
        try:
            loras.append(LoraSpec(path=path.strip(), scale=float(scale)))
        except Exception as exc:
            return {"error": f"invalid lora entry: {exc}"}

    model = payload.get("model")
    if model is not None and not isinstance(model, str):
        return {"error": "model must be a string"}

    negative = payload.get("negative") or payload.get("negative_prompt")
    if negative is not None and not isinstance(negative, str):
        return {"error": "negative must be a string"}

    scheduler = payload.get("scheduler")
    if scheduler is not None and not isinstance(scheduler, str):
        return {"error": "scheduler must be a string"}

    steps = payload.get("steps")
    if steps is not None:
        try:
            steps = int(steps)
        except (TypeError, ValueError):
            return {"error": "steps must be an integer"}

    cfg = payload.get("cfg")
    if cfg is not None:
        try:
            cfg = float(cfg)
        except (TypeError, ValueError):
            return {"error": "cfg must be a number"}

    width = payload.get("width")
    if width is not None:
        try:
            width = int(width)
        except (TypeError, ValueError):
            return {"error": "width must be an integer"}

    height = payload.get("height")
    if height is not None:
        try:
            height = int(height)
        except (TypeError, ValueError):
            return {"error": "height must be an integer"}

    seed = payload.get("seed")
    if seed is not None:
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            return {"error": "seed must be an integer"}

    try:
        result = generate_image(
            settings,
            prompt=prompt.strip(),
            model=model.strip() if isinstance(model, str) else None,
            negative_prompt=negative,
            scheduler=scheduler,
            steps=steps,
            cfg=cfg,
            width=width,
            height=height,
            seed=seed,
            loras=loras,
        )
        return result.model_dump()
    except InferenceError as exc:
        return {"error": str(exc)}
