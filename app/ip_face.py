"""IP-Adapter (face/plus) and per-face detailer helpers for the Pony SDXL pipeline.

Volume layout (Modal ``ponyinf``):
    /my_vol/ipadapter/image_encoder/{config.json,model.safetensors}   # CLIP ViT-H
    /my_vol/ipadapter/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors
    /my_vol/ipadapter/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors
    /my_vol/detectors/face_yolov8m.pt                                  # bbox face detector
"""
from __future__ import annotations

import base64
import binascii
import re
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

IP_MODELS = {
    "plus_face": "ip-adapter-plus-face_sdxl_vit-h.safetensors",
    "plus": "ip-adapter-plus_sdxl_vit-h.safetensors",
}
STYLE_LORAS = {
    "preset_a": "example-lora.safetensors",
    "preset_b": "example-lora.safetensors",
    "realism": "pony-realism-slider.safetensors",
}
_DATA_URI = re.compile(r"^data:image/[\w+.-]+;base64,", re.IGNORECASE)

_loaded_ip: str | None = None
_detector = None


class FaceError(ValueError):
    pass


def decode_image(value: str) -> Image.Image:
    if not isinstance(value, str) or not value.strip():
        raise FaceError("image must be a non-empty base64 string")
    try:
        raw = base64.b64decode(_DATA_URI.sub("", value.strip()), validate=True)
        return Image.open(BytesIO(raw)).convert("RGB")
    except (binascii.Error, OSError) as exc:
        raise FaceError(f"Invalid base64 image: {exc}") from exc


def style_lora_list(spec) -> list[tuple[str, float]]:
    """``true`` -> all three at 1.0; dict -> given keys (null weight -> 1.0)."""
    if not spec:
        return []
    if spec is True:
        spec = {k: 1.0 for k in STYLE_LORAS}
    if not isinstance(spec, dict):
        raise FaceError("style_loras must be true or an object")
    out = []
    for key, weight in spec.items():
        if key not in STYLE_LORAS:
            raise FaceError(f"Unknown style lora '{key}'. Valid: {', '.join(STYLE_LORAS)}")
        out.append((STYLE_LORAS[key], 1.0 if weight is None else float(weight)))
    return out


def ensure_ip_adapter(pipe, ip_root: Path, model_key: str) -> None:
    global _loaded_ip
    if model_key not in IP_MODELS:
        raise FaceError(f"Unknown ip_adapter model '{model_key}'. Valid: {', '.join(IP_MODELS)}")
    if _loaded_ip == model_key:
        return
    if _loaded_ip is not None:
        pipe.unload_ip_adapter()
    weight = ip_root / "sdxl_models" / IP_MODELS[model_key]
    if not weight.is_file():
        raise FaceError(f"IP-Adapter weights missing on volume: {weight}")
    pipe.load_ip_adapter(
        str(ip_root),
        subfolder="sdxl_models",
        weight_name=IP_MODELS[model_key],
        image_encoder_folder=None,
    )
    if getattr(pipe, "image_encoder", None) is None:
        import torch
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

        enc = CLIPVisionModelWithProjection.from_pretrained(str(ip_root / "image_encoder"), torch_dtype=pipe.unet.dtype)
        pipe.register_modules(image_encoder=enc.to(pipe.unet.device), feature_extractor=CLIPImageProcessor())
    _loaded_ip = model_key


def disable_ip_adapter(pipe) -> None:
    global _loaded_ip
    if _loaded_ip is not None:
        pipe.unload_ip_adapter()
        _loaded_ip = None


def _detect_faces(image: Image.Image, detector_path: Path, confidence: float) -> list[tuple[int, int, int, int]]:
    global _detector
    if _detector is None:
        if not detector_path.is_file():
            raise FaceError(f"Face detector missing on volume: {detector_path}")
        from ultralytics import YOLO

        _detector = YOLO(str(detector_path))
    result = _detector.predict(image, conf=confidence, verbose=False)[0]
    boxes = [tuple(int(v) for v in b) for b in result.boxes.xyxy.tolist()]
    return sorted(boxes, key=lambda b: (b[0] + b[2]) / 2)  # left -> right


def face_detail(
    image: Image.Image,
    *,
    inpaint_pipe,
    ip_root: Path,
    detector_path: Path,
    faces: list[dict],
    prompt: str,
    negative_prompt: str,
    seed: int,
    denoise: float = 0.4,
    steps: int = 20,
    cfg: float = 5.0,
    padding: float = 1.8,
    confidence: float = 0.3,
    ip_model: str = "plus_face",
    generator_factory=None,
    on_face=None,
) -> tuple[Image.Image, int]:
    """Re-render each detected face (left->right) with its own IP-Adapter face ref."""
    boxes = _detect_faces(image, detector_path, confidence)
    if not boxes:
        return image, 0
    ensure_ip_adapter(inpaint_pipe, ip_root, ip_model)
    out = image.copy()
    W, H = out.size
    done = 0
    for i, box in enumerate(boxes):
        if i >= len(faces):
            break
        spec = faces[i] or {}
        ref = decode_image(spec.get("image", ""))
        weight = float(spec.get("weight", 0.4))
        x0, y0, x1, y1 = box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        side = int(max(x1 - x0, y1 - y0) * padding)
        side = max(64, min(side, W, H))
        left = int(min(max(cx - side / 2, 0), W - side))
        top = int(min(max(cy - side / 2, 0), H - side))
        crop = out.crop((left, top, left + side, top + side)).resize((1024, 1024), Image.LANCZOS)
        scale = 1024 / side
        mask = Image.new("L", (1024, 1024), 0)
        fx0, fy0 = (x0 - left) * scale, (y0 - top) * scale
        fx1, fy1 = (x1 - left) * scale, (y1 - top) * scale
        pad = 0.12 * (fx1 - fx0)
        ImageDraw.Draw(mask).ellipse((fx0 - pad, fy0 - pad, fx1 + pad, fy1 + pad), fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(24))
        inpaint_pipe.set_ip_adapter_scale(weight)
        if on_face:
            on_face(i)
        gen = generator_factory(seed + i) if generator_factory else None
        expr = (spec.get("prompt") or "").strip()
        face_prompt = f"{expr}, close-up face, {prompt}" if expr else prompt
        fixed = inpaint_pipe(
            prompt=face_prompt,
            negative_prompt=negative_prompt,
            image=crop,
            mask_image=mask,
            ip_adapter_image=ref,
            strength=denoise,
            num_inference_steps=steps,
            guidance_scale=cfg,
            width=1024,
            height=1024,
            generator=gen,
        ).images[0]
        fixed = fixed.resize((side, side), Image.LANCZOS)
        small_mask = mask.resize((side, side), Image.LANCZOS)
        out.paste(fixed, (left, top), small_mask)
        done += 1
    return out, done
