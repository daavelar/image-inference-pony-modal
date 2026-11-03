from __future__ import annotations

import sys
import warnings
from io import BytesIO
from pathlib import Path

import torch

if not hasattr(torch, "xpu"):
    class XPUStub:
        def is_available(self) -> bool:
            return False

        def empty_cache(self) -> None:
            pass

    torch.xpu = XPUStub()
    sys.modules["torch.xpu"] = torch.xpu

from diffusers import (  # noqa: E402
    StableDiffusionXLInpaintPipeline,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    StableDiffusionXLPipeline,
)

from app.config import Settings  # noqa: E402
from app import ip_face  # noqa: E402

warnings.filterwarnings("ignore", message=".*Already found a `peft_config` attribute.*")

SCHEDULERS = {
    "euler_a",
    "dpm++_sde_karras",
    "dpm++_2m_karras",
    "heun",
    "ddim",
    "lms",
}

_pipe: StableDiffusionXLPipeline | None = None
_current_model: str | None = None


class PipelineError(Exception):
    def __init__(self, message: str, *, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def validate_storage_paths(settings: Settings) -> None:
    if not settings.require_network_volume:
        return

    missing: list[str] = []
    if not settings.models_dir.is_dir():
        missing.append(str(settings.models_dir))
    if not settings.loras_dir.is_dir():
        missing.append(str(settings.loras_dir))
    if missing:
        raise RuntimeError(
            "Modal volume paths missing: "
            + ", ".join(missing)
            + ". Upload models with scripts/upload-models-to-modal-volume.py"
        )


def list_lora_files(settings: Settings) -> list[str]:
    if not settings.loras_dir.is_dir():
        return []
    return sorted(
        path.name
        for path in settings.loras_dir.iterdir()
        if path.is_file() and path.suffix == ".safetensors"
    )


def is_model_available(settings: Settings, model_name: str | None = None) -> bool:
    name = model_name or settings.default_model
    return (settings.models_dir / f"{name}.safetensors").is_file()


def get_scheduler(pipe_obj: StableDiffusionXLPipeline, scheduler_name: str):
    config = pipe_obj.scheduler.config
    if scheduler_name == "euler_a":
        return EulerAncestralDiscreteScheduler.from_config(config)
    if scheduler_name == "dpm++_sde_karras":
        return DPMSolverMultistepScheduler.from_config(
            config,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
        )
    if scheduler_name == "dpm++_2m_karras":
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)
    if scheduler_name == "heun":
        return HeunDiscreteScheduler.from_config(config)
    if scheduler_name == "ddim":
        return DDIMScheduler.from_config(config)
    if scheduler_name == "lms":
        return LMSDiscreteScheduler.from_config(config)
    return EulerAncestralDiscreteScheduler.from_config(config)


def load_model(settings: Settings, model_name: str) -> StableDiffusionXLPipeline:
    global _pipe, _current_model

    if _pipe is not None and _current_model == model_name:
        return _pipe

    model_path = settings.models_dir / f"{model_name}.safetensors"
    if not model_path.is_file():
        raise PipelineError(
            f"Model not found: {model_path}. Upload it to the Modal volume.",
            status_code=422,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[pipeline] loading model: {model_path}", flush=True)

    _pipe = StableDiffusionXLPipeline.from_single_file(
        str(model_path),
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True,
    )
    _pipe.to(device)

    try:
        _pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    _current_model = model_name
    print("[pipeline] model loaded", flush=True)
    return _pipe


def resolve_lora_path(settings: Settings, lora_path: str) -> Path:
    candidate = Path(lora_path)
    if candidate.is_file():
        return candidate.resolve()

    filename = Path(lora_path).name
    search_paths = [
        settings.loras_dir / filename,
        settings.loras_dir / lora_path,
    ]

    for path in search_paths:
        if path.is_file():
            return path.resolve()

    if not filename.endswith(".safetensors"):
        for path in (settings.loras_dir / f"{filename}.safetensors", settings.loras_dir / f"{lora_path}.safetensors"):
            if path.is_file():
                return path.resolve()

    raise PipelineError(f"Lora not found: {lora_path}", status_code=422)


def validate_loras(settings: Settings, loras: list[tuple[str, float]]) -> None:
    loras_dir = settings.loras_dir.resolve()
    for lora_path, _scale in loras:
        resolved = resolve_lora_path(settings, lora_path)
        if loras_dir not in resolved.parents and resolved.parent != loras_dir:
            raise PipelineError(f"Invalid lora path: {lora_path}", status_code=422)


def run_inference(
    settings: Settings,
    *,
    prompt: str,
    model: str,
    negative_prompt: str,
    scheduler_name: str,
    steps: int,
    cfg: float,
    width: int,
    height: int,
    seed: int,
    loras: list[tuple[str, float]],
    ip_adapter: dict | None = None,
    face_detailer: dict | None = None,
) -> tuple[bytes, int]:
    if scheduler_name not in SCHEDULERS:
        raise PipelineError(
            f"Unsupported scheduler: {scheduler_name}. Valid: {', '.join(sorted(SCHEDULERS))}",
            status_code=422,
        )

    validate_loras(settings, loras)
    pipe = load_model(settings, model)
    pipe.scheduler = get_scheduler(pipe, scheduler_name)

    adapter_names: list[str] = []
    adapter_weights: list[float] = []
    for index, (lora_path, lora_scale) in enumerate(loras):
        resolved = resolve_lora_path(settings, lora_path)
        adapter_name = f"adapter_{index}"
        try:
            pipe.load_lora_weights(str(resolved), adapter_name=adapter_name)
        except Exception as exc:
            raise PipelineError(f"Failed to load lora {resolved.name}: {exc}", status_code=422) from exc
        adapter_names.append(adapter_name)
        adapter_weights.append(lora_scale)

    if adapter_names:
        pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(seed)
    vol_root = (settings.models_dir or Path("/my_vol/models")).parent
    ip_root = vol_root / "ipadapter"

    try:
        extra = {}
        if ip_adapter:
            try:
                ref = ip_face.decode_image(ip_adapter.get("image", ""))
                ip_face.ensure_ip_adapter(pipe, ip_root, ip_adapter.get("model") or "plus_face")
            except ip_face.FaceError as exc:
                raise PipelineError(str(exc), status_code=422) from exc
            pipe.set_ip_adapter_scale(float(ip_adapter.get("weight", 0.4)))
            extra["ip_adapter_image"] = ref
        else:  # IP-Adapter left loaded by a previous request would demand image_embeds
            ip_face.disable_ip_adapter(pipe)
        base_image = (face_detailer or {}).get("base_image")
        with torch.inference_mode():
          if base_image:  # expression-variant mode: reuse a fixed frame, only faces change
            try:
                image = ip_face.decode_image(base_image)
            except ip_face.FaceError as exc:
                raise PipelineError(str(exc), status_code=422) from exc
          else:
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=cfg,
                width=width,
                height=height,
                generator=generator,
                **extra,
            ).images[0]
          if face_detailer:
                fd = dict(face_detailer)
                fd.pop("base_image", None)
                face_specs = fd.get("faces", []) or []
                # Per-face LoRA overrides: faces[i].loras = [{filename, weight}] (weight 0 disables a base LoRA).
                key = lambda f: Path(str(f)).name.removesuffix(".safetensors")
                name_by_file = {key(lp): adapter_names[i] for i, (lp, _w) in enumerate(loras)}
                base_w = dict(zip(adapter_names, adapter_weights))
                for spec in face_specs:
                    for item in (spec or {}).get("loras") or []:
                        fname = item.get("filename") or item.get("path")
                        if key(fname) not in name_by_file:
                            resolved = resolve_lora_path(settings, fname)
                            name = f"face_adapter_{len(name_by_file)}"
                            try:
                                pipe.load_lora_weights(str(resolved), adapter_name=name)
                            except Exception as exc:
                                raise PipelineError(f"Failed to load lora {resolved.name}: {exc}", status_code=422) from exc
                            name_by_file[key(fname)] = name
                            adapter_names.append(name)
                            base_w[name] = 0.0

                def on_face(i: int) -> None:
                    if not adapter_names:
                        return
                    w = dict(base_w)
                    spec = face_specs[i] if i < len(face_specs) else {}
                    for item in (spec or {}).get("loras") or []:
                        w[name_by_file[key(item.get("filename") or item.get("path"))]] = float(item.get("weight", 1.0))
                    pipe.set_adapters(list(w), adapter_weights=list(w.values()))

                inpaint = StableDiffusionXLInpaintPipeline(**pipe.components)
                try:
                    image, n = ip_face.face_detail(
                        image,
                        inpaint_pipe=inpaint,
                        ip_root=ip_root,
                        detector_path=vol_root / "detectors" / fd.pop("detector", "face_yolov8m.pt"),
                        faces=fd.pop("faces", []),
                        prompt=fd.pop("prompt", None) or prompt,
                        negative_prompt=negative_prompt,
                        seed=seed,
                        cfg=float(fd.pop("cfg", cfg)),
                        generator_factory=lambda s: torch.Generator(device=device).manual_seed(s),
                        on_face=on_face,
                        **{k: (str(v) if k == "ip_model" else float(v) if k != "steps" else int(v)) for k, v in fd.items()},
                    )
                except (ip_face.FaceError, TypeError) as exc:
                    raise PipelineError(f"face_detailer: {exc}", status_code=422) from exc
                print(f"[face_detailer] faces_detailed={n}")
    finally:
        if adapter_names:
            pipe.unload_lora_weights()

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return buffered.getvalue(), seed
