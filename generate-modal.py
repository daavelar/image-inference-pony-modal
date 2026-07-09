#!/usr/bin/env python3
"""
Pony SDXL image inference on Modal — HTTP + GPU worker.

Job payload (flat JSON, no "input" wrapper):
    {"prompt", "model"?, "negative"?, "scheduler"?, "steps"?, "cfg"?, "width"?, "height"?, "seed"?, "loras"?}

Storage: mount a Modal Volume (default ``ponyinf``) with:
    <mount>/models/<model>.safetensors
    <mount>/loras/*.safetensors

Secrets (Modal Secret name from ``MODAL_AWS_SECRET``):
    AWS_BUCKET, AWS_ENDPOINT, AWS_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
    Optional: AWS_DEFAULT_REGION, AWS_USE_PATH_STYLE_ENDPOINT

GPU: default fallback chain ``A100`` → ``L40S`` → ``A10`` → ``L4`` (override with ``MODAL_GPU``).

Cost: each job returns ``cost`` (USD) estimated from container wall time using
Modal list prices for the detected GPU plus CPU/memory. Tune with
``MODAL_TAIL_IDLE_SECONDS``, ``MODAL_CPU_CORES``, ``MODAL_MEMORY_GIB``,
``MODAL_REGION_MULTIPLIER``.

Populate the volume once from your machine:
    python scripts/upload-models-to-modal-volume.py
    python scripts/upload-loras-to-modal-volume.py

Usage:
    export MODAL_AWS_SECRET=pony-r2
    modal deploy generate-modal.py
    modal run generate-modal.py --payload-json '{"prompt":"score_9, portrait, natural lighting"}'

HTTP (``pony_http_app`` — GET ``/health``, GET ``/loras``, POST ``/generate``):
    Default async: POST /generate returns 202 with ``call_id``; poll GET ``/result/{call_id}``.
    Sync: POST /generate?blocking=true

Docs: https://modal.com/docs/guide
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent

VOL_NAME = os.environ.get("MODAL_VOLUME_NAME", "ponyinf")
VOL_MOUNT = os.environ.get("MODAL_VOLUME_MOUNT", "/my_vol")
# Strongest first; Modal tries each type in order when capacity is limited.
_DEFAULT_GPU: list[str] = ["A100", "L40S", "A10", "L4"]
GPU_COST_PER_SECOND: dict[str, float] = {
    "A100": 0.000583,
    "L40S": 0.000542,
    "A10": 0.000306,
    "L4": 0.000222,
}


def _parse_modal_gpu(raw: str) -> str | list[str]:
    if not raw:
        return _DEFAULT_GPU
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return parts[0] if len(parts) == 1 else parts


def _primary_gpu_type(gpu: str | list[str]) -> str:
    primary = gpu[0] if isinstance(gpu, list) else gpu
    return primary.split(":")[0].strip().upper()


def _gpu_cost_per_second(gpu_type: str) -> float:
    key = gpu_type.upper().split(":")[0]
    if key.startswith("A100"):
        key = "A100"
    return GPU_COST_PER_SECOND.get(key, GPU_COST_PER_SECOND["L40S"])


MODAL_GPU_RAW = os.environ.get("MODAL_GPU", "").strip()
INFERENCE_GPU: str | list[str] = _parse_modal_gpu(MODAL_GPU_RAW)

_MOUNT_IGNORE = [
    ".git/**/*",
    ".venv/**/*",
    "venv/**/*",
    "__pycache__/**/*",
    "*.py[cod]",
    ".cursor/**/*",
    ".env",
    ".env.*",
    "models/**/*",
    "loras/**/*",
    "outputs/**/*",
]

_volume = modal.Volume.from_name(VOL_NAME, create_if_missing=True)

web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fastapi>=0.115.0", "httpx>=0.27.0")
    .env({"PYTHONPATH": "/app"})
    .add_local_dir(str(HERE / "app"), remote_path="/app/app")
)

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime",
    )
    .env(
        {
            "DEBIAN_FRONTEND": "noninteractive",
            "PYTHONUNBUFFERED": "1",
            "PIP_NO_CACHE_DIR": "1",
            "PIP_BREAK_SYSTEM_PACKAGES": "1",
        }
    )
    .apt_install(
        "git",
        "libgl1",
        "libglib2.0-0",
        "libgomp1",
    )
    .workdir("/app")
    .pip_install(
        "fastapi>=0.115.0",
        "pydantic-settings>=2.0.0",
        "python-dotenv>=1.0.0",
        "boto3>=1.34.0",
        "diffusers==0.29.2",
        "transformers==4.40.2",
        "accelerate==0.30.1",
        "safetensors==0.4.3",
        "peft==0.11.1",
        "Pillow==10.3.0",
        "xformers",
    )
    .add_local_dir(
        str(HERE),
        remote_path="/app",
        ignore=_MOUNT_IGNORE,
        copy=True,
    )
)


def _configure_modal_env() -> None:
    os.environ["NETWORK_VOLUME_PATH"] = VOL_MOUNT
    os.environ["MODELS_DIR"] = str(Path(VOL_MOUNT) / "models")
    os.environ["LORAS_DIR"] = str(Path(VOL_MOUNT) / "loras")
    os.environ["REQUIRE_NETWORK_VOLUME"] = "1"
    if not os.environ.get("COST_PER_SECOND", "").strip():
        from app.modal_billing import detect_gpu_cost_per_second

        os.environ["COST_PER_SECOND"] = str(detect_gpu_cost_per_second())

    from app.config import get_settings

    get_settings.cache_clear()


MODAL_AWS_SECRET_NAME = os.environ.get("MODAL_AWS_SECRET", "pony-r2").strip() or "pony-r2"
_r2_secret = modal.Secret.from_name(MODAL_AWS_SECRET_NAME)


def _list_loras_from_volume() -> list[dict]:
    loras_dir = Path(VOL_MOUNT) / "loras"
    if not loras_dir.is_dir():
        return []
    return [
        {"filename": path.name}
        for path in sorted(loras_dir.iterdir())
        if path.is_file() and path.suffix == ".safetensors"
    ]


def _build_fastapi_app():
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, Response

    api = FastAPI(title="Pony Inference", version="1.0")

    @api.get("/health")
    def health_get():
        models_dir = Path(VOL_MOUNT) / "models"
        default_model = os.environ.get("DEFAULT_MODEL", "untitledPony_b10")
        model_ready = (models_dir / f"{default_model}.safetensors").is_file()
        return JSONResponse(
            {
                "status": "ok",
                "model_ready": model_ready,
                "loras_count": len(_list_loras_from_volume()),
            }
        )

    @api.head("/health")
    def health_head():
        return Response(status_code=200)

    @api.get("/loras")
    def list_loras():
        return JSONResponse(_list_loras_from_volume())

    @api.get("/result/{call_id}")
    def get_job_result(call_id: str):
        fc = modal.FunctionCall.from_id(call_id)
        try:
            result = fc.get(timeout=0)
        except TimeoutError:
            return JSONResponse(
                {"status": "pending", "call_id": call_id},
                status_code=202,
            )
        except Exception as e:
            return JSONResponse(
                {"status": "failed", "call_id": call_id, "error": str(e)},
                status_code=500,
            )
        if result.get("error"):
            payload = dict(result)
            payload.setdefault("status", "failed")
            payload.setdefault("call_id", call_id)
            return JSONResponse(payload, status_code=400)
        payload = dict(result)
        payload.setdefault("status", "completed")
        payload.setdefault("call_id", call_id)
        return JSONResponse(payload, status_code=200)

    @api.post("/generate")
    def generate_http(body: dict, blocking: bool = False):
        if not isinstance(body, dict):
            return JSONResponse({"error": "JSON body must be an object"}, status_code=400)

        from app.callbacks import build_callback_payload, extract_callback_meta, post_callback

        try:
            job, callback_url, external_id = extract_callback_meta(body)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)

        worker = PonyWorker()

        if blocking:
            try:
                result = worker.run_job.remote(job)
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)
            if callback_url:
                payload = build_callback_payload("blocking", result, external_id)
                post_callback(callback_url, payload)
            status_code = 400 if result.get("error") else 200
            return JSONResponse(result, status_code=status_code)

        try:
            fc = worker.run_job.spawn(job)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)
        cid = fc.object_id
        if callback_url:
            deliver_callback.spawn(cid, callback_url, external_id)
        return JSONResponse(
            {
                "status": "pending",
                "call_id": cid,
                "task_id": cid,
                "result_url": f"/result/{cid}",
            },
            status_code=202,
        )

    return api


app = modal.App("pony-modal")


@app.cls(
    image=image,
    gpu=INFERENCE_GPU,
    volumes={VOL_MOUNT: _volume},
    secrets=[_r2_secret],
    timeout=60 * 30,
    scaledown_window=300,
)
class PonyWorker:
    @modal.enter()
    def bootstrap(self):
        from app.modal_billing import ContainerCostTracker, detect_gpu_cost_per_second, get_gpu_name

        self._cost_tracker = ContainerCostTracker.from_env(scaledown_window=300)
        boot_started = self._cost_tracker.on_boot_start()

        _configure_modal_env()
        from app.config import get_settings
        from app.pony_pipeline import validate_storage_paths

        gpu_name = get_gpu_name()
        cost_per_second = detect_gpu_cost_per_second()
        os.environ["COST_PER_SECOND"] = str(cost_per_second)

        self.settings = get_settings()
        validate_storage_paths(self.settings)
        self._cost_tracker.on_boot_complete(boot_started)
        print(
            f"[boot] gpu={gpu_name or 'unknown'} cost_per_second={cost_per_second} "
            f"bootstrap_seconds={self._cost_tracker._bootstrap_seconds:.1f}",
            flush=True,
        )

    @modal.method()
    def run_job(self, payload: dict) -> dict:
        from app.inference import generate_from_payload
        from app.modal_billing import run_with_billing

        return run_with_billing(
            self._cost_tracker,
            generate_from_payload,
            self.settings,
            payload,
        )


@app.function(
    image=web_image,
    volumes={VOL_MOUNT: _volume},
    secrets=[_r2_secret],
    timeout=60 * 30,
    scaledown_window=300,
)
def deliver_callback(call_id: str, callback_url: str, external_id: int | str | None = None):
    from app.callbacks import deliver_callback_for_call

    deliver_callback_for_call(call_id, callback_url, external_id)


@app.function(
    image=web_image,
    volumes={VOL_MOUNT: _volume},
    timeout=60 * 30,
    scaledown_window=300,
)
@modal.asgi_app(label="pony-http")
def pony_http_app():
    return _build_fastapi_app()


@app.local_entrypoint()
def main(payload_json: str | None = None):
    worker = PonyWorker()
    raw = payload_json or os.environ.get("MODAL_JOB_JSON", "").strip()
    if not raw:
        raise SystemExit(
            "Set MODAL_JOB_JSON or run with --payload-json, e.g. "
            '\'{"prompt":"score_9, portrait, natural lighting"}\''
        )
    payload = json.loads(raw)
    out = worker.run_job.remote(payload)
    print(json.dumps(out, indent=2, default=str))
