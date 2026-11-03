from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException

from app.config import Settings, get_settings
from app.inference import InferenceError, generate_image
from app.pony_pipeline import is_model_available, list_lora_files
from app.schemas import GenerateRequest, GenerateResponse, HealthResponse, LoraInfo, LoraSpec

app = FastAPI(title="Pony Inference API", version="1.0.0")


@app.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_ready=is_model_available(settings),
        loras_count=len(list_lora_files(settings)),
    )


@app.get("/loras", response_model=list[LoraInfo])
def list_loras(settings: Settings = Depends(get_settings)) -> list[LoraInfo]:
    return [LoraInfo(filename=name) for name in list_lora_files(settings)]


@app.post("/generate", response_model=GenerateResponse)
def generate(
    request: GenerateRequest,
    settings: Settings = Depends(get_settings),
) -> GenerateResponse:
    negative = request.negative or request.negative_prompt
    try:
        return generate_image(
            settings,
            prompt=request.prompt,
            model=request.model,
            negative_prompt=negative,
            scheduler=request.scheduler,
            steps=request.steps,
            cfg=request.cfg,
            width=request.width,
            height=request.height,
            seed=request.seed,
            loras=[LoraSpec(path=lora.path, scale=lora.scale) for lora in request.loras],
            style_loras=request.style_loras,
            ip_adapter=request.ip_adapter,
            face_detailer=request.face_detailer,
        )
    except InferenceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
