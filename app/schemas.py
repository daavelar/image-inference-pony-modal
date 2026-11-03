from __future__ import annotations

from pydantic import BaseModel, Field


class LoraSpec(BaseModel):
    path: str
    scale: float = Field(default=1.0, ge=-20.0, le=20.0)  # sliders may exceed 1 / go negative


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    model: str | None = None
    negative: str | None = None
    negative_prompt: str | None = None
    scheduler: str | None = None
    steps: int | None = Field(default=None, ge=1, le=150)
    cfg: float | None = Field(default=None, ge=0.0, le=30.0)
    width: int | None = Field(default=None, ge=64, le=2048)
    height: int | None = Field(default=None, ge=64, le=2048)
    seed: int | None = None
    loras: list[LoraSpec] = Field(default_factory=list)
    style_loras: bool | dict[str, float | None] | None = None  # preset_a/preset_b/realism, default weight 1.0
    ip_adapter: dict | None = None  # {"image": b64, "weight": 0.4, "model": "plus_face"|"plus"}
    face_detailer: dict | None = None  # {"faces": [{"image": b64, "weight": 0.4}, ...] left->right, "denoise": 0.4}
    callback_url: str | None = None
    external_id: int | str | None = None


class GenerateResponse(BaseModel):
    url: str
    key: str
    seed: int
    duration_seconds: float
    cost: float


class LoraInfo(BaseModel):
    filename: str


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    loras_count: int
