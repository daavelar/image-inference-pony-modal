from __future__ import annotations

from pydantic import BaseModel, Field


class LoraSpec(BaseModel):
    path: str
    scale: float = Field(default=1.0, ge=0.0)


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
