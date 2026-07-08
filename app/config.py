from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_NEGATIVE_PROMPT = (
    "score_6, score_5, score_4, (worst quality, low quality:1.4), "
    "(cgpainting, cartoon, anime, 3d, render, sketch, drawing, painting, watercolor:1.2), "
    "(plastic skin:1.2), flat color, airbrushed, digital art"
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    aws_bucket: str
    aws_default_region: str = "auto"
    aws_endpoint: str
    aws_url: str
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_use_path_style_endpoint: bool = False

    network_volume_path: Path | None = None
    models_dir: Path | None = None
    loras_dir: Path | None = None
    require_network_volume: bool = False

    default_model: str = "untitledPony_b10"
    default_scheduler: str = "euler_a"
    default_steps: int = 30
    default_cfg: float = 5.0
    default_width: int = 1024
    default_height: int = 1024
    default_negative_prompt: str = DEFAULT_NEGATIVE_PROMPT

    api_host: str = "0.0.0.0"
    api_port: int = 8003
    cost_per_second: float = 0.000583  # Modal A100 40GB ($0.000583/s); override via COST_PER_SECOND

    @field_validator("aws_use_path_style_endpoint", "require_network_volume", mode="before")
    @classmethod
    def parse_bool_env(cls, value: object) -> object:
        if value is None or value == "":
            return False
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return value


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    if settings.network_volume_path is not None:
        vol = settings.network_volume_path.expanduser().resolve()
        return settings.model_copy(
            update={
                "models_dir": vol / "models",
                "loras_dir": vol / "loras",
            }
        )
    return settings.model_copy(
        update={
            "models_dir": (settings.models_dir or PROJECT_ROOT / "models").expanduser().resolve(),
            "loras_dir": (settings.loras_dir or PROJECT_ROOT / "loras").expanduser().resolve(),
        }
    )
