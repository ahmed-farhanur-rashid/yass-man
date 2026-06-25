"""
Runtime configuration loaded from .env via Pydantic Settings.

Model details live in model_config.yaml — not here.
This file only holds runtime settings: URLs, paths, feature flags, pipeline tuning.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── SearXNG ──────────────────────────────────────────────────────────────
    searxng_url: str = "https://searx.be"

    # ── Paths ─────────────────────────────────────────────────────────────────
    model_dir: Path = Path("./models")
    log_dir: Path = Path("./data/logs")
    model_config_path: Path = Path("./model_config.yaml")

    # ── Feature flags ─────────────────────────────────────────────────────────
    enable_llm: bool = True
    enable_clustering: bool = True
    enable_feedback: bool = True

    # ── Server ────────────────────────────────────────────────────────────────
    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = False

    @field_validator("model_dir", "log_dir", mode="after")
    @classmethod
    def ensure_dir_exists(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v

    @property
    def searxng_search_url(self) -> str:
        return f"{self.searxng_url.rstrip('/')}/search"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
