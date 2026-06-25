"""
Loads and validates model_config.yaml at startup.

This is the single source of truth for all model definitions and pipeline tuning.
No model names, paths, or repos should appear anywhere else in the codebase.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


# ── Sub-configs ───────────────────────────────────────────────────────────────


@dataclass
class EmbedderConfig:
    model: str
    source: str
    repo: str
    backend: str
    max_seq_length: int = 512
    batch_size: int = 32

    @classmethod
    def from_dict(cls, d: dict) -> "EmbedderConfig":
        _require_keys(d, ["model", "source", "repo", "backend"], section="embedder")
        return cls(
            model=d["model"],
            source=d["source"],
            repo=d["repo"],
            backend=d["backend"],
            max_seq_length=d.get("max_seq_length", 512),
            batch_size=d.get("batch_size", 32),
        )


@dataclass
class RerankerConfig:
    model: str
    source: str
    repo: str
    backend: str
    max_seq_length: int = 512
    top_k: int = 10

    @classmethod
    def from_dict(cls, d: dict) -> "RerankerConfig":
        _require_keys(d, ["model", "source", "repo", "backend"], section="reranker")
        return cls(
            model=d["model"],
            source=d["source"],
            repo=d["repo"],
            backend=d["backend"],
            max_seq_length=d.get("max_seq_length", 512),
            top_k=d.get("top_k", 10),
        )


@dataclass
class RouterConfig:
    mode: str = "rule-based"

    @classmethod
    def from_dict(cls, d: dict) -> "RouterConfig":
        return cls(mode=d.get("mode", "rule-based"))


@dataclass
class LLMConfig:
    enabled: bool
    model: str
    path: str
    download_url: Optional[str]
    context_length: int = 4096
    max_answer_tokens: int = 512
    temperature: float = 0.3
    n_gpu_layers: int = 0
    n_threads: Optional[int] = None
    max_sources_in_prompt: int = 8
    require_citations: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "LLMConfig":
        _require_keys(d, ["enabled", "model", "path"], section="llm")
        return cls(
            enabled=d["enabled"],
            model=d["model"],
            path=d["path"],
            download_url=d.get("download_url"),
            context_length=d.get("context_length", 4096),
            max_answer_tokens=d.get("max_answer_tokens", 512),
            temperature=d.get("temperature", 0.3),
            n_gpu_layers=d.get("n_gpu_layers", 0),
            n_threads=d.get("n_threads"),
            max_sources_in_prompt=d.get("max_sources_in_prompt", 8),
            require_citations=d.get("require_citations", True),
        )


@dataclass
class PipelineConfig:
    max_expanded_queries: int = 5
    search_timeout_seconds: float = 2.0
    dedup_similarity_threshold: float = 0.92
    embedding_cache: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        return cls(
            max_expanded_queries=d.get("max_expanded_queries", 5),
            search_timeout_seconds=d.get("search_timeout_seconds", 2.0),
            dedup_similarity_threshold=d.get("dedup_similarity_threshold", 0.92),
            embedding_cache=d.get("embedding_cache", True),
        )


# ── Root config ───────────────────────────────────────────────────────────────


@dataclass
class ModelConfig:
    embedder: EmbedderConfig
    reranker: RerankerConfig
    router: RouterConfig
    llm: LLMConfig
    pipeline: PipelineConfig

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        _require_keys(d, ["embedder", "reranker", "router", "llm", "pipeline"], section="root")
        return cls(
            embedder=EmbedderConfig.from_dict(d["embedder"]),
            reranker=RerankerConfig.from_dict(d["reranker"]),
            router=RouterConfig.from_dict(d["router"]),
            llm=LLMConfig.from_dict(d["llm"]),
            pipeline=PipelineConfig.from_dict(d["pipeline"]),
        )


# ── Loader ────────────────────────────────────────────────────────────────────


def _require_keys(d: dict, keys: list[str], section: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise ValueError(
            f"model_config.yaml [{section}] is missing required keys: {missing}"
        )


@lru_cache(maxsize=1)
def load_model_config(path: str = "./model_config.yaml") -> ModelConfig:
    """
    Load, parse, and validate model_config.yaml.

    Cached after first call. Raises ValueError with a clear message if
    required fields are absent.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"model_config.yaml not found at '{config_path.resolve()}'. "
            "Copy model_config.yaml to the project root before starting."
        )

    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError("model_config.yaml must be a YAML mapping at the top level.")

    cfg = ModelConfig.from_dict(raw)

    # Warn if model-based routing is requested but not implemented
    if cfg.router.mode == "model":
        logger.warning(
            "router.mode is set to 'model' in model_config.yaml, but "
            "model-based routing is not yet implemented. Falling back to rule-based."
        )

    logger.info("model_config.yaml loaded successfully")
    logger.info("  embedder : %s (%s)", cfg.embedder.model, cfg.embedder.repo)
    logger.info("  reranker : %s (%s)", cfg.reranker.model, cfg.reranker.repo)
    logger.info("  llm      : %s (enabled=%s)", cfg.llm.model, cfg.llm.enabled)

    return cfg
