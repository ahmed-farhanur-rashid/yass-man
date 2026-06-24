"""
FastAPI Dependency Injection — provides access to pipeline components via request.app.state.
"""

from __future__ import annotations

from typing import Optional

from fastapi import Request

from backend.cache.embedding_cache import EmbeddingCache
from backend.config import Settings
from backend.logging.feedback import FeedbackLogger
from backend.logging.query_logger import QueryLogger
from backend.model_config_loader import ModelConfig
from backend.models.embedder_model import EmbedderModel
from backend.models.llm_model import LLMModel
from backend.models.reranker_model import RerankerModel
from backend.pipeline.aggregator import Aggregator
from backend.pipeline.embedder import EmbedderStage
from backend.pipeline.expander import QueryExpander
from backend.pipeline.reranker import RerankerStage
from backend.pipeline.retriever import Retriever
from backend.pipeline.router import QueryRouter
from backend.pipeline.synthesizer import Synthesizer


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_model_config(request: Request) -> ModelConfig:
    return request.app.state.model_cfg


def get_router(request: Request) -> QueryRouter:
    return request.app.state.query_router


def get_expander(request: Request) -> QueryExpander:
    return request.app.state.query_expander


def get_retriever(request: Request) -> Retriever:
    return request.app.state.retriever


def get_aggregator(request: Request) -> Aggregator:
    return request.app.state.aggregator


def get_embedder_stage(request: Request) -> EmbedderStage:
    return request.app.state.embedder_stage


def get_reranker_stage(request: Request) -> RerankerStage:
    return request.app.state.reranker_stage


def get_synthesizer(request: Request) -> Optional[Synthesizer]:
    return getattr(request.app.state, "synthesizer", None)


def get_query_logger(request: Request) -> QueryLogger:
    return request.app.state.query_logger


def get_feedback_logger(request: Request) -> FeedbackLogger:
    return request.app.state.feedback_logger
