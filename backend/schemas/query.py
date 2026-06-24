"""Pydantic models for incoming search requests."""

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    q: str = Field(..., min_length=1, max_length=500, description="The raw search query")
    top_k: int = Field(default=10, ge=1, le=50, description="Max results to return")
