from pydantic import BaseModel, Field

from src.core.config import settings


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(default=settings.DEFAULT_TOP_K, ge=1, le=1000)
    min_score: float | None = None


class BatchSearchRequest(BaseModel):
    queries: list[str] = Field(..., min_length=1)
    k: int = Field(default=settings.DEFAULT_TOP_K, ge=1, le=1000)
    min_score: float | None = None
