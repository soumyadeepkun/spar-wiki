from fastapi import APIRouter, Depends, Query

from src.core.config import settings
from src.search.schemas import BatchSearchRequest, SearchRequest
from src.search.services import SearchService, get_search_service

router = APIRouter()


@router.get("/single")
async def search_get(
    query: str = Query(..., min_length=1),
    k: int = Query(settings.DEFAULT_TOP_K, ge=1, le=1000),
    min_score: float | None = Query(default=None),
    search_service: SearchService = Depends(get_search_service),
):
    return search_service.search_single(query=query, k=k, min_score=min_score)


@router.post("/single")
async def search_post(
    req: SearchRequest, search_service: SearchService = Depends(get_search_service)
):
    return search_service.search_single(query=req.query, k=req.k, min_score=req.min_score)


@router.post("/batch")
async def search_batch_post(
    req: BatchSearchRequest, search_service: SearchService = Depends(get_search_service)
):
    return search_service.search_batch(queries=req.queries, k=req.k, min_score=req.min_score)
