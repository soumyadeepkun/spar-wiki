from fastapi import Request
from fastapi.exceptions import HTTPException
from sqlalchemy import select

from src.core.config import settings
from src.models import Document


class SearchService:
    def __init__(self, query_encoder, sparse_engine, session_factory):
        self.query_encoder = query_encoder
        self.sparse_engine = sparse_engine
        self.session_factory = session_factory

    def search_batch(self, queries: list[str], k: int, min_score: float | None):
        if len(queries) > settings.MAX_QUERY_BATCH_SIZE:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Batch size {len(queries)} exceeds MAX_QUERY_BATCH_SIZE="
                    f"{settings.MAX_QUERY_BATCH_SIZE}"
                ),
            )

        encoded = self.query_encoder.encode_queries(queries)
        per_query_hits: list[tuple[str, list[dict[str, int | float]]]] = []
        all_doc_ids: set[int] = set()

        for query, (term_ids, term_weights) in zip(queries, encoded):
            hits = self.sparse_engine.search(
                term_ids=term_ids,
                term_weights=term_weights,
                k=k,
                min_score=min_score,
            )
            serializable_hits = []
            for hit in hits:
                serializable_hits.append({"doc_id": hit.doc_id, "score": hit.score})
                all_doc_ids.add(hit.doc_id)
            per_query_hits.append((query, serializable_hits))

        docs_map: dict[int, tuple[str, str]] = {}
        if all_doc_ids:
            session_factory = self.session_factory
            sorted_doc_ids = sorted(all_doc_ids)
            with session_factory() as session:
                for offset in range(0, len(sorted_doc_ids), settings.DOC_FETCH_CHUNK_SIZE):
                    chunk = sorted_doc_ids[offset : offset + settings.DOC_FETCH_CHUNK_SIZE]
                    docs = (
                        session.execute(select(Document).where(Document.id.in_(chunk)))
                        .scalars()
                        .all()
                    )
                    for doc in docs:
                        docs_map[doc.id] = (doc.title, doc.body)

        results = []
        for query, hits in per_query_hits:
            enriched_hits = []
            for hit in hits:
                title, body = docs_map.get(hit["doc_id"], ("", ""))
                enriched_hits.append(
                    {
                        "doc_id": hit["doc_id"],
                        "score": hit["score"],
                        "title": title,
                        "text": body,
                    }
                )
            results.append({"query": query, "count": len(enriched_hits), "topk": enriched_hits})

        return {
            "batch_size": len(queries),
            "results": results,
        }

    def search_single(self, query: str, k: int, min_score: float | None):
        result = self.search_batch([query], k, min_score)["results"][0]
        return result


def get_search_service(request: Request) -> SearchService:
    return SearchService(
        query_encoder=request.app.state.query_encoder,
        sparse_engine=request.app.state.sparse_engine,
        session_factory=request.app.state.db_session_factory,
    )
