from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from heapq import nlargest
from typing import Any, Sequence

import numpy as np
import torch
from sentence_transformers import SparseEncoder

from src.retrieval.base import RetrievalEngine


@dataclass(frozen=True)
class SearchHit:  # DTO for a search hit
    doc_id: int
    score: float


class LinearUInt8Codec:
    def __init__(self, w_max: float = 8.0, qmax: int = 255):
        self.w_max = float(w_max)
        self.qmax = int(qmax)
        self.scale = self.w_max / self.qmax

    def dequantize(self, quantized: np.ndarray) -> np.ndarray:
        return quantized.astype(np.float32, copy=False) * self.scale


class SparseRetrievalEngine(RetrievalEngine):
    def __init__(
        self,
        term_offsets: np.ndarray,
        doc_ids: np.ndarray,
        term_weights: np.ndarray,
        *,
        w_max: float = 8.0,
        qmax: int = 255,
    ):
        if term_offsets.ndim != 2 or term_offsets.shape[1] != 3:
            raise ValueError("term_offsets must have shape [n_terms, 3]")
        if doc_ids.shape[0] != term_weights.shape[0]:
            raise ValueError("doc_ids and term_weights must have equal lengths")

        self.term_offsets = term_offsets
        self.doc_ids = doc_ids
        self.term_weights = term_weights
        self.codec = LinearUInt8Codec(w_max=w_max, qmax=qmax)

    @classmethod
    def from_files(
        cls,
        term_offsets_path: str,
        doc_ids_path: str,
        term_weights_path: str,
        *,
        offsets_key: str = "offsets",
        mmap_mode: str = "r",
        w_max: float = 8.0,
        qmax: int = 255,
    ) -> "SparseRetrievalEngine":
        with np.load(term_offsets_path) as loaded:
            term_offsets = loaded[offsets_key]
        doc_ids = np.load(doc_ids_path, mmap_mode=mmap_mode)
        term_weights = np.load(term_weights_path, mmap_mode=mmap_mode)
        return cls(
            term_offsets=term_offsets,
            doc_ids=doc_ids,
            term_weights=term_weights,
            w_max=w_max,
            qmax=qmax,
        )

    @property
    def indexed_terms(self) -> int:
        return int(self.term_offsets.shape[0])

    def _lookup_span(self, term_id: int) -> tuple[int, int] | None:
        terms = self.term_offsets[:, 0]
        pos = int(np.searchsorted(terms, term_id))
        if pos >= terms.size or int(terms[pos]) != term_id:
            return None
        start, end = self.term_offsets[pos, 1:3]
        return int(start), int(end)

    def search(
        self,
        term_ids: Sequence[int],
        term_weights: Sequence[float] | None = None,
        k: int = 10,
        min_score: float | None = None,
    ) -> list[SearchHit]:
        if len(term_ids) == 0:
            return []

        q_term_ids = np.asarray(term_ids, dtype=np.int64)
        if term_weights is None:
            q_weights = np.ones_like(q_term_ids, dtype=np.float32)
        else:
            if len(term_weights) != len(q_term_ids):
                raise ValueError("term_weights length must match term_ids length")
            q_weights = np.asarray(term_weights, dtype=np.float32)

        scores: dict[int, float] = defaultdict(float)

        for term_id, query_weight in zip(q_term_ids, q_weights):
            span = self._lookup_span(int(term_id))
            if span is None:
                continue

            start, end = span
            docs = self.doc_ids[start:end].astype(np.int64, copy=False)
            weights = self.codec.dequantize(self.term_weights[start:end]) * float(query_weight)

            if docs.size == 0:
                continue

            # Aggregate repeated doc IDs within a postings slice.
            unique_docs, inverse = np.unique(docs, return_inverse=True)
            partial_scores = np.bincount(inverse, weights=weights)
            for doc_id, score in zip(unique_docs.tolist(), partial_scores.tolist()):
                scores[int(doc_id)] += float(score)

        if min_score is not None:
            scored_items = [
                (doc_id, score) for doc_id, score in scores.items() if score >= min_score
            ]
        else:
            scored_items = list(scores.items())

        top_hits = nlargest(max(0, int(k)), scored_items, key=lambda item: item[1])
        return [SearchHit(doc_id=doc_id, score=score) for doc_id, score in top_hits]


class SparseQueryEncoder:
    def __init__(
        self,
        hf_token: str,
        model_name: str = "naver/splade-v3",
        device: str = "cpu",
    ):
        if not hf_token or not hf_token.strip():
            raise ValueError("HF token is required for SparseQueryEncoder")

        init_kwargs: dict[str, Any] = {"device": device, "token": hf_token}
        self.model = SparseEncoder(model_name, **init_kwargs)

    def _from_torch_row(self, row: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        if row.is_sparse:
            coalesced = row.coalesce()
            term_ids = coalesced.indices().view(-1).cpu().numpy().astype(np.int64)
            term_weights = coalesced.values().cpu().numpy().astype(np.float32)
            return term_ids, term_weights

        nz = torch.nonzero(row, as_tuple=False).view(-1)
        term_ids = nz.cpu().numpy().astype(np.int64)
        term_weights = row[nz].cpu().numpy().astype(np.float32)
        return term_ids, term_weights

    def _rows_to_pairs(self, embeddings: object) -> list[tuple[np.ndarray, np.ndarray]]:
        if isinstance(embeddings, torch.Tensor):
            if embeddings.ndim == 1:
                return [self._from_torch_row(embeddings)]
            return [self._from_torch_row(embeddings[i]) for i in range(embeddings.shape[0])]

        if isinstance(embeddings, np.ndarray):
            if embeddings.ndim == 1:
                nz = np.flatnonzero(embeddings)
                return [(nz.astype(np.int64), embeddings[nz].astype(np.float32))]
            rows: list[tuple[np.ndarray, np.ndarray]] = []
            for i in range(embeddings.shape[0]):
                row = embeddings[i]
                nz = np.flatnonzero(row)
                rows.append((nz.astype(np.int64), row[nz].astype(np.float32)))
            return rows

        if isinstance(embeddings, list):
            rows = []
            for emb in embeddings:
                if isinstance(emb, torch.Tensor):
                    rows.append(self._from_torch_row(emb))
                elif isinstance(emb, np.ndarray):
                    nz = np.flatnonzero(emb)
                    rows.append((nz.astype(np.int64), emb[nz].astype(np.float32)))
                else:
                    raise TypeError(f"Unsupported embedding element type: {type(emb)!r}")
            return rows

        raise TypeError(f"Unsupported embedding container type: {type(embeddings)!r}")

    def encode_queries(self, queries: Sequence[str]) -> list[tuple[np.ndarray, np.ndarray]]:
        embeddings = self.model.encode_query(list(queries))
        return self._rows_to_pairs(embeddings)
