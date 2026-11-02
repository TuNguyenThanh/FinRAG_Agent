"""Hybrid retrieval: dense (pgvector) + sparse (Postgres FTS), fused with RRF, then reranked."""

import asyncio
import time

from app.core.telemetry import RETRIEVAL_LATENCY
from app.rag.embeddings import Embedder
from app.rag.models import RetrievedChunk
from app.rag.rerank import Reranker
from app.rag.store import ChunkStore


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedChunk]], k: int = 60
) -> list[RetrievedChunk]:
    """Score = sum(1 / (k + rank)). Robust to the two retrievers' scores being incomparable."""
    fused: dict[int, RetrievedChunk] = {}
    scores: dict[int, float] = {}
    for results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            fused.setdefault(chunk.id, chunk)
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
    ordered = sorted(fused.values(), key=lambda c: scores[c.id], reverse=True)
    for c in ordered:
        c.score = scores[c.id]
    return ordered


class HybridRetriever:
    def __init__(
        self,
        store: ChunkStore,
        embedder: Embedder,
        reranker: Reranker | None = None,
        candidates: int = 20,
        top_k: int = 5,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.candidates = candidates
        self.top_k = top_k

    async def search(
        self, query: str, doc_name: str | None = None, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        start = time.perf_counter()
        top_k = top_k or self.top_k
        embedding = await self.embedder.embed_query(query)
        dense, sparse = await asyncio.gather(
            self.store.vector_search(embedding, self.candidates, doc_name),
            self.store.keyword_search(query, self.candidates, doc_name),
        )
        fused = reciprocal_rank_fusion([dense, sparse])
        if self.reranker:
            fused = await self.reranker.rerank(query, fused[: top_k * 3])
        RETRIEVAL_LATENCY.observe(time.perf_counter() - start)
        return fused[:top_k]
