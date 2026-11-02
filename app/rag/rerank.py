import asyncio
from typing import Protocol

from app.rag.models import RetrievedChunk


class Reranker(Protocol):
    async def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]: ...


class CrossEncoderReranker:
    """Multilingual cross-encoder (ONNX via FastEmbed) to re-score fused candidates."""

    def __init__(self, model_name: str) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name=model_name)

    async def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return chunks
        scores = await asyncio.to_thread(
            lambda: list(self._model.rerank(query, [c.content for c in chunks]))
        )
        for c, s in zip(chunks, scores, strict=True):
            c.score = float(s)
        return sorted(chunks, key=lambda c: c.score, reverse=True)
