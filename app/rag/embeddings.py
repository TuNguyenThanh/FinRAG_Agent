"""Embedding backends. FastEmbed runs locally (ONNX), so no embedding API key is needed."""

import asyncio
import hashlib
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int

    async def embed_documents(self, texts: list[str]) -> list[np.ndarray]: ...
    async def embed_query(self, text: str) -> np.ndarray: ...


class FastEmbedEmbedder:
    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)
        self.dim = len(next(iter(self._model.embed(["dim probe"]))))

    async def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        return await asyncio.to_thread(lambda: list(self._model.embed(texts, batch_size=32)))

    async def embed_query(self, text: str) -> np.ndarray:
        return (await self.embed_documents([text]))[0]


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder for tests (no model download)."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in text.lower().split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    async def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        return [self._embed(t) for t in texts]

    async def embed_query(self, text: str) -> np.ndarray:
        return self._embed(text)
