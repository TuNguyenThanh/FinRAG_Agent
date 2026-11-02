"""Semantic answer cache: reuse an answer when a new question is close enough in meaning.

Entries live in Redis with a TTL so cached answers expire after document updates. Lookup is
a brute-force cosine scan over at most `max_entries` vectors, which is fast (<1 ms for 1k x
384 floats) and keeps the infra to plain Redis. Swap for a Redis/pgvector ANN index at scale.
"""

import hashlib
import json
import time
from dataclasses import dataclass

import numpy as np
from redis.asyncio import Redis

from app.rag.embeddings import Embedder

_INDEX = "semcache:index"


@dataclass
class CacheHit:
    answer: str
    sources: list[dict]
    similarity: float
    question: str


class SemanticCache:
    def __init__(
        self,
        redis: Redis,
        embedder: Embedder,
        threshold: float,
        ttl_seconds: int,
        max_entries: int,
    ) -> None:
        self._redis = redis
        self._embedder = embedder
        self._threshold = threshold
        self._ttl = ttl_seconds
        self._max = max_entries

    async def lookup(self, question: str) -> CacheHit | None:
        keys = await self._redis.zrevrange(_INDEX, 0, self._max - 1)
        if not keys:
            return None
        raw = await self._redis.mget(keys)
        live = [(k, json.loads(v)) for k, v in zip(keys, raw, strict=True) if v is not None]
        stale = [k for k, v in zip(keys, raw, strict=True) if v is None]
        if stale:
            await self._redis.zrem(_INDEX, *stale)
        if not live:
            return None
        query = await self._embedder.embed_query(question)
        matrix = np.asarray([e["embedding"] for _, e in live], dtype=np.float32)
        sims = matrix @ query / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(query) + 1e-9)
        best = int(np.argmax(sims))
        if sims[best] < self._threshold:
            return None
        entry = live[best][1]
        return CacheHit(entry["answer"], entry["sources"], float(sims[best]), entry["question"])

    async def store(self, question: str, answer: str, sources: list[dict]) -> None:
        emb = await self._embedder.embed_query(question)
        key = "semcache:" + hashlib.sha256(question.strip().lower().encode()).hexdigest()[:32]
        payload = {
            "question": question,
            "answer": answer,
            "sources": sources,
            "embedding": [float(x) for x in emb],
        }
        pipe = self._redis.pipeline()
        pipe.set(key, json.dumps(payload, ensure_ascii=False), ex=self._ttl)
        pipe.zadd(_INDEX, {key: time.time()})
        pipe.zremrangebyrank(_INDEX, 0, -self._max - 1)
        await pipe.execute()

    async def clear(self) -> None:
        keys = await self._redis.zrange(_INDEX, 0, -1)
        if keys:
            await self._redis.delete(*keys)
        await self._redis.delete(_INDEX)
