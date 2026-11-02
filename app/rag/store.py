"""pgvector + Postgres full-text chunk store."""

import hashlib
import re
import uuid
from typing import Protocol

import numpy as np
from psycopg_pool import AsyncConnectionPool

from app.rag.models import Chunk, RetrievedChunk

_TOKEN = re.compile(r"\w+", re.UNICODE)


def to_or_tsquery(text: str) -> str:
    """Natural questions should match on ANY term, ranked by ts_rank (plainto_tsquery ANDs)."""
    tokens = {t.lower() for t in _TOKEN.findall(text) if len(t) > 1}
    return " | ".join(sorted(tokens))


class ChunkStore(Protocol):
    async def vector_search(
        self, embedding: np.ndarray, k: int, doc_name: str | None = None
    ) -> list[RetrievedChunk]: ...

    async def keyword_search(
        self, query: str, k: int, doc_name: str | None = None
    ) -> list[RetrievedChunk]: ...


class PgChunkStore:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def upsert_document(
        self, name: str, data: bytes, chunks: list[Chunk], embeddings: list[np.ndarray]
    ) -> tuple[str, bool]:
        """Replace a document's chunks. Returns (document_id, changed)."""
        sha = hashlib.sha256(data).hexdigest()
        async with self._pool.connection() as conn, conn.transaction():
            row = await (
                await conn.execute("SELECT id, sha256 FROM documents WHERE name = %s", (name,))
            ).fetchone()
            if row and row[1] == sha:
                return str(row[0]), False
            if row:
                await conn.execute("DELETE FROM documents WHERE id = %s", (row[0],))
            doc_id = uuid.uuid4()
            await conn.execute(
                "INSERT INTO documents (id, name, sha256, chunk_count) VALUES (%s, %s, %s, %s)",
                (doc_id, name, sha, len(chunks)),
            )
            async with conn.cursor() as cur:
                await cur.executemany(
                    "INSERT INTO chunks (document_id, doc_name, page, section, content, embedding)"
                    " VALUES (%s, %s, %s, %s, %s, %s)",
                    [
                        (doc_id, c.doc_name, c.page, c.section, c.content, emb)
                        for c, emb in zip(chunks, embeddings, strict=True)
                    ],
                )
            return str(doc_id), True

    async def list_documents(self) -> list[dict]:
        async with self._pool.connection() as conn:
            rows = await (
                await conn.execute(
                    "SELECT id, name, chunk_count, created_at FROM documents ORDER BY name"
                )
            ).fetchall()
        return [
            {"id": str(r[0]), "name": r[1], "chunk_count": r[2], "created_at": r[3].isoformat()}
            for r in rows
        ]

    async def delete_document(self, name: str) -> bool:
        async with self._pool.connection() as conn:
            cur = await conn.execute("DELETE FROM documents WHERE name = %s", (name,))
            return cur.rowcount > 0

    async def _search(self, sql: str, params: tuple) -> list[RetrievedChunk]:
        async with self._pool.connection() as conn:
            rows = await (await conn.execute(sql, params)).fetchall()
        return [
            RetrievedChunk(
                id=r[0], doc_name=r[1], page=r[2], section=r[3], content=r[4], score=float(r[5])
            )
            for r in rows
        ]

    async def vector_search(
        self, embedding: np.ndarray, k: int, doc_name: str | None = None
    ) -> list[RetrievedChunk]:
        return await self._search(
            "SELECT id, doc_name, page, section, content, 1 - (embedding <=> %s) AS score "
            "FROM chunks WHERE (%s::text IS NULL OR doc_name = %s) "
            "ORDER BY embedding <=> %s LIMIT %s",
            (embedding, doc_name, doc_name, embedding, k),
        )

    async def keyword_search(
        self, query: str, k: int, doc_name: str | None = None
    ) -> list[RetrievedChunk]:
        tsq = to_or_tsquery(query)
        if not tsq:
            return []
        return await self._search(
            "SELECT id, doc_name, page, section, content, "
            "ts_rank_cd(tsv, to_tsquery('simple', %s)) AS score "
            "FROM chunks WHERE tsv @@ to_tsquery('simple', %s) "
            "AND (%s::text IS NULL OR doc_name = %s) "
            "ORDER BY score DESC LIMIT %s",
            (tsq, tsq, doc_name, doc_name, k),
        )
