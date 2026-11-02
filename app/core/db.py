"""Postgres connection pool + schema bootstrap (pgvector, full-text, audit)."""

from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool


async def _configure(conn: AsyncConnection) -> None:
    await register_vector_async(conn)


async def create_pool(database_url: str) -> AsyncConnectionPool:
    # The extension must exist before register_vector can look up the `vector` type.
    async with await AsyncConnection.connect(database_url, autocommit=True) as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    pool = AsyncConnectionPool(
        database_url, min_size=1, max_size=10, configure=_configure, open=False
    )
    await pool.open(wait=True)
    return pool


def schema_sql(embedding_dim: int) -> str:
    return f"""
CREATE TABLE IF NOT EXISTS documents (
    id          UUID PRIMARY KEY,
    name        TEXT UNIQUE NOT NULL,
    sha256      TEXT NOT NULL,
    chunk_count INT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    doc_name    TEXT NOT NULL,
    page        INT,
    section     TEXT,
    content     TEXT NOT NULL,
    embedding   VECTOR({embedding_dim}) NOT NULL,
    tsv         TSVECTOR GENERATED ALWAYS AS (
                    to_tsvector('simple', coalesce(section, '') || ' ' || content)
                ) STORED
);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_doc_name ON chunks (doc_name);

CREATE TABLE IF NOT EXISTS audit_log (
    id        BIGSERIAL PRIMARY KEY,
    ts        TIMESTAMPTZ NOT NULL DEFAULT now(),
    user_id   TEXT NOT NULL,
    thread_id TEXT,
    event     TEXT NOT NULL,
    detail    JSONB NOT NULL DEFAULT '{{}}'::jsonb
);
CREATE INDEX IF NOT EXISTS audit_log_user_ts ON audit_log (user_id, ts DESC);
"""


async def init_schema(pool: AsyncConnectionPool, embedding_dim: int) -> None:
    async with pool.connection() as conn:
        await conn.execute(schema_sql(embedding_dim))
