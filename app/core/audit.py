"""Append-only audit trail. Only masked text is ever written here."""

import json
import logging
from typing import Any, Protocol

from psycopg_pool import AsyncConnectionPool

log = logging.getLogger(__name__)


class AuditLog(Protocol):
    async def write(
        self, user_id: str, event: str, thread_id: str | None = None, **detail: Any
    ) -> None: ...


class PostgresAuditLog:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def write(
        self, user_id: str, event: str, thread_id: str | None = None, **detail: Any
    ) -> None:
        try:
            async with self._pool.connection() as conn:
                await conn.execute(
                    "INSERT INTO audit_log (user_id, thread_id, event, detail) "
                    "VALUES (%s, %s, %s, %s::jsonb)",
                    (user_id, thread_id, event, json.dumps(detail, ensure_ascii=False)),
                )
        except Exception:  # audit must never break the request path
            log.exception("audit write failed")


class MemoryAuditLog:
    """Used in tests and local runs without Postgres."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    async def write(
        self, user_id: str, event: str, thread_id: str | None = None, **detail: Any
    ) -> None:
        self.records.append({"user_id": user_id, "event": event, "thread_id": thread_id, **detail})
