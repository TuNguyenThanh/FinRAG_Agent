"""Wire concrete infrastructure into the app. Tests build `Services` with fakes instead."""

import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass, field

from fastapi import Request
from redis.asyncio import Redis

from app.agent.graph import build_agent
from app.agent.service import ChatService
from app.core.audit import AuditLog, PostgresAuditLog
from app.core.cache import SemanticCache
from app.core.config import Settings
from app.core.db import create_pool, init_schema
from app.rag.embeddings import Embedder, FastEmbedEmbedder
from app.rag.retriever import HybridRetriever
from app.rag.store import PgChunkStore
from app.security.ratelimit import RateLimiter

log = logging.getLogger(__name__)


@dataclass
class Services:
    settings: Settings
    chat: ChatService
    retriever: HybridRetriever
    embedder: Embedder
    audit: AuditLog
    store: PgChunkStore | None = None
    cache: SemanticCache | None = None
    rate_limiter: RateLimiter | None = None
    stack: AsyncExitStack = field(default_factory=AsyncExitStack)

    async def aclose(self) -> None:
        await self.stack.aclose()


async def build_services(settings: Settings) -> Services:
    from langchain.chat_models import init_chat_model

    stack = AsyncExitStack()

    embedder = FastEmbedEmbedder(settings.embedding_model)
    reranker = None
    if settings.rerank_enabled:
        from app.rag.rerank import CrossEncoderReranker

        reranker = CrossEncoderReranker(settings.rerank_model)

    pool = await create_pool(settings.database_url)
    stack.push_async_callback(pool.close)
    await init_schema(pool, embedder.dim)
    store = PgChunkStore(pool)
    retriever = HybridRetriever(
        store, embedder, reranker, settings.retrieval_candidates, settings.retrieval_top_k
    )

    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    stack.push_async_callback(redis.aclose)
    cache = (
        SemanticCache(
            redis,
            embedder,
            settings.cache_similarity_threshold,
            settings.cache_ttl_seconds,
            settings.cache_max_entries,
        )
        if settings.cache_enabled
        else None
    )

    if settings.checkpointer == "postgres":
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        checkpointer = await stack.enter_async_context(
            AsyncPostgresSaver.from_conn_string(settings.database_url)
        )
        await checkpointer.setup()
    else:
        from langgraph.checkpoint.memory import InMemorySaver

        checkpointer = InMemorySaver()

    model_kwargs: dict = {"max_tokens": settings.llm_max_tokens}
    if settings.llm_temperature is not None:
        model_kwargs["temperature"] = settings.llm_temperature
    model = init_chat_model(settings.llm_model, **model_kwargs)
    audit = PostgresAuditLog(pool)
    agent = build_agent(model, retriever, checkpointer, settings)

    return Services(
        settings=settings,
        chat=ChatService(agent, settings, audit, cache),
        retriever=retriever,
        embedder=embedder,
        audit=audit,
        store=store,
        cache=cache,
        rate_limiter=RateLimiter(redis, settings.rate_limit_per_minute),
        stack=stack,
    )


def get_services(request: Request) -> Services:
    return request.app.state.services
