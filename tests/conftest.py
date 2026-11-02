import json

import fakeredis
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.graph import build_agent
from app.agent.service import ChatService
from app.core.audit import MemoryAuditLog
from app.core.cache import SemanticCache
from app.core.config import Settings, get_settings
from app.main import create_app
from app.rag.embeddings import HashEmbedder
from app.rag.models import Chunk, RetrievedChunk
from app.rag.retriever import HybridRetriever
from app.rag.store import to_or_tsquery
from app.security.ratelimit import RateLimiter
from app.services import Services


class FakeModel(GenericFakeChatModel):
    """Scripted chat model; `bind_tools` is a no-op so it can drive a real agent graph."""

    def bind_tools(self, tools, **kwargs):
        return self


class InMemoryChunkStore:
    def __init__(self, embedder: HashEmbedder, chunks: list[Chunk]) -> None:
        self.embedder = embedder
        self.rows = [RetrievedChunk(id=i, **vars(c)) for i, c in enumerate(chunks, start=1)]
        self.vectors = [embedder._embed(c.content) for c in chunks]

    def _copy(self, row: RetrievedChunk, score: float) -> RetrievedChunk:
        return RetrievedChunk(**{**vars(row), "score": score})

    async def vector_search(self, embedding, k, doc_name=None):
        scored = [
            (float(np.dot(v, embedding)), r)
            for v, r in zip(self.vectors, self.rows, strict=True)
            if doc_name is None or r.doc_name == doc_name
        ]
        return [self._copy(r, s) for s, r in sorted(scored, key=lambda x: -x[0])[:k]]

    async def keyword_search(self, query, k, doc_name=None):
        terms = set(to_or_tsquery(query).split(" | "))
        scored = [
            (len(terms & set(r.content.lower().split())), r)
            for r in self.rows
            if doc_name is None or r.doc_name == doc_name
        ]
        return [self._copy(r, s) for s, r in sorted(scored, key=lambda x: -x[0]) if s][:k]


SAMPLE_CHUNKS = [
    Chunk(
        "bieu-phi.md",
        "Phí chuyển khoản liên ngân hàng là 0 đồng trên kênh app.",
        None,
        "Chuyển khoản",
    ),
    Chunk("bieu-phi.md", "Phí thường niên thẻ tín dụng Gold là 500.000 đồng mỗi năm.", None, "Thẻ"),
    Chunk("vay.md", "Lãi suất vay tiêu dùng tín chấp từ 12% đến 18% một năm.", None, "Lãi suất"),
]


def tool_call(query: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "search_documents", "args": {"query": query}, "id": "call_1"}],
        usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110},
    )


def final(text: str) -> AIMessage:
    return AIMessage(
        content=text,
        usage_metadata={"input_tokens": 300, "output_tokens": 40, "total_tokens": 340},
    )


@pytest.fixture
def settings() -> Settings:
    return Settings(
        app_env="test",
        api_keys="test-key:alice,other-key:bob",
        rerank_enabled=False,
        checkpointer="memory",
        llm_price_input_per_mtok=1.0,
        llm_price_output_per_mtok=5.0,
        rate_limit_per_minute=1000,
    )


@pytest.fixture
def embedder() -> HashEmbedder:
    return HashEmbedder()


@pytest.fixture
def retriever(embedder) -> HybridRetriever:
    return HybridRetriever(InMemoryChunkStore(embedder, SAMPLE_CHUNKS), embedder, None, 10, 3)


@pytest.fixture
def redis():
    return fakeredis.FakeAsyncRedis(decode_responses=True)


@pytest.fixture
def make_services(settings, embedder, retriever, redis):
    def _make(scripted: list[AIMessage], with_cache: bool = True) -> Services:
        model = FakeModel(messages=iter(scripted), disable_streaming=True)
        agent = build_agent(model, retriever, InMemorySaver(), settings)
        cache = SemanticCache(redis, embedder, 0.93, 60, 100) if with_cache else None
        audit = MemoryAuditLog()
        return Services(
            settings=settings,
            chat=ChatService(agent, settings, audit, cache),
            retriever=retriever,
            embedder=embedder,
            audit=audit,
            cache=cache,
            rate_limiter=RateLimiter(redis, settings.rate_limit_per_minute),
        )

    return _make


@pytest.fixture
def client_for(settings):
    """Build an httpx client for a given Services instance."""

    async def _client(services: Services) -> AsyncClient:
        app = create_app(services)
        app.dependency_overrides[get_settings] = lambda: settings
        app.state.services = services
        return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

    return _client


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events
