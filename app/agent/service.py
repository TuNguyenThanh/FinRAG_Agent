"""Chat orchestration: guardrails -> PII masking -> semantic cache -> agent stream -> audit.

Yields transport-agnostic events so the same logic backs SSE streaming and plain JSON.
"""

import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.core.audit import AuditLog
from app.core.cache import SemanticCache
from app.core.config import Settings
from app.core.telemetry import (
    CACHE_EVENTS,
    FIRST_TOKEN_LATENCY,
    GUARD_BLOCKS,
    LLM_COST,
    LLM_TOKENS,
    PII_MASKED,
    REQUEST_LATENCY,
    estimate_cost,
)
from app.security.guard import REFUSAL_MESSAGE, check_input
from app.security.pii import mask_pii

log = logging.getLogger(__name__)


@dataclass
class ChatEvent:
    event: str  # meta | token | sources | done | error
    data: dict[str, Any] = field(default_factory=dict)


class ChatService:
    def __init__(
        self,
        agent: Any,
        settings: Settings,
        audit: AuditLog,
        cache: SemanticCache | None = None,
    ) -> None:
        self.agent = agent
        self.settings = settings
        self.audit = audit
        self.cache = cache

    @staticmethod
    def _config(user_id: str, thread_id: str) -> dict:
        # Namespacing by user means one customer can never load another's conversation.
        return {"configurable": {"thread_id": f"{user_id}:{thread_id}"}}

    async def _is_new_thread(self, config: dict) -> bool:
        state = await self.agent.aget_state(config)
        return not state.values.get("messages")

    async def stream(
        self, user_id: str, message: str, thread_id: str | None = None
    ) -> AsyncIterator[ChatEvent]:
        start = time.perf_counter()
        thread_id = thread_id or uuid.uuid4().hex
        config = self._config(user_id, thread_id)

        guard = check_input(message, self.settings.max_message_chars)
        if not guard.allowed:
            GUARD_BLOCKS.labels(reason=guard.reason).inc()
            await self.audit.write(user_id, "chat.blocked", thread_id, reason=guard.reason)
            yield ChatEvent("meta", {"thread_id": thread_id, "cached": False})
            yield ChatEvent("token", {"text": REFUSAL_MESSAGE})
            yield ChatEvent("done", {"blocked": True, "reason": guard.reason})
            return

        masked = mask_pii(message)
        for kind, n in masked.found.items():
            PII_MASKED.labels(kind=kind).inc(n)
        question = masked.text

        # Cache only first turns: follow-ups depend on conversation context.
        use_cache = self.cache is not None and not masked.has_pii
        new_thread = await self._is_new_thread(config) if use_cache else False
        if use_cache and new_thread:
            hit = await self.cache.lookup(question)
            CACHE_EVENTS.labels(result="hit" if hit else "miss").inc()
            if hit:
                # Keep memory consistent so follow-up questions still have context.
                await self.agent.aupdate_state(
                    config,
                    {"messages": [HumanMessage(question), AIMessage(hit.answer)]},
                    as_node="model",
                )
                latency = time.perf_counter() - start
                REQUEST_LATENCY.labels(cached="true").observe(latency)
                await self.audit.write(
                    user_id,
                    "chat.cache_hit",
                    thread_id,
                    question=question,
                    similarity=round(hit.similarity, 4),
                )
                yield ChatEvent("meta", {"thread_id": thread_id, "cached": True})
                yield ChatEvent("token", {"text": hit.answer})
                yield ChatEvent("sources", {"sources": hit.sources})
                yield ChatEvent(
                    "done",
                    {
                        "cached": True,
                        "latency_ms": round(latency * 1000, 1),
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cost_usd": 0.0,
                        "pii_masked": masked.found,
                    },
                )
                return

        yield ChatEvent("meta", {"thread_id": thread_id, "cached": False})

        answer_parts: list[str] = []
        sources: dict[str, dict] = {}
        in_tok = out_tok = 0
        first_token_at: float | None = None
        try:
            async for msg, meta in self.agent.astream(
                {"messages": [HumanMessage(question)]}, config, stream_mode="messages"
            ):
                node = meta.get("langgraph_node")
                if isinstance(msg, ToolMessage) and msg.artifact:
                    for src in msg.artifact:
                        sources.setdefault(f"{src['doc_name']}#{src['id']}", src)
                elif isinstance(msg, AIMessage) and node == "model":
                    if msg.usage_metadata:
                        in_tok += msg.usage_metadata.get("input_tokens", 0)
                        out_tok += msg.usage_metadata.get("output_tokens", 0)
                    if text := msg.text:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                            FIRST_TOKEN_LATENCY.observe(first_token_at - start)
                        answer_parts.append(text)
                        yield ChatEvent("token", {"text": text})
        except Exception as exc:
            log.exception("agent run failed")
            await self.audit.write(user_id, "chat.error", thread_id, error=type(exc).__name__)
            yield ChatEvent("error", {"message": "Hệ thống đang bận, vui lòng thử lại sau."})
            return

        answer = "".join(answer_parts)
        source_list = list(sources.values())
        cost = estimate_cost(
            in_tok,
            out_tok,
            self.settings.llm_price_input_per_mtok,
            self.settings.llm_price_output_per_mtok,
        )
        LLM_TOKENS.labels(kind="input").inc(in_tok)
        LLM_TOKENS.labels(kind="output").inc(out_tok)
        LLM_COST.inc(cost)
        latency = time.perf_counter() - start
        REQUEST_LATENCY.labels(cached="false").observe(latency)

        # Only cache grounded answers, never "I don't know" or uncited replies.
        if use_cache and new_thread and source_list and answer:
            await self.cache.store(question, answer, source_list)

        await self.audit.write(
            user_id,
            "chat.answer",
            thread_id,
            question=question,
            answer_chars=len(answer),
            sources=len(source_list),
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=round(latency * 1000, 1),
            pii_masked=masked.found,
        )
        yield ChatEvent("sources", {"sources": source_list})
        yield ChatEvent(
            "done",
            {
                "cached": False,
                "latency_ms": round(latency * 1000, 1),
                "ttft_ms": round((first_token_at - start) * 1000, 1) if first_token_at else None,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "cost_usd": round(cost, 6),
                "pii_masked": masked.found,
            },
        )

    async def answer(self, user_id: str, message: str, thread_id: str | None = None) -> dict:
        """Non-streaming variant (mobile clients without SSE, evals)."""
        result: dict[str, Any] = {"answer": "", "sources": []}
        async for ev in self.stream(user_id, message, thread_id):
            if ev.event == "token":
                result["answer"] += ev.data["text"]
            elif ev.event == "sources":
                result["sources"] = ev.data["sources"]
            elif ev.event in ("meta", "done", "error"):
                result.update(ev.data if ev.event != "error" else {"error": ev.data["message"]})
        return result
