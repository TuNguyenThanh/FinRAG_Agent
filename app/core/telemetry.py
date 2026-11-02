"""Prometheus metrics for latency, token usage, cost and cache efficiency."""

from prometheus_client import Counter, Histogram

REQUEST_LATENCY = Histogram(
    "finrag_chat_latency_seconds",
    "End-to-end /chat latency",
    ["cached"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32),
)
FIRST_TOKEN_LATENCY = Histogram(
    "finrag_chat_first_token_seconds",
    "Time to first streamed token",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8),
)
RETRIEVAL_LATENCY = Histogram(
    "finrag_retrieval_latency_seconds", "Hybrid retrieval + rerank latency"
)
LLM_TOKENS = Counter("finrag_llm_tokens_total", "LLM tokens", ["kind"])
LLM_COST = Counter("finrag_llm_cost_usd_total", "Estimated LLM cost in USD")
CACHE_EVENTS = Counter("finrag_cache_events_total", "Semantic cache lookups", ["result"])
GUARD_BLOCKS = Counter("finrag_guard_blocks_total", "Requests blocked by guardrails", ["reason"])
PII_MASKED = Counter("finrag_pii_masked_total", "PII entities masked", ["kind"])


def estimate_cost(
    input_tokens: int, output_tokens: int, in_price: float, out_price: float
) -> float:
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000
