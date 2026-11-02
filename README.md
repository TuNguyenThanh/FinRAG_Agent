# FinRAG Agent

A production-style **RAG Q&A agent over banking documents**, built with **LangChain v1**, **FastAPI**, **PostgreSQL + pgvector** and **Redis**.

It answers customer questions about fees, cards, loans and eKYC. Every figure in an answer comes with a citation, and the agent says "I don't know" when the documents don't cover the question. It is built around the concerns a bank actually has: PII never reaches the LLM, prompt injection is blocked before any token is spent, every request is audited, and latency/cost are measured.

> All documents in `data/samples/` describe a **fictional "DemoBank"**. Numbers are made up for demo purposes.

```
┌──────────────┐  REST + SSE   ┌───────────────────────────── api (FastAPI) ─────────────────────────────┐
│ Web / Mobile │ ────────────► │ auth (API key) → rate limit (Redis) → guardrail → PII masking           │
└──────────────┘               │      → semantic cache (Redis) ──hit──► answer (0 tokens)                │
                               │      → LangChain v1 agent ──► tool: search_documents                    │
                               │            │  middleware: call limits · transient retry · summarization │
                               │            │  memory: Postgres checkpointer (per-user threads)          │
                               │            ▼                                                            │
                               │      Hybrid retriever: pgvector (dense) + Postgres FTS (sparse) → RRF   │
                               │                        → optional cross-encoder rerank                  │
                               │      → audit log (Postgres) · Prometheus metrics                        │
                               └─────────────────────────────────────────────────────────────────────────┘
```

## Features

| Area | What's implemented |
|---|---|
| **Agent** | LangChain v1 `create_agent` with a retrieval tool. The agent decides when to search and can search several times for multi-part questions. Answers must cite `[doc, page, section]`. |
| **Memory** | LangGraph `AsyncPostgresSaver` stores conversation state per `thread_id`. Threads are namespaced by user, so one customer can never load another's history. `SummarizationMiddleware` compacts long threads. |
| **Retrieval** | Section-aware chunking (Markdown headers, DOCX headings, PDF pages). Hybrid **dense (pgvector HNSW) + sparse (Postgres full-text)** search fused with **Reciprocal Rank Fusion**, plus an optional multilingual cross-encoder reranker. Local ONNX embeddings, so no embedding API cost. |
| **Security** | API-key auth. Redis rate limit (works across replicas). Deterministic **PII masking** for Vietnamese IDs (CCCD, phone, account numbers by cue word, Luhn-validated card numbers, email) at the API boundary: the LLM, cache, logs and audit trail only see masked text. **Prompt-injection guard** in Vietnamese and English (diacritic-insensitive) that blocks requests before any LLM call. Retrieved text is fenced as untrusted data against indirect injection. Provider errors are never streamed to users. The container runs as a non-root user. |
| **Performance / cost** | SSE token streaming. **Semantic cache** (cosine similarity on embeddings, TTL, invalidated on document change) serves repeat questions with zero tokens. Model/tool call limits per run. Retries only on transient errors (429/5xx/timeouts). Prometheus metrics: latency, time-to-first-token, retrieval latency, tokens, USD cost, cache hits, guardrail blocks. |
| **Ops** | Docker Compose (Postgres + pgvector, Redis, API, optional Prometheus). Models are baked into the image. Health check. GitHub Actions CI runs lint, tests and a Docker build. |
| **Evaluation** | `evals/` contains 30 questions (26 answerable, 4 out-of-scope). It reports retrieval hit@k / MRR / evidence recall, and answer accuracy, citation rate, abstention, p50/p95 latency, cost and LLM-judge faithfulness. |
| **LLM provider** | Any `init_chat_model` id: Claude (default), OpenAI or Gemini, switched with one env var. |

## Quick start

```bash
cp .env.example .env          # set ANTHROPIC_API_KEY (or another provider)
make up                       # postgres + redis + api on :8000
make ingest                   # index data/samples/*
open http://localhost:8000    # demo chat UI  (API docs: /docs)
```

```bash
make ask Q="Phí thường niên thẻ tín dụng Gold là bao nhiêu?"   # example output below is abridged
```

```
event: meta
data: {"thread_id": "9f1c…", "cached": false}

event: token
data: {"text": "Phí thường niên thẻ Gold là 599.000 đồng/năm, miễn phí năm đầu …"}

event: sources
data: {"sources": [{"doc_name": "dieu-khoan-the-tin-dung.md", "section": "Các hạng thẻ", "citation": "…"}]}

event: done
data: {"cached": false, "latency_ms": …, "ttft_ms": …, "input_tokens": …, "output_tokens": …, "cost_usd": …}
```

Clients without SSE (e.g. a simple mobile integration) can send `"stream": false` and get a single JSON response. Pass the returned `thread_id` back to continue the conversation.

### API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/chat` | Ask a question (SSE stream or JSON) |
| `POST` | `/v1/search` | Inspect hybrid retrieval results (debugging, evals) |
| `POST` | `/v1/documents` | Upload PDF / DOCX / MD / TXT. Idempotent by content hash. |
| `GET` / `DELETE` | `/v1/documents[/{name}]` | List / remove documents (clears the semantic cache) |
| `GET` | `/metrics`, `/healthz` | Prometheus, liveness |

All `/v1` endpoints require the `X-API-Key` header.

## Evaluation results

Retrieval, measured on the 27 answerable questions on an Apple-silicon laptop CPU (`make eval`):

| Config | hit@1 | hit@5 | evidence@5 | MRR | retrieval latency (avg) |
|---|---|---|---|---|---|
| Hybrid (dense + FTS, RRF) — **default** | 0.889 | 1.000 | 0.852 | 0.933 | **~8 ms** |
| Hybrid + cross-encoder rerank | **0.926** | 1.000 | **0.889** | **0.963** | ~1.7 s |

**Decision:** reranking gains about 4 points of hit@1 but costs about 200× retrieval latency on CPU. That is a poor trade for an interactive banking assistant, so it is off by default (`RERANK_ENABLED=true` to turn it on, ideally on GPU or with a smaller reranker).

End-to-end answer quality (needs an LLM key, which spends tokens):

```bash
make eval-answer   # accuracy, citation rate, out-of-scope abstention, p50/p95 latency, cost, faithfulness
```

## Design notes

- **Why mask PII before the agent instead of inside it?** Masking at the boundary is one choke point covering the LLM, cache, checkpointer, logs and audit. Account numbers are masked only after a cue word ("STK", "số tài khoản"), so amounts like `50000000 VND` survive.
- **Why a regex guardrail and not an LLM classifier?** It is free, adds under 1 ms and is deterministic. It is only the first layer: retrieved content is also fenced as data, and the agent has **no write tools**, so a successful injection can at worst produce a wrong answer, never a transaction. An LLM classifier can be added as a middleware later.
- **Why cache only first-turn questions?** A follow-up like "what about Platinum?" depends on context, so caching it would return wrong answers. On a cache hit the Q&A is still written into the thread memory so follow-ups keep working. Uncited or "I don't know" answers are never cached.
- **Why RRF?** Cosine similarity and `ts_rank` scores are not on comparable scales. Rank-based fusion needs no score calibration.
- **Why fail fast on non-transient errors?** Retrying a 400/401 only triples latency. Only 429/5xx/timeouts are retried.

## Project layout

```
app/
  agent/      graph.py (create_agent + middleware), service.py (guard → PII → cache → stream → audit), prompts.py
  rag/        ingest.py (loaders + chunking), store.py (pgvector + FTS), retriever.py (RRF), rerank.py, embeddings.py
  security/   pii.py, guard.py, auth.py, ratelimit.py
  core/       config.py, db.py, cache.py, audit.py, telemetry.py
  api/        chat.py, documents.py
evals/        dataset.jsonl, run_eval.py
tests/        32 unit/API tests: a scripted fake LLM drives the real agent graph, no network needed
```

## Development

```bash
uv sync
make test      # pytest
make lint      # ruff
docker compose up -d postgres redis
uv run uvicorn app.main:app --reload
```

## Roadmap

This service is the knowledge layer of a larger multi-agent banking assistant:

- [ ] Multi-agent supervisor (LangGraph): knowledge agent (this) + account agent + transfer agent
- [ ] Core-banking mock service in **Go** over **gRPC** for balances and transactions
- [ ] Human-in-the-loop confirmation (OTP) for any write action (`HumanInTheLoopMiddleware`)
- [ ] Kubernetes manifests / Helm with HPA on the API, Grafana dashboard
- [ ] JWT/OIDC auth, LLM-based injection classifier, Langfuse tracing
