# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    FASTEMBED_CACHE_PATH=/models PYTHONUNBUFFERED=1
WORKDIR /app

# Dependencies first for layer caching.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project

# Bake ONNX models into the image so pods start fast and need no internet at runtime.
ARG EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
ARG RERANK_MODEL=jinaai/jina-reranker-v2-base-multilingual
ARG PRELOAD_RERANKER=0
RUN uv run --no-sync python -c "from fastembed import TextEmbedding; TextEmbedding('${EMBEDDING_MODEL}')" \
 && if [ "$PRELOAD_RERANKER" = "1" ]; then uv run --no-sync python -c \
    "from fastembed.rerank.cross_encoder import TextCrossEncoder; TextCrossEncoder('${RERANK_MODEL}')"; fi

COPY app ./app
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

RUN useradd --uid 10001 --no-create-home appuser && chown -R appuser /models
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"
CMD ["/app/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
