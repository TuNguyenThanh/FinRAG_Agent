import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import FileResponse, Response

from app.api import chat, documents
from app.core.config import get_settings
from app.services import Services, build_services


def create_app(services: Services | None = None) -> FastAPI:
    """`services` is injected by tests; production builds real infra in the lifespan."""
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.services = services or await build_services(settings)
        try:
            yield
        finally:
            await app.state.services.aclose()

    app = FastAPI(
        title="FinRAG Agent",
        version="0.1.0",
        description="RAG Q&A agent over banking documents — LangChain v1, pgvector, FastAPI.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )
    app.include_router(chat.router)
    app.include_router(documents.router)

    @app.get("/", include_in_schema=False)
    async def demo_ui():
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.get("/healthz", tags=["ops"])
    async def healthz():
        return {"status": "ok"}

    @app.get("/metrics", tags=["ops"], include_in_schema=False)
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
