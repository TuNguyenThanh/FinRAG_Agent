import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.service import ChatEvent
from app.api.deps import rate_limited_user
from app.services import Services, get_services

router = APIRouter(prefix="/v1", tags=["chat"])


class ChatRequest(BaseModel):
    message: str = Field(
        ..., min_length=1, examples=["Phí thường niên thẻ tín dụng Gold là bao nhiêu?"]
    )
    thread_id: str | None = Field(None, max_length=64, pattern=r"^[\w-]+$")
    stream: bool = True


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(5, ge=1, le=20)
    doc_name: str | None = None


async def _sse(events: AsyncIterator[ChatEvent]) -> AsyncIterator[str]:
    async for ev in events:
        yield f"event: {ev.event}\ndata: {json.dumps(ev.data, ensure_ascii=False)}\n\n"


@router.post("/chat", summary="Ask a question (SSE stream or JSON)")
async def chat(
    req: ChatRequest,
    user: str = Depends(rate_limited_user),
    services: Services = Depends(get_services),
):
    if not req.stream:
        return await services.chat.answer(user, req.message, req.thread_id)
    return StreamingResponse(
        _sse(services.chat.stream(user, req.message, req.thread_id)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/search", summary="Debug: inspect hybrid retrieval results")
async def search(
    req: SearchRequest,
    _: str = Depends(rate_limited_user),
    services: Services = Depends(get_services),
):
    chunks = await services.retriever.search(req.query, doc_name=req.doc_name, top_k=req.top_k)
    return {"results": [c.to_source() for c in chunks]}
