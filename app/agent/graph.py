"""LangChain v1 agent: model + retrieval tool + middleware + checkpointer memory."""

from html import escape

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
)
from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from langgraph.types import Checkpointer

from app.agent.prompts import SYSTEM_PROMPT
from app.core.config import Settings
from app.rag.retriever import HybridRetriever

_TRANSIENT_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


def is_transient_error(exc: BaseException) -> bool:
    """Retry only what can succeed on a second try: rate limits, overload, 5xx, network.

    Bad requests / auth errors fail fast instead of tripling latency.
    """
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    if isinstance(status, int):
        return status in _TRANSIENT_STATUS
    return isinstance(exc, TimeoutError | ConnectionError) or type(exc).__name__ in {
        "APIConnectionError",
        "APITimeoutError",
    }


def build_search_tool(retriever: HybridRetriever):
    @tool(response_format="content_and_artifact")
    async def search_documents(query: str, doc_name: str | None = None) -> tuple[str, list[dict]]:
        """Tìm kiếm trong tài liệu nội bộ của DemoBank (biểu phí, điều khoản thẻ, quy định
        cho vay, hướng dẫn eKYC...). Trả về các đoạn liên quan kèm trích dẫn.

        Args:
            query: Câu truy vấn ngắn gọn, cụ thể (vd: "phí thường niên thẻ tín dụng Gold").
            doc_name: (tùy chọn) Giới hạn tìm trong một tài liệu, vd "bieu-phi-dich-vu.md".
        """
        chunks = await retriever.search(query, doc_name=doc_name)
        if not chunks:
            return "Không tìm thấy đoạn tài liệu nào phù hợp.", []
        # Fence retrieved text as untrusted data (indirect prompt-injection defence).
        body = "\n\n".join(
            f'<document citation="[{escape(c.citation())}]">\n{escape(c.content)}\n</document>'
            for c in chunks
        )
        return body, [c.to_source() for c in chunks]

    return search_documents


def build_agent(
    model: BaseChatModel,
    retriever: HybridRetriever,
    checkpointer: Checkpointer,
    settings: Settings,
):
    return create_agent(
        model=model,
        tools=[build_search_tool(retriever)],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
        middleware=[
            # Bounded cost/latency: a runaway tool loop can't burn tokens indefinitely.
            ModelCallLimitMiddleware(
                run_limit=settings.max_model_calls_per_run, exit_behavior="end"
            ),
            ToolCallLimitMiddleware(run_limit=settings.max_tool_calls_per_run),
            # Errors propagate to ChatService, which returns a generic message: provider
            # error text must never be streamed to customers.
            ModelRetryMiddleware(
                max_retries=2, initial_delay=0.5, retry_on=is_transient_error, on_failure="error"
            ),
            # Long conversations: summarise old turns instead of growing the prompt forever.
            SummarizationMiddleware(model=model, trigger=("messages", 30), keep=("messages", 12)),
        ],
    )
