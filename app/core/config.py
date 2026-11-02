from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration, loaded from environment variables / `.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"

    # --- LLM -------------------------------------------------------------
    # Any `init_chat_model` string: "anthropic:claude-sonnet-5-5", "openai:gpt-...",
    # "google_genai:gemini-...". The matching provider package + API key must be installed/set.
    llm_model: str = "anthropic:claude-sonnet-5-5"
    # Unset by default: some newer models reject non-default sampling parameters.
    llm_temperature: float | None = None
    llm_max_tokens: int = 1024
    # USD per 1M tokens, used only for cost telemetry. Set these to your provider's price list.
    llm_price_input_per_mtok: float = 0.0
    llm_price_output_per_mtok: float = 0.0

    # --- Storage ---------------------------------------------------------
    database_url: str = "postgresql://finrag:finrag@localhost:5433/finrag"
    redis_url: str = "redis://localhost:6379/0"
    checkpointer: Literal["postgres", "memory"] = "postgres"

    # --- Retrieval -------------------------------------------------------
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    rerank_enabled: bool = False  # +3.7pt hit@1 but ~1.7s/query on CPU, see README
    rerank_model: str = "jinaai/jina-reranker-v2-base-multilingual"
    chunk_size: int = 800
    chunk_overlap: int = 120
    retrieval_candidates: int = 20  # per retriever (vector / keyword) before fusion
    retrieval_top_k: int = 5

    # --- Semantic cache --------------------------------------------------
    cache_enabled: bool = True
    cache_similarity_threshold: float = 0.93
    cache_ttl_seconds: int = 3600
    cache_max_entries: int = 1000

    # --- Security --------------------------------------------------------
    # "key:user_id,key2:user_id2". Never commit real keys.
    api_keys: str = "dev-key:demo-user"
    rate_limit_per_minute: int = 30
    max_message_chars: int = 2000
    max_upload_mb: int = 20

    # --- Agent guardrails ------------------------------------------------
    max_model_calls_per_run: int = 6
    max_tool_calls_per_run: int = 4

    def api_key_map(self) -> dict[str, str]:
        pairs = (p.split(":", 1) for p in self.api_keys.split(",") if ":" in p)
        return {k.strip(): v.strip() for k, v in pairs}


@lru_cache
def get_settings() -> Settings:
    return Settings()
