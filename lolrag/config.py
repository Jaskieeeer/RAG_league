from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration, read from environment and .env.

    Args:
        ddragon_version: Pinned Data Dragon patch version, e.g. "16.14.1".
        ddragon_locale: Data Dragon locale code.
        ddragon_base_url: Data Dragon CDN root URL.
        embedding_model_name: HuggingFace embedding model identifier.
        chroma_persist_dir: Filesystem path for the persistent Chroma store.
        chroma_collection_name: Chroma collection name.
        retriever_k: Number of documents the retriever returns per query.
        google_api_key: Gemini API key, kept out of logs and reprs.
        llm_model_name: Gemini chat model identifier.
        llm_fallback_model_name: Gemini chat model identifier used as fallback.
        llm_temperature: Sampling temperature for the LLM.
        langsmith_tracing: Whether to enable LangSmith tracing at runtime.
        langsmith_api_key: LangSmith API key, kept out of logs and reprs.
        langsmith_project: LangSmith project name traces are grouped under.
        eval_judge_model_name: Gemini chat model identifier used by the faithfulness judge.
        eval_report_dir: Filesystem path where evaluation reports are written.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="forbid")

    ddragon_version: str
    ddragon_locale: str = "en_US"
    ddragon_base_url: str = "https://ddragon.leagueoflegends.com"

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    chroma_persist_dir: str = "./data/chroma"
    chroma_collection_name: str = "lol_champions"

    retriever_k: int = 4

    google_api_key: SecretStr | None = None
    llm_model_name: str = "gemini-3.5-flash"
    llm_fallback_model_name: str = "gemini-3.1-flash-lite"
    llm_temperature: float = 0.0

    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "lolrag-eval"
    eval_judge_model_name: str = "gemini-3.5-flash"
    eval_report_dir: str = "./eval_reports"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide cached Settings instance.

    Returns:
        Settings loaded from environment variables and .env.
    """
    return Settings()
