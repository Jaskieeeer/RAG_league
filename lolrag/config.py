from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration, read from environment and .env.

    Args:
        ddragon_version: Pinned Data Dragon patch version, e.g. "16.14.1".
        ddragon_locale: Data Dragon locale code.
        ddragon_base_url: Data Dragon CDN root URL.
        cdragon_base_url: Community Dragon raw asset root URL.
        universe_base_url: Riot Universe JSON API root URL, including locale segment.
        riot_static_base_url: Riot developer static documentation root URL,
            serving the reference tables the game CDNs do not carry.
        cache_dir: Filesystem path for the on-disk raw response cache.
        http_user_agent: User-Agent header sent with every corpus fetch.
        http_concurrency: Maximum number of in-flight corpus requests.
        http_delay_seconds: Delay applied before each request inside the concurrency limit.
        http_timeout_seconds: Per-request timeout for corpus fetches.
        http_max_retries: Maximum attempts per request before giving up.
        embedding_model_name: HuggingFace embedding model identifier.
        retriever_k: Number of documents the retriever returns per query.
        google_api_key: Gemini API key, kept out of logs and reprs.
        llm_model_name: Gemini chat model identifier.
        llm_fallback_model_name: Gemini chat model identifier used as fallback.
        llm_temperature: Sampling temperature for the LLM.
        langsmith_tracing: Whether to enable LangSmith tracing at runtime.
        langsmith_api_key: LangSmith API key, kept out of logs and reprs.
        langsmith_project: LangSmith project name traces are grouped under.
        eval_model_name: Gemini chat model identifier the evaluation harness
            generates answers with, for both the pipeline and the no-retrieval
            baseline. Held apart from llm_model_name so an eval run can be
            priced and rate-limited independently of what the product answers
            with, and shared by both systems so the comparison stays honest.
        eval_judge_model_name: Gemini chat model identifier used by the evaluation judges.
        eval_report_dir: Filesystem path where evaluation reports are written.
        eval_agreement_sample_size: Number of entailment judgements dumped for
            hand-scoring, which is what the judge-human agreement figure is
            measured over.
        database_url: SQLAlchemy connection URL for the Postgres/pgvector database.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="forbid")

    ddragon_version: str
    ddragon_locale: str = "en_US"
    ddragon_base_url: str = "https://ddragon.leagueoflegends.com"
    cdragon_base_url: str = "https://raw.communitydragon.org"
    universe_base_url: str = "https://universe-meeps.leagueoflegends.com/v1/en_us"
    riot_static_base_url: str = "https://static.developer.riotgames.com"

    cache_dir: str = "./data/cache"
    http_user_agent: str = "lolrag/0.1 (+https://github.com/Jaskieeeer)"
    http_concurrency: int = 5
    http_delay_seconds: float = 0.1
    http_timeout_seconds: float = 30.0
    http_max_retries: int = 3

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    retriever_k: int = 4

    google_api_key: SecretStr | None = None
    llm_model_name: str = "gemini-3.5-flash"
    llm_fallback_model_name: str = "gemini-3.1-flash-lite"
    llm_temperature: float = 0.0

    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "lolrag-eval"
    eval_model_name: str = "gemini-3.1-flash-lite"
    eval_judge_model_name: str = "gemini-3.1-flash-lite"
    eval_report_dir: str = "./eval_reports"
    eval_agreement_sample_size: int = 20

    database_url: str = "postgresql+psycopg://lolrag:lolrag@localhost:5432/lolrag"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide cached Settings instance.

    Returns:
        Settings loaded from environment variables and .env.
    """
    return Settings()
