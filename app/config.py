"""All tunables in one place. Read once at import, validated at boot."""

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: str = "development"
    secret_key: str = "dev-only-change-me"
    log_level: str = "INFO"
    log_json: bool = False

    # ---- Mesh API. MESH_API_KEY is accepted too: it is the secret name the
    # hackathon brief asks you to set in GitHub Actions.
    meshapi_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("MESHAPI_API_KEY", "MESH_API_KEY"),
    )
    meshapi_base_url: str = "https://api.meshapi.ai/v1"
    meshapi_model: str = "openai/gpt-4o-mini"
    meshapi_model_fast: str = "openai/gpt-4o-mini"
    meshapi_embedding_model: str = "openai/text-embedding-3-small"
    meshapi_timeout_seconds: float = 90.0

    # Reasoning models (kimi-k3, gpt-5, o-series) burn hidden reasoning tokens before
    # emitting a single visible character. Measured: kimi-k3 spent 220 reasoning tokens
    # on a 396-token completion. A 512-token ceiling silently returns "".
    llm_max_output_tokens: int = 4000
    llm_enabled: bool = True
    llm_daily_budget_usd: float = 1.00
    llm_max_calls_per_user_per_day: int = 25

    # ---- Storage
    database_url: str = "sqlite+pysqlite:///./data/smartreco.db"
    vector_backend: str = "chroma"  # chroma | pinecone
    chroma_dir: str = "./data/chroma"
    embedding_dim: int = 1536

    pinecone_api_key: str = ""
    pinecone_index: str = "smartreco"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_namespace: str = ""

    # ---- Event ingest. Flush whichever comes first.
    event_queue_max: int = 20_000
    event_flush_batch: int = 500
    event_flush_interval_seconds: float = 1.0
    event_max_batch_size: int = 100

    # ---- Behavior profile
    profile_halflife_hours: float = 48.0
    profile_lookback_days: int = 30
    profile_max_events: int = 400

    # ---- Trigger policy: every gate that decides whether an LLM call happens.
    rec_min_events: int = 5
    rec_cooldown_seconds: int = 90
    # Minimum interest drift (1 - cosine between the current interest centroid and the
    # one the last recommendation was built from) needed to justify spending a call.
    # 0.10 ~ "the user has moved on to something meaningfully different".
    rec_drift_threshold: float = 0.10
    rec_staleness_events: int = 15
    rec_ttl_minutes: int = 120
    rec_top_k: int = 4

    # ---- Retrieval
    retrieval_candidates: int = 40
    retrieval_mmr_lambda: float = 0.65
    # A kNN search always returns k results, however bad — on a small catalogue that
    # means every product becomes a candidate and rank fusion ends up ranking noise.
    # Cut relative to the best hit so the floor travels across embedding models
    # instead of being tuned to one, with a small absolute floor underneath.
    #
    # Swept against the live index over 10 paraphrase probes (scripts/eval_retrieval.py):
    # recall@1 held at 7/10 and recall@5 at 8/10 for every ratio from 0.0 to 0.6, while
    # the candidate pool fell from 31.7 to 5.0. So this buys precision and latency at no
    # cost in recall — 0.35 is the middle of that flat region rather than its edge,
    # which leaves headroom for catalogues whose score distributions differ.
    retrieval_score_ratio: float = 0.35
    retrieval_min_score: float = 0.10
    retrieval_grade_threshold: float = 0.60
    retrieval_max_refines: int = 2
    retrieval_rerank: bool = True

    # ---- Email digest
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_starttls: bool = True
    mail_from: str = "SmartReco <no-reply@smartreco.local>"
    digest_hour: int = 16
    digest_minute: int = 0
    scheduler_enabled: bool = True

    # ---- Guardrails. The deterministic rails always run; NeMo costs tokens per
    # generation, so it is opt-in.
    guardrails_llm_enabled: bool = False

    # ---- Agent
    agent_mode: str = "langgraph"  # langgraph | deepagents

    # ---- Observability
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "smartreco"

    @field_validator("secret_key")
    @classmethod
    def _reject_default_secret_in_prod(cls, v: str, info) -> str:
        if info.data.get("app_env") == "production" and v.startswith("dev-only"):
            raise ValueError("SECRET_KEY must be set in production")
        return v

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def data_dir(self) -> Path:
        return ROOT / "data"

    @property
    def guardrails_config_dir(self) -> Path:
        return ROOT / "app" / "guardrails"

    @property
    def has_llm(self) -> bool:
        """False => the agent uses its deterministic path and spends nothing."""
        return self.llm_enabled and bool(self.meshapi_api_key)

    @property
    def has_smtp(self) -> bool:
        return bool(self.smtp_host)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    return s


settings = get_settings()
