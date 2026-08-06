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
    base_url: str = "http://localhost:8000"  # absolute links in emails
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
    # means every course becomes a candidate and rank fusion ends up ranking noise.
    # Cut relative to the best hit so the floor travels across embedding models
    # instead of being tuned to one, with a small absolute floor underneath.
    #
    # Swept against the live index over 20 paraphrase probes (`make sweep`). On the course
    # catalogue quality is *flat* across the whole range — the fused ranking is already
    # right, so the floor only controls how much noise MMR has to diversify over:
    #     ratio  0.00  0.35  0.50  0.55  0.60  0.70  0.80
    #     r@1    0.95  0.95  0.95  0.95  0.95  0.95  0.95
    #     r@5    1.00  1.00  1.00  1.00  1.00  1.00  1.00
    #     pool   32.8  26.0  13.3  10.0   7.5   3.9   1.9
    # So the floor is chosen on pool size, not recall: `retrieve` asks for 8 candidates,
    # and a pool of 10 gives MMR a real choice while 0.60 and tighter starves it below
    # what was requested. Tuned on one catalogue — re-run `make sweep` if it changes shape.
    retrieval_score_ratio: float = 0.55
    retrieval_min_score: float = 0.10
    retrieval_grade_threshold: float = 0.60
    retrieval_max_refines: int = 2
    retrieval_rerank: bool = True

    # ---- Ranking (app/agent/ranking.py)
    # Confidence-gap the heuristic ranker needs over the runner-up before grade() skips
    # the LLM call entirely and trusts the heuristic ranking as-is. Hand-picked, not
    # swept — there's no labelled outcome data yet to sweep it against.
    ranker_skip_margin: float = 0.25
    # Probability grade()'s candidate order gets one slot swapped for a low-exposure
    # candidate instead of the ranker's actual top pick — see ranking.apply_exploration_slot.
    explore_epsilon: float = 0.15

    # ---- Email digest
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_starttls: bool = True
    mail_from: str = "SmartReco <no-reply@smartreco.local>"
    mail_backend: str = "auto"  # auto | smtp | file | ses
    aws_region: str = "us-east-1"
    digest_hour: int = 16
    digest_minute: int = 0
    digest_min_events: int = 3  # don't email someone who barely visited
    scheduler_enabled: bool = True
    outbox_interval_seconds: int = 60
    reconcile_interval_minutes: int = 60

    # ---- Guardrails. The deterministic rails always run; NeMo costs tokens per
    # generation, so it is opt-in.
    guardrails_llm_enabled: bool = False

    # ---- Agent
    agent_mode: str = "langgraph"  # langgraph | deepagents

    # ---- Observability. LangSmith traces the agent graph; Logfire traces the request
    # around it (HTTP, SQL, Mesh) and can receive the graph too — see app/observability.py.
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "smartreco"

    logfire_enabled: bool = False
    logfire_token: str = ""
    logfire_project_name: str = "smartreco"
    logfire_console: bool = False  # print spans to stdout instead of shipping them
    logfire_instrument_langchain: bool = True

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
