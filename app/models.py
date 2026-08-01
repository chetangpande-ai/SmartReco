"""The schema. Nine tables, all related.

    users ─┬─1:1─ user_profiles
           ├─1:N─ events ──N:1─ products ─1:N─ vector_outbox
           ├─1:N─ recommendations ─1:N─ recommendation_items ─N:1─ products
           ├─1:N─ agent_runs
           └─1:N─ notifications
"""

from datetime import datetime, timedelta, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def ensure_utc(dt: datetime | None) -> datetime | None:
    """SQLite has no native timezone type, so DateTime(timezone=True) columns come back
    naive there and aware on Postgres. Every datetime we store is UTC, so re-attaching
    the tzinfo on read makes arithmetic work identically on both backends. Without this,
    `utcnow() - row.created_at` raises TypeError on SQLite only — the worst kind of bug,
    one that passes CI on Postgres and dies in the demo."""
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(128))
    name: Mapped[str] = mapped_column(String(120), default="")
    role: Mapped[str] = mapped_column(String(16), default="user")  # user | admin

    digest_opt_in: Mapped[bool] = mapped_column(Boolean, default=True)
    last_digest_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    profile: Mapped["UserProfile"] = relationship(
        back_populates="user", uselist=False, cascade="all, delete-orphan"
    )
    events: Mapped[list["Event"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    recommendations: Mapped[list["Recommendation"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


class Product(Base):
    __tablename__ = "products"

    id: Mapped[int] = mapped_column(primary_key=True)
    slug: Mapped[str] = mapped_column(String(160), unique=True, index=True)
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    category: Mapped[str] = mapped_column(String(80), index=True)
    level: Mapped[str] = mapped_column(String(24), default="beginner")
    tags: Mapped[list] = mapped_column(JSON, default=list)
    price_cents: Mapped[int] = mapped_column(Integer, default=0)
    currency: Mapped[str] = mapped_column(String(3), default="USD")
    instructor: Mapped[str] = mapped_column(String(120), default="")
    duration_minutes: Mapped[int] = mapped_column(Integer, default=0)
    rating: Mapped[float] = mapped_column(Float, default=0.0)
    enrollments: Mapped[int] = mapped_column(Integer, default=0)
    is_published: Mapped[bool] = mapped_column(Boolean, default=True, index=True)

    # content_hash drives the dual-write: if it changed, the vector copy is stale.
    content_hash: Mapped[str] = mapped_column(String(64), default="")
    vector_synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    @property
    def price(self) -> float:
        return self.price_cents / 100

    @property
    def vector_in_sync(self) -> bool:
        return self.vector_synced_at is not None

    def embedding_text(self) -> str:
        """What gets embedded. Title is repeated because it carries the most signal."""
        tags = " ".join(self.tags or [])
        return (
            f"{self.title}. {self.title}. Category: {self.category}. "
            f"Level: {self.level}. Topics: {tags}. {self.description}"
        )


class Event(Base):
    """One tracked user action. Written in bulk, never one at a time."""

    __tablename__ = "events"
    __table_args__ = (
        Index("ix_events_user_time", "user_id", "server_ts"),
        Index("ix_events_anon_time", "anon_id", "server_ts"),
        Index("ix_events_type_time", "type", "server_ts"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    anon_id: Mapped[str] = mapped_column(String(64), default="")
    session_id: Mapped[str] = mapped_column(String(64), default="")

    type: Mapped[str] = mapped_column(String(32))
    product_id: Mapped[int | None] = mapped_column(ForeignKey("products.id", ondelete="SET NULL"))
    query: Mapped[str | None] = mapped_column(String(300))
    path: Mapped[str] = mapped_column(String(300), default="")
    dwell_ms: Mapped[int] = mapped_column(Integer, default=0)
    position: Mapped[int] = mapped_column(Integer, default=0)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)

    # Client-generated. Makes beacon retries free instead of duplicating rows.
    dedupe_key: Mapped[str] = mapped_column(String(64), unique=True)
    client_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    server_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)

    user: Mapped["User"] = relationship(back_populates="events")
    product: Mapped["Product"] = relationship()


class UserProfile(Base):
    """Rolled-up behavior. Recomputed on ingest so the trigger check stays O(1)."""

    __tablename__ = "user_profiles"

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    interests: Mapped[dict] = mapped_column(JSON, default=dict)  # category -> decayed score
    tag_scores: Mapped[dict] = mapped_column(JSON, default=dict)
    top_terms: Mapped[list] = mapped_column(JSON, default=list)
    recent_queries: Mapped[list] = mapped_column(JSON, default=list)
    viewed_product_ids: Mapped[list] = mapped_column(JSON, default=list)

    price_affinity_cents: Mapped[int] = mapped_column(Integer, default=0)
    level_affinity: Mapped[str] = mapped_column(String(24), default="")

    centroid: Mapped[bytes | None] = mapped_column(LargeBinary)  # float32 interest vector
    last_rec_centroid: Mapped[bytes | None] = mapped_column(LargeBinary)
    last_rec_signature: Mapped[str] = mapped_column(String(64), default="")

    events_total: Mapped[int] = mapped_column(Integer, default=0)
    events_since_rec: Mapped[int] = mapped_column(Integer, default=0)
    last_event_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_rec_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    user: Mapped["User"] = relationship(back_populates="profile")


class Recommendation(Base):
    __tablename__ = "recommendations"
    __table_args__ = (Index("ix_recs_user_current", "user_id", "is_current"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)

    headline: Mapped[str] = mapped_column(String(200), default="")
    narrative: Mapped[str] = mapped_column(Text, default="")
    cta: Mapped[str] = mapped_column(String(120), default="")

    strategy: Mapped[str] = mapped_column(String(32), default="agentic")  # agentic|coldstart|fallback
    model: Mapped[str] = mapped_column(String(80), default="")
    trigger_reason: Mapped[str] = mapped_column(String(64), default="")
    behavior_signature: Mapped[str] = mapped_column(String(64), default="", index=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)

    is_current: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    user: Mapped["User"] = relationship(back_populates="recommendations")
    items: Mapped[list["RecommendationItem"]] = relationship(
        back_populates="recommendation",
        cascade="all, delete-orphan",
        order_by="RecommendationItem.rank",
    )

    def is_expired(self, now: datetime | None = None) -> bool:
        if self.expires_at is None:
            return False
        # ensure_utc is not optional here: on SQLite this column reads back naive and
        # comparing it to an aware utcnow() raises TypeError.
        return (now or utcnow()) >= ensure_utc(self.expires_at)


class RecommendationItem(Base):
    """Normalized so click-through can be joined per product later."""

    __tablename__ = "recommendation_items"
    __table_args__ = (UniqueConstraint("recommendation_id", "product_id", name="uq_rec_product"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    recommendation_id: Mapped[int] = mapped_column(
        ForeignKey("recommendations.id", ondelete="CASCADE"), index=True
    )
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id", ondelete="CASCADE"))

    rank: Mapped[int] = mapped_column(Integer, default=0)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    reason: Mapped[str] = mapped_column(Text, default="")

    recommendation: Mapped["Recommendation"] = relationship(back_populates="items")
    product: Mapped["Product"] = relationship()


class AgentRun(Base):
    """One LangGraph execution. This is the observability record behind /admin/agent."""

    __tablename__ = "agent_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    recommendation_id: Mapped[int | None] = mapped_column(
        ForeignKey("recommendations.id", ondelete="SET NULL")
    )

    trigger: Mapped[str] = mapped_column(String(64), default="")
    status: Mapped[str] = mapped_column(String(16), default="ok")  # ok | error | skipped
    node_path: Mapped[list] = mapped_column(JSON, default=list)
    retrieval_stats: Mapped[dict] = mapped_column(JSON, default=dict)
    grade_score: Mapped[float] = mapped_column(Float, default=0.0)
    refine_loops: Mapped[int] = mapped_column(Integer, default=0)

    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    llm_calls: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)

    request_id: Mapped[str] = mapped_column(String(64), default="")
    langsmith_url: Mapped[str] = mapped_column(String(300), default="")
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class VectorOutbox(Base):
    """Transactional outbox. Written in the same commit as the product row, so SQL and
    the vector store can never silently disagree about what happened."""

    __tablename__ = "vector_outbox"
    __table_args__ = (Index("ix_outbox_status_created", "status", "created_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    product_id: Mapped[int] = mapped_column(Integer, index=True)
    op: Mapped[str] = mapped_column(String(16))  # upsert | delete
    payload: Mapped[dict] = mapped_column(JSON, default=dict)

    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending|done|dead
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str] = mapped_column(Text, default="")
    next_attempt_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Notification(Base):
    __tablename__ = "notifications"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    recommendation_id: Mapped[int | None] = mapped_column(
        ForeignKey("recommendations.id", ondelete="SET NULL")
    )

    channel: Mapped[str] = mapped_column(String(16), default="email")
    subject: Mapped[str] = mapped_column(String(200), default="")
    status: Mapped[str] = mapped_column(String(16), default="sent")  # sent | failed | skipped
    error: Mapped[str] = mapped_column(Text, default="")

    # "digest:<user_id>:<YYYY-MM-DD>" — a duplicate scheduler tick cannot double-send.
    dedupe_key: Mapped[str] = mapped_column(String(120), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class EmbeddingCache(Base):
    """Embeddings are pure functions of (model, text) and cost money. Re-seeding the
    catalog or re-running a search should never pay twice."""

    __tablename__ = "embedding_cache"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)  # sha256(embedder_id|text)
    embedder_id: Mapped[str] = mapped_column(String(120), index=True)
    dim: Mapped[int] = mapped_column(Integer)
    vector: Mapped[bytes] = mapped_column(LargeBinary)  # float32
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


def rec_expiry(minutes: int) -> datetime:
    return utcnow() + timedelta(minutes=minutes)
