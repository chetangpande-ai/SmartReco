"""The schema. Fourteen tables: thirteen related, plus a standalone embedding cache.

    users ─┬─1:1─ user_profiles
           ├─1:1─ career_profiles
           ├─1:N─ career_plans
           ├─1:N─ enrollments ────N:1─┐
           ├─1:N─ events ────────N:1─ products ─┬─1:N─ vector_outbox
           ├─1:N─ recommendations ─1:N─ recommendation_items ─N:1─┘
           ├─1:N─ agent_runs        └──1:N─ course_skills
           └─1:N─ notifications

`embedding_cache` deliberately joins to nothing: it is keyed by (embedder, text), not by
product, so re-seeding the catalogue or re-running a search reuses vectors already paid
for even when the row that first needed them is gone.

**Skills are slugs, not rows.** `course_skills.skill` holds a canonical slug from
`app.taxonomy`, which owns the vocabulary as versioned reference data. A `skills` table
would be a second copy of that vocabulary, kept in sync by hand, whose only job would be
to hand back the name the taxonomy already knows. The join table earns its place because
"which courses teach RAG?" has to be an indexed query; the vocabulary does not.
"""

from datetime import UTC, datetime, timedelta

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
    return datetime.now(UTC)


def ensure_utc(dt: datetime | None) -> datetime | None:
    """SQLite has no native timezone type, so DateTime(timezone=True) columns come back
    naive there and aware on Postgres. Every datetime we store is UTC, so re-attaching
    the tzinfo on read makes arithmetic work identically on both backends. Without this,
    `utcnow() - row.created_at` raises TypeError on SQLite only — the worst kind of bug,
    one that passes CI on Postgres and dies in the demo."""
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


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
    # beginner | intermediate | advanced — the progression ladder the agent reasons
    # about when it decides whether someone has outgrown the introductory material.
    tier: Mapped[str] = mapped_column(String(24), default="beginner")
    tags: Mapped[list] = mapped_column(JSON, default=list)
    price_cents: Mapped[int] = mapped_column(Integer, default=0)
    currency: Mapped[str] = mapped_column(String(3), default="USD")
    brand: Mapped[str] = mapped_column(String(120), default="")
    spec: Mapped[str] = mapped_column(String(200), default="")
    rating: Mapped[float] = mapped_column(Float, default=0.0)
    reviews: Mapped[int] = mapped_column(Integer, default=0)
    is_published: Mapped[bool] = mapped_column(Boolean, default=True, index=True)

    # ---- marketplace metadata ----
    # `category` above is a taxonomy category slug; this narrows it one level. Both are
    # stored rather than derived because a course belongs to exactly one place in the
    # tree and walking the tree to find it on every card render would be absurd.
    subcategory: Mapped[str] = mapped_column(String(80), default="", index=True)
    # One of taxonomy.program_types — Bootcamp, Project, Professional Certificate, …
    # This is what separates the marketplace's shelves from one another.
    format: Mapped[str] = mapped_column(String(40), default="Self-Paced Course", index=True)
    duration_hours: Mapped[int] = mapped_column(Integer, default=0)
    language: Mapped[str] = mapped_column(String(24), default="English")
    delivery_mode: Mapped[str] = mapped_column(String(24), default="Self-Paced")
    certificate: Mapped[bool] = mapped_column(Boolean, default=False)
    instructor: Mapped[str] = mapped_column(String(120), default="")
    learners: Mapped[int] = mapped_column(Integer, default=0)
    objectives: Mapped[list] = mapped_column(JSON, default=list)  # "what you'll learn"
    curriculum: Mapped[list] = mapped_column(JSON, default=list)  # [{title, lessons: []}]
    projects: Mapped[list] = mapped_column(JSON, default=list)
    labs: Mapped[int] = mapped_column(Integer, default=0)
    assessments: Mapped[int] = mapped_column(Integer, default=0)

    skills: Mapped[list["CourseSkill"]] = relationship(
        back_populates="product", cascade="all, delete-orphan", order_by="CourseSkill.position"
    )

    @property
    def teaches(self) -> list[str]:
        return [s.skill for s in self.skills if s.kind == "teaches"]

    @property
    def requires(self) -> list[str]:
        return [s.skill for s in self.skills if s.kind == "requires"]

    @property
    def is_free(self) -> bool:
        return self.price_cents == 0

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
        """What gets embedded. Title is repeated because it carries the most signal, and
        the provider is included so "deeplearning.ai course" matches without an exact
        title hit.

        Deliberately built from columns only. Reaching into `self.skills` here would put
        a lazy load inside `compute_content_hash` and inside retrieval's per-candidate
        loop, and the skill names are already carried by `tags` and the title anyway.
        """
        tags = " ".join(self.tags or [])
        return (
            f"{self.title}. {self.title}. {self.brand}. Track: {self.category}. "
            f"Level: {self.tier}. Topics: {tags}. {self.spec}. {self.description}"
        )


class CourseSkill(Base):
    """What a course teaches, and what it assumes you already know.

    The edge that makes the catalogue reason-able: skill-gap analysis produces a set of
    slugs, and this is what turns that set back into courses. `skill` is a canonical
    slug owned by `app.taxonomy` — see the module docstring for why there is no `skills`
    table behind it.
    """

    __tablename__ = "course_skills"
    __table_args__ = (
        UniqueConstraint("product_id", "skill", "kind", name="uq_course_skill"),
        Index("ix_course_skills_skill", "skill", "kind"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    product_id: Mapped[int] = mapped_column(
        ForeignKey("products.id", ondelete="CASCADE"), index=True
    )
    skill: Mapped[str] = mapped_column(String(80))
    kind: Mapped[str] = mapped_column(String(16), default="teaches")  # teaches | requires
    position: Mapped[int] = mapped_column(Integer, default=0)

    product: Mapped["Product"] = relationship(back_populates="skills")


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
    tier_affinity: Mapped[str] = mapped_column(String(24), default="")
    brand_scores: Mapped[dict] = mapped_column(JSON, default=dict)

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


class Enrollment(Base):
    """A course the learner actually started.

    Separate from the `enroll` *event* on purpose: the event is an immutable record that
    a click happened, and this is mutable state that changes every time they make
    progress. Deriving "62% through, two lessons left" from an append-only event log
    would mean replaying it on every dashboard render.
    """

    __tablename__ = "enrollments"
    __table_args__ = (UniqueConstraint("user_id", "product_id", name="uq_enrollment"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id", ondelete="CASCADE"))

    status: Mapped[str] = mapped_column(String(16), default="active")  # active|completed|saved
    progress_pct: Mapped[int] = mapped_column(Integer, default=0)
    # Issued only on completion, and only when the course offers one. Rendered as the
    # certificate id, so it has to be stable once written.
    certificate_code: Mapped[str] = mapped_column(String(32), default="")

    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_activity_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    product: Mapped["Product"] = relationship()


class CareerProfile(Base):
    """What the learner told the AI Career Advisor about themselves.

    Kept apart from `user_profiles`, which is inferred from behaviour and rewritten on
    every event batch. This is *stated* intent: it changes only when someone edits it,
    it outranks behaviour when the two disagree, and overwriting it with a decayed score
    would be a bug. Two sources of truth about a person, deliberately not merged.
    """

    __tablename__ = "career_profiles"

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    current_role: Mapped[str] = mapped_column(String(120), default="")
    target_role: Mapped[str] = mapped_column(String(80), default="")  # taxonomy role slug
    years_experience: Mapped[int] = mapped_column(Integer, default=0)

    skills: Mapped[list] = mapped_column(JSON, default=list)  # canonical slugs
    # What they typed that the taxonomy did not recognise. Kept rather than dropped: it
    # is real information about them, and showing it back is how they find out we did
    # not understand it instead of quietly assuming we did.
    extra_skills: Mapped[list] = mapped_column(JSON, default=list)
    interests: Mapped[str] = mapped_column(String(300), default="")
    weekly_hours: Mapped[int] = mapped_column(Integer, default=0)
    level: Mapped[str] = mapped_column(String(24), default="")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    user: Mapped["User"] = relationship()


class CareerPlan(Base):
    """One generated roadmap: from where they are to the role they named.

    `stages` is the whole composed roadmap as JSON rather than a table of rows, because
    it is a document that is written once and read whole. Normalising it would buy a
    join nobody needs and cost the ability to keep an old plan renderable after the
    catalogue moves underneath it.
    """

    __tablename__ = "career_plans"
    __table_args__ = (Index("ix_plans_user_current", "user_id", "is_current"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)

    target_role: Mapped[str] = mapped_column(String(80), default="")
    path_slug: Mapped[str] = mapped_column(String(80), default="")

    headline: Mapped[str] = mapped_column(String(200), default="")
    narrative: Mapped[str] = mapped_column(Text, default="")
    have: Mapped[list] = mapped_column(JSON, default=list)  # skill slugs already held
    gaps: Mapped[list] = mapped_column(JSON, default=list)  # skill slugs still missing
    stages: Mapped[list] = mapped_column(JSON, default=list)  # [{key, title, items: []}]

    strategy: Mapped[str] = mapped_column(String(24), default="agentic")  # agentic|deterministic
    model: Mapped[str] = mapped_column(String(80), default="")
    readiness: Mapped[float] = mapped_column(Float, default=0.0)  # 0-1, share of role skills held

    is_current: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)

    user: Mapped["User"] = relationship()


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
    # {"grade": "v1", "generate": "v1"} — which prompts.py constant produced this run's
    # copy. Nullable because rows written before this column existed have none.
    prompt_versions: Mapped[dict | None] = mapped_column(JSON, nullable=True)
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
