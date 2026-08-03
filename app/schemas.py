"""Request/response contracts. Validation happens here, at the edge, so the rest of
the app can trust its inputs."""

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field, field_validator

# Only these reach the profile builder. An unknown type from a stale cached tracker.js
# is dropped rather than silently polluting the interest scores.
EVENT_TYPES = frozenset(
    {
        "page_view",
        "product_view",
        "product_click",
        "search",
        "search_result_click",
        "filter",
        "dwell",
        "scroll_depth",
        "enroll",
        "wishlist",
        # Retired names for enroll/wishlist. Kept accepted rather than rejected: a
        # tracker.js sitting in someone's browser cache goes on sending them for days,
        # and profile.EVENT_WEIGHTS still scores them identically.
        "add_to_cart",
        "purchase",
        "rec_impression",
        "rec_click",
    }
)


class EventIn(BaseModel):
    type: str
    product_id: int | None = None
    product_slug: str | None = Field(default=None, max_length=160)
    query: str | None = Field(default=None, max_length=300)
    path: str = Field(default="", max_length=300)
    dwell_ms: int = Field(default=0, ge=0, le=3_600_000)
    position: int = Field(default=0, ge=0, le=10_000)
    meta: dict = Field(default_factory=dict)
    session_id: str = Field(default="", max_length=64)
    idem: str = Field(default="", max_length=64)
    ts: int | None = None  # client epoch milliseconds

    @field_validator("type")
    @classmethod
    def _known_type(cls, v: str) -> str:
        if v not in EVENT_TYPES:
            raise ValueError(f"unknown event type: {v}")
        return v

    @field_validator("meta")
    @classmethod
    def _bounded_meta(cls, v: dict) -> dict:
        # Truncate rather than reject: a tracker bug should not be able to write
        # megabytes into the events table, but it also should not cost us the event.
        return {k: v[k] for k in list(v)[:20]}


class EventBatchIn(BaseModel):
    """Events arrive as raw dicts and are validated one at a time in the router.

    Validating the batch as `list[EventIn]` would mean a single malformed event — a
    retired `type` from a tracker.js still sitting in someone's browser cache — returns
    422 and discards up to 99 perfectly good events with it. Telemetry has to degrade
    per-event, not per-batch.
    """

    events: list[dict] = Field(min_length=1, max_length=100)
    anon_id: str = Field(default="", max_length=64)


class EventBatchAccepted(BaseModel):
    accepted: int
    rejected: int = 0
    queued: bool = True


# The learner's level. Ordered, because the agent reasons about progression: someone
# consistently opening advanced material is not shopping the introductory shelf.
TIERS = ("beginner", "intermediate", "advanced")


class ProductIn(BaseModel):
    title: str = Field(min_length=3, max_length=200)
    description: str = Field(default="", max_length=8000)
    category: str = Field(min_length=2, max_length=80)
    tier: str = Field(default="beginner")
    tags: list[str] = Field(default_factory=list)
    price_cents: int = Field(default=0, ge=0, le=1_000_000_00)
    brand: str = Field(default="", max_length=120)
    spec: str = Field(default="", max_length=200)
    rating: float = Field(default=0.0, ge=0.0, le=5.0)
    is_published: bool = True

    @field_validator("tier")
    @classmethod
    def _known_tier(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in TIERS:
            raise ValueError(f"tier must be one of {', '.join(TIERS)}")
        return v

    @field_validator("tags")
    @classmethod
    def _clean_tags(cls, v: list[str]) -> list[str]:
        return [t.strip().lower() for t in v if t.strip()][:12]


class ProductOut(BaseModel):
    id: int
    slug: str
    title: str
    description: str
    category: str
    tier: str
    tags: list[str]
    price_cents: int
    brand: str
    spec: str
    rating: float
    is_published: bool
    vector_synced_at: datetime | None

    model_config = {"from_attributes": True}


class RecommendationItemOut(BaseModel):
    product: ProductOut
    rank: int
    score: float
    reason: str

    model_config = {"from_attributes": True}


class RecommendationOut(BaseModel):
    id: int
    headline: str
    narrative: str
    cta: str
    strategy: str
    model: str
    trigger_reason: str
    confidence: float
    created_at: datetime
    items: list[RecommendationItemOut]

    model_config = {"from_attributes": True}


class RegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    name: str = Field(default="", max_length=120)


class LoginIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)
