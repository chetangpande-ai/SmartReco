"""Test environment.

Everything runs offline: LLM_ENABLED=false selects the deterministic hashing embedder
and the template-based copy writer, so the suite is hermetic, free, and identical on a
laptop and in CI with no API key. The Mesh client itself is tested against a stub.

These environment variables must be set before `app.config` is imported anywhere, which
is why they live at module scope in conftest rather than in a fixture.
"""

import os
import shutil
import tempfile

_TMP = tempfile.mkdtemp(prefix="smartreco_tests_")
os.environ.update(
    LLM_ENABLED="false",
    SCHEDULER_ENABLED="false",
    DATABASE_URL=f"sqlite+pysqlite:///{_TMP}/test.db".replace("\\", "/"),
    CHROMA_DIR=f"{_TMP}/chroma",
    EMBEDDING_DIM="256",
    SMTP_HOST="",
    SECRET_KEY="test-secret-key-not-for-production",
    APP_ENV="test",
    LOG_LEVEL="WARNING",
)

import pytest  # noqa: E402
from sqlalchemy import delete  # noqa: E402

from app.db import init_db, session_scope  # noqa: E402
from app.models import (  # noqa: E402
    AgentRun,
    Event,
    Notification,
    Product,
    Recommendation,
    RecommendationItem,
    User,
    UserProfile,
    VectorOutbox,
)
from app.schemas import ProductIn  # noqa: E402
from app.security import hash_password  # noqa: E402

# Small fixed catalogue: big enough for retrieval to be meaningful, small enough that
# the whole suite stays fast. Two brands appear twice so brand affinity is testable.
CATALOG = [
    ("Sony WH-1000XM5 Wireless Headphones", "Sony", "audio", "flagship", 39900,
     ["noise-cancelling", "over-ear", "wireless", "travel"],
     "30h battery · adaptive ANC · Bluetooth 5.2",
     "Eight microphones cancel low-frequency cabin noise well enough that long flights stop being tiring."),
    ("Bose QuietComfort Ultra Headphones", "Bose", "audio", "flagship", 42900,
     ["noise-cancelling", "over-ear", "wireless", "spatial-audio"],
     "24h battery · immersive spatial audio · Bluetooth 5.3",
     "The strongest noise cancellation available, with a head-tracked spatial mode."),
    ("Anker Soundcore Q30 Headphones", "Anker", "audio", "entry", 7900,
     ["noise-cancelling", "over-ear", "wireless", "budget"],
     "40h battery with ANC · hybrid ANC · multipoint",
     "Cancellation good enough for an open-plan office at a fraction of the price."),
    ("Apple AirPods Pro 2 (USB-C)", "Apple", "audio", "mid", 24900,
     ["earbuds", "noise-cancelling", "wireless"],
     "6h buds / 30h case · adaptive audio · USB-C",
     "In-ear cancellation that adapts to your surroundings in real time."),
    ("Apple iPhone 15 Pro 256GB", "Apple", "phones", "flagship", 109900,
     ["smartphone", "titanium", "usb-c"],
     "6.1in 120Hz OLED · A17 Pro · 48MP main",
     "Titanium frame, customisable Action button, and a 48MP main sensor."),
    ("Google Pixel 8a 128GB", "Google", "phones", "mid", 49900,
     ["smartphone", "camera", "android", "value"],
     "6.1in 120Hz · Tensor G3 · 7 years of updates",
     "Computational photography in a mid-range body with unusually long software support."),
    ("LG C4 65in OLED evo TV", "LG", "tv", "flagship", 179900,
     ["oled", "4k", "gaming", "hdr"],
     "65in OLED · 144Hz · 4x HDMI 2.1 · Dolby Vision",
     "Perfect blacks and per-pixel contrast, the reference choice for a dark room."),
    ("Hisense U6N 55in Mini-LED TV", "Hisense", "tv", "entry", 54900,
     ["mini-led", "4k", "budget", "hdr"],
     "55in mini-LED · 60Hz · Dolby Vision",
     "Genuine mini-LED backlighting and Dolby Vision at an entry-level price."),
    ("Valve Steam Deck OLED 1TB", "Valve", "gaming", "flagship", 64900,
     ["handheld", "pc-gaming", "portable"],
     "7.4in HDR OLED 90Hz · 1TB · SteamOS",
     "A full PC that plays your existing library on a train."),
    ("Logitech G Pro X Superlight 2", "Logitech", "gaming", "mid", 15900,
     ["mouse", "wireless", "esports", "lightweight"],
     "60g · 32K DPI · 95h battery · USB-C",
     "Sixty grams with no perceptible wireless latency."),
]

# Names the tests refer to, kept as constants so a catalogue edit is a one-line change.
FLAGSHIP_AUDIO = "Sony WH-1000XM5 Wireless Headphones"
SECOND_AUDIO = "Bose QuietComfort Ultra Headphones"
ENTRY_AUDIO = "Anker Soundcore Q30 Headphones"
MID_AUDIO = "Apple AirPods Pro 2 (USB-C)"
MID_PHONE = "Google Pixel 8a 128GB"
ENTRY_TV = "Hisense U6N 55in Mini-LED TV"
FLAGSHIP_TV = "LG C4 65in OLED evo TV"
FLAGSHIP_GAMING = "Valve Steam Deck OLED 1TB"

_VOLATILE = [
    RecommendationItem, Recommendation, AgentRun, Notification, Event, UserProfile, User,
]


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_TMP, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def _database():
    init_db()
    yield


@pytest.fixture(scope="session")
def catalog(_database):
    """Seeded, vector-synced catalogue. Built once — embedding is the slow part."""
    from app.services import outbox
    from app.services.catalog import create_product

    with session_scope() as db:
        for title, brand, category, tier, price, tags, spec, description in CATALOG:
            create_product(
                db,
                ProductIn(
                    title=title, description=description, category=category, tier=tier,
                    tags=tags, price_cents=price, rating=4.5, brand=brand, spec=spec,
                ),
            )
    outbox.drain_all()
    with session_scope() as db:
        yield {p.title: p.id for p in db.scalars(__import__("sqlalchemy").select(Product))}


@pytest.fixture(autouse=True)
def _clean_volatile_tables(_database):
    """Users, events and recommendations reset between tests; the catalogue does not."""
    yield
    with session_scope() as db:
        for model in _VOLATILE:
            db.execute(delete(model))


@pytest.fixture
def db():
    with session_scope() as session:
        yield session


@pytest.fixture
def user_factory():
    counter = {"n": 0}

    def make(email: str | None = None, *, role: str = "user", **kwargs) -> int:
        counter["n"] += 1
        with session_scope() as session:
            u = User(
                email=email or f"user{counter['n']}@test.local",
                password_hash=hash_password("password123"),
                name=f"User {counter['n']}",
                role=role,
                **kwargs,
            )
            session.add(u)
            session.flush()
            session.add(UserProfile(user_id=u.id))
            return u.id

    return make


@pytest.fixture
def event_factory():
    from datetime import timedelta

    from app.models import utcnow

    def add(user_id: int, type_: str, *, product_id=None, query=None, dwell_ms=0,
            hours_ago: float = 0.0, count: int = 1) -> None:
        with session_scope() as session:
            for _ in range(count):
                session.add(
                    Event(
                        user_id=user_id, type=type_, product_id=product_id, query=query,
                        dwell_ms=dwell_ms, dedupe_key=os.urandom(12).hex(),
                        server_ts=utcnow() - timedelta(hours=hours_ago),
                    )
                )

    return add


@pytest.fixture
def fresh_vector_store():
    """For tests that mutate the index. Restores the catalogue afterwards."""
    from app.services import outbox
    from app.services.vectorstore import get_vector_store

    store = get_vector_store()
    yield store
    with session_scope() as db:
        db.execute(delete(VectorOutbox))
    outbox.reindex_all()
