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
    # Off even when .env has real credentials: a test run must not ship spans to
    # someone's LangSmith project, and logfire's instrumentation is process-global.
    # test_observability.py drives both against a stub instead.
    LANGSMITH_TRACING="false",
    LOGFIRE_ENABLED="false",
    # Real randomness by default (ranking.apply_exploration_slot) would make any test
    # with more than top_k candidates occasionally flaky. Tests that specifically cover
    # exploration call apply_exploration_slot directly with an explicit epsilon instead
    # of going through settings, so this is safe to pin off suite-wide.
    EXPLORE_EPSILON="0.0",
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
# the whole suite stays fast. Two providers appear twice so provider affinity is testable.
#
# `SKILLS` maps a title to (teaches, requires) in canonical `app.taxonomy` slugs. Kept
# beside the catalogue rather than as more tuple positions because only the career tests
# care, and threading two more fields through every row would make the rest unreadable.
# The chain it encodes is deliberate: python -> llms -> rag -> ai-agents, each requiring
# the one before, which is what lets a test assert the advisor sequences rather than
# merely lists.
CATALOG = [
    ("Deep Learning Specialization", "DeepLearning.AI", "ai-ml", "advanced", 5900,
     ["deep-learning", "neural-networks", "tensorflow", "cnn"],
     "5 courses · 3 months at 10h/week · TensorFlow",
     "Builds neural networks from the ground up before touching a framework, then covers convolutional and sequence models."),
    ("Natural Language Processing with Transformers", "O'Reilly", "ai-ml", "advanced", 8900,
     ["nlp", "transformers", "huggingface", "fine-tuning"],
     "11 chapters · ~30h · Hugging Face",
     "Attention, tokenisation, fine-tuning and distillation, worked through the Hugging Face stack."),
    ("Machine Learning Specialization", "DeepLearning.AI", "ai-ml", "beginner", 4900,
     ["machine-learning", "python", "supervised-learning", "regression"],
     "3 courses · 2 months at 10h/week · Python",
     "Linear and logistic regression, neural networks and clustering, with the maths kept to what you need."),
    ("Practical Deep Learning for Coders", "fast.ai", "ai-ml", "intermediate", 0,
     ["deep-learning", "pytorch", "computer-vision", "nlp"],
     "8 lessons · ~40h · PyTorch and fastai",
     "Top-down teaching: you train a working image classifier in lesson one, then peel back the layers."),
    ("Total TypeScript", "Matt Pocock", "web-dev", "advanced", 24900,
     ["typescript", "types", "generics", "javascript"],
     "20h+ · interactive exercises · 200+ challenges",
     "Type-level programming taught as puzzles you solve in your editor: generics and conditional types."),
    ("Complete Intro to React", "Frontend Masters", "web-dev", "intermediate", 3900,
     ["react", "javascript", "hooks", "frontend"],
     "8h video · builds one app throughout",
     "React taught by building a single application from an empty folder, with no scaffolding to hide the wiring."),
    ("Data Engineering Zoomcamp", "DataTalks.Club", "data", "advanced", 0,
     ["data-engineering", "dbt", "airflow", "spark"],
     "9 weeks · Docker, dbt, Airflow, Spark",
     "A full pipeline built week by week: ingestion, warehousing, transformation, orchestration and streaming."),
    ("SQL for Data Analysis", "Udacity", "data", "beginner", 7900,
     ["sql", "postgres", "joins", "window-functions"],
     "4 weeks · PostgreSQL · query workspace",
     "Joins, aggregations, subqueries and window functions, practised against a real transactional database."),
    ("Offensive Security Certified Professional", "OffSec", "security", "advanced", 164900,
     ["pentesting", "certification", "exploitation", "reporting"],
     "PEN-200 · 90 days lab access · 24h exam",
     "The practical penetration testing certification: a 24-hour hands-on exam against machines you must compromise."),
    ("Practical Ethical Hacking", "TCM Security", "security", "beginner", 3000,
     ["pentesting", "linux", "networking", "active-directory"],
     "25h video · lab environment · certificate",
     "Networking and Linux fundamentals first, then reconnaissance and exploitation in a lab you build yourself."),
]

SKILLS = {
    "Deep Learning Specialization": (("deep-learning", "neural-networks"), ("python",)),
    "Natural Language Processing with Transformers": (("nlp", "transformers"), ("deep-learning",)),
    "Machine Learning Specialization": (("machine-learning", "statistics"), ("python",)),
    "Practical Deep Learning for Coders": (("python", "llms", "rag", "ai-agents"), ()),
    "Total TypeScript": (("typescript",), ("javascript",)),
    "Complete Intro to React": (("react", "javascript"), ()),
    "Data Engineering Zoomcamp": (("etl", "airflow", "spark"), ("python", "sql")),
    "SQL for Data Analysis": (("sql", "databases"), ()),
    "Offensive Security Certified Professional": (("penetration-testing",), ("linux",)),
    "Practical Ethical Hacking": (("linux", "networking", "security"), ()),
}

# Names the tests refer to, kept as constants so a catalogue edit is a one-line change.
ADVANCED_AI = "Deep Learning Specialization"
SECOND_AI = "Natural Language Processing with Transformers"
BEGINNER_AI = "Machine Learning Specialization"
MID_AI = "Practical Deep Learning for Coders"
MID_WEB = "Complete Intro to React"
BEGINNER_DATA = "SQL for Data Analysis"
ADVANCED_DATA = "Data Engineering Zoomcamp"
ADVANCED_SECURITY = "Offensive Security Certified Professional"

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
            teaches, requires = SKILLS.get(title, ((), ()))
            create_product(
                db,
                ProductIn(
                    title=title, description=description, category=category, tier=tier,
                    tags=tags, price_cents=price, rating=4.5, brand=brand, spec=spec,
                    teaches=list(teaches), requires=list(requires),
                    duration_hours=20, certificate=True, instructor=brand,
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
