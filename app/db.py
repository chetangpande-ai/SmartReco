"""Engine, session factory, and the SQLite pragmas that make concurrent writes safe.

Sessions are sync. FastAPI runs `def` endpoints in a threadpool, and the async event
consumer hops to a thread for its bulk inserts, so nothing blocks the event loop.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings

log = logging.getLogger(__name__)

_connect_args = {"check_same_thread": False, "timeout": 30} if settings.is_sqlite else {}

engine = create_engine(
    settings.database_url,
    connect_args=_connect_args,
    pool_pre_ping=True,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


if settings.is_sqlite:

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_conn, _record):
        cur = dbapi_conn.cursor()
        # WAL lets the ingest worker write while page requests read. Without it the
        # tracking beacon can block a page render under load.
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA busy_timeout=30000")
        cur.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """For background workers and scripts: commit on success, roll back on failure."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    from app import models  # noqa: F401  (registers mappers before create_all)

    models.Base.metadata.create_all(bind=engine)
    log.info("database ready", extra={"url": settings.database_url.split("://")[0]})
