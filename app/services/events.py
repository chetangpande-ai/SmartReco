"""Event ingest: accept fast, write in batches, never block the browser.

The request path does three cheap things — validate, stamp, `put_nowait` — and returns
202. All database work happens on a worker task that batches up to `event_flush_batch`
rows or `event_flush_interval_seconds`, whichever comes first, and writes them in one
statement on a thread so the event loop keeps serving.

The queue is bounded on purpose. Under a flood the correct behaviour is to shed
telemetry and keep the site responsive, not to grow a queue until the process dies.
Drops are counted, not silent.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from app import metrics
from app.config import settings
from app.db import SessionLocal
from app.models import Event, Product, utcnow

log = logging.getLogger(__name__)


def _insert_ignoring_duplicates(table):
    """Same idempotency guarantee on both backends: a replayed beacon is a no-op."""
    return sqlite_insert(table) if settings.is_sqlite else pg_insert(table)


class EventIngestor:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[dict] | None = None
        self._worker: asyncio.Task | None = None
        self._refreshers: set[asyncio.Task] = set()
        self.dropped = 0

    async def start(self) -> None:
        self._queue = asyncio.Queue(maxsize=settings.event_queue_max)
        self._worker = asyncio.create_task(self._run(), name="event-ingest")
        log.info("event ingest started", extra={"queue_max": settings.event_queue_max})

    async def stop(self) -> None:
        if self._worker is None:
            return
        self._worker.cancel()
        try:
            await self._worker
        except asyncio.CancelledError:
            pass
        # Drain whatever is still queued so a clean shutdown doesn't lose the last batch.
        if self._queue and not self._queue.empty():
            rows = []
            while not self._queue.empty():
                rows.append(self._queue.get_nowait())
            await asyncio.to_thread(self._write, rows)
            log.info("flushed on shutdown", extra={"rows": len(rows)})
        self._worker = None

    @property
    def depth(self) -> int:
        return self._queue.qsize() if self._queue else 0

    def submit(self, rows: list[dict]) -> int:
        """Non-blocking. Returns how many were accepted; the rest were shed."""
        if self._queue is None:
            raise RuntimeError("ingestor not started")
        accepted = 0
        for row in rows:
            try:
                self._queue.put_nowait(row)
                accepted += 1
            except asyncio.QueueFull:
                self.dropped += 1
        if accepted:
            metrics.inc("smartreco_events_ingested_total", accepted)
        if accepted < len(rows):
            log.warning("event queue full, shedding", extra={"dropped": len(rows) - accepted})
        return accepted

    async def _run(self) -> None:
        assert self._queue is not None
        while True:
            rows = await self._collect()
            if not rows:
                continue
            user_ids = await asyncio.to_thread(self._write, rows)
            if user_ids:
                # Deliberately not awaited. Rebuilding a profile can embed new search
                # strings, which is a network call — blocking the writer on it would
                # make ingest latency depend on Mesh, which is exactly what the queue
                # exists to prevent.
                task = asyncio.create_task(self._refresh_profiles(user_ids))
                self._refreshers.add(task)
                task.add_done_callback(self._refreshers.discard)

    async def _refresh_profiles(self, user_ids: set[int]) -> None:
        from app.services import profile

        await asyncio.to_thread(profile.refresh_users, user_ids)

    async def _collect(self) -> list[dict]:
        """Block for the first row, then fill the batch until it's full or time is up."""
        assert self._queue is not None
        rows = [await self._queue.get()]
        loop = asyncio.get_running_loop()
        deadline = loop.time() + settings.event_flush_interval_seconds
        while len(rows) < settings.event_flush_batch:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                rows.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
            except TimeoutError:
                break
        return rows

    def _write(self, rows: list[dict]) -> set[int]:
        """Returns the logged-in users touched by this batch, for profile refresh."""
        db = SessionLocal()
        try:
            self._resolve_slugs(db, rows)
            # RETURNING gives an exact inserted count. An executemany insert yields an
            # IteratorResult, which has no .rowcount at all — and on backends that do
            # expose it, a rowcount alongside ON CONFLICT DO NOTHING is not portable.
            stmt = (
                _insert_ignoring_duplicates(Event)
                .on_conflict_do_nothing(index_elements=["dedupe_key"])
                .returning(Event.id)
            )
            written = len(db.execute(stmt, rows).all())
            db.commit()
        except Exception:
            db.rollback()
            # Telemetry is not worth taking the process down for: one malformed batch
            # must not kill the ingest worker for every user. Scoped to the database
            # work only, so a bug in the bookkeeping below cannot masquerade as a
            # write failure.
            log.exception("event batch write failed", extra={"rows": len(rows)})
            return set()
        finally:
            db.close()

        metrics.inc("smartreco_events_persisted_total", written)
        duplicates = len(rows) - written
        if duplicates > 0:
            metrics.inc("smartreco_events_duplicate_total", duplicates)
        log.debug("events written", extra={"rows": len(rows), "written": written, "dupes": duplicates})
        return {r["user_id"] for r in rows if r.get("user_id")}

    @staticmethod
    def _resolve_slugs(db, rows: list[dict]) -> None:
        """Clients know the slug, not the id. One query resolves the whole batch.

        `_slug` is stripped from every row here — it is a transport field, not a column.
        """
        wanted = {r["_slug"] for r in rows if r.get("_slug") and r.get("product_id") is None}
        mapping = (
            dict(db.execute(select(Product.slug, Product.id).where(Product.slug.in_(wanted))).all())
            if wanted
            else {}
        )
        for row in rows:
            slug = row.pop("_slug", None)
            if slug and row.get("product_id") is None:
                row["product_id"] = mapping.get(slug)


def to_row(event, *, user_id: int | None, anon_id: str, path_fallback: str = "") -> dict:
    """Validated EventIn -> a dict ready for a bulk insert."""
    client_ts = None
    if event.ts:
        # Clients occasionally send seconds, or a clock that is wildly wrong. Anything
        # implausible is dropped rather than poisoning the recency decay.
        try:
            client_ts = datetime.fromtimestamp(event.ts / 1000, tz=timezone.utc)
            if not (2020 < client_ts.year < 2100):
                client_ts = None
        except (ValueError, OSError, OverflowError):
            client_ts = None

    return {
        "user_id": user_id,
        "anon_id": anon_id or "",
        "session_id": event.session_id or "",
        "type": event.type,
        "product_id": event.product_id,
        "_slug": event.product_slug,  # transport only; _resolve_slugs strips it
        "query": (event.query or None),
        "path": event.path or path_fallback,
        "dwell_ms": event.dwell_ms,
        "position": event.position,
        "meta": event.meta or {},
        "dedupe_key": event.idem or uuid.uuid4().hex,
        "client_ts": client_ts,
        "server_ts": utcnow(),
    }


ingestor = EventIngestor()
