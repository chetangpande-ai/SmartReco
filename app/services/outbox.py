"""The other half of the dual-write: draining the outbox into the vector store.

Why an outbox instead of writing to Chroma inside the request? Because a naive dual
write has a window where SQL committed and the vector write then failed — and nothing
remembers it needs fixing. Here the intent to sync is committed atomically with the
data, so a crash, a restart, or a vector store outage is a delay rather than permanent
divergence. `reconcile()` is the belt to the outbox's braces: it diffs the two stores
by content hash and repairs drift in both directions.
"""

import logging
import random
from datetime import timedelta

from sqlalchemy import func, select

from app import metrics
from app.db import session_scope
from app.models import Product, VectorOutbox, utcnow
from app.services.catalog import enqueue_sync, published_hashes, vector_metadata
from app.services.embeddings import embedder
from app.services.vectorstore import VectorRecord, get_vector_store

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 5
_EMPTY = {"processed": 0, "upserts": 0, "deletes": 0, "superseded": 0, "failed": 0}


def _backoff_seconds(attempts: int) -> float:
    """Exponential with jitter, capped at 5 minutes. Jitter matters when a whole batch
    fails together — without it every row retries in lockstep forever."""
    return min(300.0, 2.0**attempts) * (0.5 + random.random())


def _coalesce(rows: list[VectorOutbox], now) -> tuple[dict[int, VectorOutbox], int]:
    """Keep only the last row per product, closing the ones it supersedes.

    Five rapid edits to one product need the final state applied once, not five
    embeddings of intermediate states we are about to overwrite anyway.
    """
    latest: dict[int, VectorOutbox] = {}
    superseded = 0
    for row in rows:
        if row.product_id in latest:
            stale = latest[row.product_id]
            stale.status, stale.processed_at, stale.last_error = "done", now, "superseded"
            superseded += 1
        latest[row.product_id] = row
    return latest, superseded


def drain(limit: int = 200) -> dict:
    now = utcnow()
    with session_scope() as db:
        rows = list(
            db.scalars(
                select(VectorOutbox)
                .where(VectorOutbox.status == "pending", VectorOutbox.next_attempt_at <= now)
                .order_by(VectorOutbox.id)
                .limit(limit)
            )
        )
        if not rows:
            _update_pending_gauge(db)
            return dict(_EMPTY)

        latest, superseded = _coalesce(rows, now)

        upsert_ids = [pid for pid, r in latest.items() if r.op == "upsert"]
        delete_ids = [pid for pid, r in latest.items() if r.op == "delete"]

        products = {}
        if upsert_ids:
            products = {
                p.id: p for p in db.scalars(select(Product).where(Product.id.in_(upsert_ids)))
            }

        # An upsert whose product has since been deleted or unpublished must become a
        # delete, otherwise the index keeps a ghost the recommender can still surface.
        resolved_upserts = []
        for pid in upsert_ids:
            p = products.get(pid)
            if p is not None and p.is_published:
                resolved_upserts.append(pid)
            else:
                delete_ids.append(pid)

        store = get_vector_store()
        try:
            if resolved_upserts:
                vectors = embedder.embed_documents(
                    [products[pid].embedding_text() for pid in resolved_upserts], db=db
                )
                store.upsert(
                    [
                        VectorRecord(
                            id=str(pid),
                            embedding=vectors[i].tolist(),
                            document=products[pid].embedding_text(),
                            metadata=vector_metadata(products[pid]),
                        )
                        for i, pid in enumerate(resolved_upserts)
                    ]
                )
            if delete_ids:
                store.delete([str(pid) for pid in delete_ids])
        except Exception as exc:
            # Broad catch is deliberate and bounded: this is the retry boundary. An
            # embedding timeout, a Chroma lock, a Pinecone 503 all get the same correct
            # response — back off and try again, dead-letter after MAX_ATTEMPTS.
            for row in latest.values():
                row.attempts += 1
                row.last_error = str(exc)[:500]
                if row.attempts >= MAX_ATTEMPTS:
                    row.status = "dead"
                else:
                    row.next_attempt_at = now + timedelta(seconds=_backoff_seconds(row.attempts))
            log.error("outbox batch failed", extra={"rows": len(latest), "error": str(exc)[:200]})
            _update_pending_gauge(db)
            return {**_EMPTY, "failed": len(latest), "superseded": superseded}

        for pid in resolved_upserts:
            products[pid].vector_synced_at = now
        for row in latest.values():
            row.status, row.processed_at, row.last_error = "done", now, ""

        _update_pending_gauge(db)
        log.info(
            "outbox drained",
            extra={"upserts": len(resolved_upserts), "deletes": len(delete_ids)},
        )
        return {
            "processed": len(latest),
            "upserts": len(resolved_upserts),
            "deletes": len(delete_ids),
            "superseded": superseded,
            "failed": 0,
        }


def drain_all(max_batches: int = 50) -> dict:
    """Loop until nothing is pending. Used by seeding and reindex, where waiting for
    the scheduler would just be slow."""
    total = dict(_EMPTY)
    for _ in range(max_batches):
        batch = drain()
        for k in total:
            total[k] += batch[k]
        if batch["processed"] == 0 or batch["failed"]:
            break
    return total


def reconcile() -> dict:
    """Diff SQL against the vector store by content hash and queue repairs.

    Three drift classes, all real: rows SQL has and the index lost (failed sync,
    restored backup), rows both have with different content (a write that landed in one
    store only), and rows the index has that SQL no longer does (deleted while the
    vector store was unreachable).
    """
    store = get_vector_store()
    if getattr(store, "stale_embedder", False):
        log.warning("embedder changed — full reindex instead of incremental reconcile")
        return reindex_all()

    with session_scope() as db:
        sql_hashes = published_hashes(db)
        vector_hashes = store.all_hashes()

        missing = [pid for pid in sql_hashes if pid not in vector_hashes]
        stale = [
            pid
            for pid, h in sql_hashes.items()
            if pid in vector_hashes and vector_hashes[pid] != h
        ]
        orphaned = [pid for pid in vector_hashes if pid not in sql_hashes]

        if missing or stale:
            repair = db.scalars(
                select(Product).where(Product.id.in_([int(p) for p in missing + stale]))
            )
            for p in repair:
                enqueue_sync(db, p, "upsert")
        for pid in orphaned:
            db.add(
                VectorOutbox(product_id=int(pid), op="delete", payload={"reason": "orphaned"})
            )

    result = {"missing": len(missing), "stale": len(stale), "orphaned": len(orphaned)}
    if any(result.values()):
        log.warning("drift detected", extra=result)
        result["repaired"] = drain_all()
    return result


def reindex_all() -> dict:
    """Wipe the index and rebuild from SQL. The recovery path when the embedding model
    changes, since old vectors are not comparable to new query vectors."""
    store = get_vector_store()
    store.reset()
    with session_scope() as db:
        db.query(VectorOutbox).filter(VectorOutbox.status == "pending").update(
            {"status": "done", "last_error": "superseded by reindex"}
        )
        for p in db.scalars(select(Product).where(Product.is_published.is_(True))):
            enqueue_sync(db, p, "upsert")
    log.info("full reindex queued")
    return {"reindexed": drain_all()}


def _update_pending_gauge(db) -> None:
    pending = db.scalar(
        select(func.count()).select_from(VectorOutbox).where(VectorOutbox.status == "pending")
    )
    metrics.gauge("smartreco_outbox_pending", float(pending or 0))


def health() -> dict:
    """What the admin dashboard's sync badge renders."""
    store = get_vector_store()
    with session_scope() as db:
        counts = dict(
            db.execute(
                select(VectorOutbox.status, func.count()).group_by(VectorOutbox.status)
            ).all()
        )
        sql_published = db.scalar(
            select(func.count()).select_from(Product).where(Product.is_published.is_(True))
        )
        unsynced = db.scalar(
            select(func.count())
            .select_from(Product)
            .where(Product.is_published.is_(True), Product.vector_synced_at.is_(None))
        )

    vector_count = store.count()
    return {
        "sql_published": int(sql_published or 0),
        "vector_count": vector_count,
        "unsynced_products": int(unsynced or 0),
        "pending": int(counts.get("pending", 0)),
        "dead": int(counts.get("dead", 0)),
        "done": int(counts.get("done", 0)),
        "in_sync": int(sql_published or 0) == vector_count
        and not counts.get("pending")
        and not counts.get("dead"),
        "store": store.health(),
    }
