"""Background jobs.

APScheduler rather than Celery because every job here is small, idempotent, and
in-process — adding a broker and a worker container would be infrastructure with no
job to do. The trade-off is explicit: this design assumes one application process. The
jobs are all safe to run concurrently anyway (the outbox coalesces, reconcile is a
diff, the digest is guarded by a unique key), so moving to multiple workers later means
changing the scheduler, not the jobs.

Every job carries `max_instances=1` and `coalesce=True`: if one run overruns its
interval, the next is skipped rather than piling up behind it.
"""

import logging

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from app.config import settings
from app.ratelimit import auth_limiter, events_limiter, recommend_limiter

log = logging.getLogger(__name__)
scheduler: BackgroundScheduler | None = None


def _drain_outbox() -> None:
    from app.services import outbox

    result = outbox.drain()
    if result["processed"] or result["failed"]:
        log.info("scheduled outbox drain", extra=result)


def _reconcile() -> None:
    from app.services import outbox

    result = outbox.reconcile()
    if any(v for k, v in result.items() if k in ("missing", "stale", "orphaned")):
        log.warning("scheduled reconcile repaired drift", extra=result)


def _digest() -> None:
    from app.services.digest import send_daily_digests

    send_daily_digests()


def _prune_limiters() -> None:
    freed = sum(b.prune() for b in (events_limiter, auth_limiter, recommend_limiter))
    if freed:
        log.debug("pruned rate limiter buckets", extra={"freed": freed})


def start() -> BackgroundScheduler | None:
    global scheduler
    if not settings.scheduler_enabled:
        log.info("scheduler disabled")
        return None
    if scheduler is not None:
        return scheduler

    scheduler = BackgroundScheduler(timezone="UTC")

    scheduler.add_job(
        _drain_outbox,
        IntervalTrigger(seconds=settings.outbox_interval_seconds),
        id="outbox_drain",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=30,
    )
    scheduler.add_job(
        _reconcile,
        IntervalTrigger(minutes=settings.reconcile_interval_minutes),
        id="reconcile",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=300,
    )
    scheduler.add_job(
        _digest,
        # jitter spreads the sends so a large user base does not hit Mesh in one burst
        CronTrigger(hour=settings.digest_hour, minute=settings.digest_minute, jitter=120),
        id="daily_digest",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,
    )
    scheduler.add_job(
        _prune_limiters,
        IntervalTrigger(minutes=15),
        id="prune_limiters",
        max_instances=1,
        coalesce=True,
    )

    scheduler.start()
    log.info(
        "scheduler started",
        extra={
            "digest_at_utc": f"{settings.digest_hour:02d}:{settings.digest_minute:02d}",
            "jobs": [j.id for j in scheduler.get_jobs()],
        },
    )
    return scheduler


def shutdown() -> None:
    global scheduler
    if scheduler is not None:
        scheduler.shutdown(wait=False)
        scheduler = None
        log.info("scheduler stopped")


def job_status() -> list[dict]:
    """Rendered on the admin dashboard so "there is a real scheduler" is checkable."""
    if scheduler is None:
        return []
    return [
        {
            "id": job.id,
            "trigger": str(job.trigger),
            "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
        }
        for job in scheduler.get_jobs()
    ]
