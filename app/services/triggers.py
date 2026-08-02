"""When is a recommendation worth an LLM call?

The brief asks for this explicitly — "don't fire an LLM call on every single user
action" — so the policy is a first-class, inspectable object rather than an `if` buried
in a handler. Every decision records a reason, which is what makes the admin dashboard
able to say "we served 168 events with 25 model calls and skipped 143, here's why".

Gates run cheapest-first, and the ordering is also what makes the accounting honest: a
request skipped for cooldown is counted as cooldown, not as a cache hit that happened
to also be in cooldown.
"""

import logging
from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import metrics
from app.config import settings
from app.models import AgentRun, Recommendation, UserProfile, ensure_utc, utcnow
from app.services import profile as profile_service
from app.services.mesh import mesh

log = logging.getLogger(__name__)


@dataclass
class Decision:
    run: bool
    reason: str
    cached: Recommendation | None = None
    drift: float = 0.0
    signature: str = ""

    @property
    def served_from_cache(self) -> bool:
        return self.cached is not None


def current_recommendation(db: Session, user_id: int) -> Recommendation | None:
    return db.scalar(
        select(Recommendation)
        .where(Recommendation.user_id == user_id, Recommendation.is_current.is_(True))
        .order_by(Recommendation.created_at.desc())
    )


def find_cached(db: Session, user_id: int, signature: str) -> Recommendation | None:
    """A previous recommendation built from the same interest fingerprint.

    Not restricted to the current one on purpose: if someone drifts to a new topic and
    back again, the earlier recommendation is still the right answer and reusing it
    costs nothing.
    """
    if not signature:
        return None
    return db.scalar(
        select(Recommendation)
        .where(
            Recommendation.user_id == user_id,
            Recommendation.behavior_signature == signature,
            Recommendation.expires_at > utcnow(),
            Recommendation.strategy != "fallback",
        )
        .order_by(Recommendation.created_at.desc())
    )


def _calls_today(db: Session, user_id: int) -> int:
    start = utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    return int(
        db.scalar(
            select(func.count())
            .select_from(AgentRun)
            .where(
                AgentRun.user_id == user_id,
                AgentRun.created_at >= start,
                AgentRun.status == "ok",
            )
        )
        or 0
    )


def _skip(reason: str, **extra) -> Decision:
    metrics.inc("smartreco_llm_calls_skipped_total", reason=reason)
    return Decision(run=False, reason=reason, **extra)


def evaluate(
    db: Session, user_id: int, prof: UserProfile | None, *, force: bool = False
) -> Decision:
    if prof is None or not prof.events_total:
        return _skip("no_activity")

    signature = profile_service.signature(prof)
    current = current_recommendation(db, user_id)
    interest_drift = profile_service.drift(prof.centroid, prof.last_rec_centroid)

    # A manual refresh is an explicit request, so it skips the behavioural gates — but
    # never the spend limits. "The user asked" is not a licence to ignore the budget.
    if force:
        blocked = _budget_block(db, user_id)
        if blocked:
            return _skip(blocked, drift=interest_drift, signature=signature)
        return Decision(True, "manual_refresh", drift=interest_drift, signature=signature)

    # Same interests as a recommendation we already paid for: reuse it. This is the
    # single biggest saver, because most events do not change what someone is into.
    cached = find_cached(db, user_id, signature)
    if cached is not None:
        metrics.inc("smartreco_llm_calls_skipped_total", reason="cache_hit")
        return Decision(False, "cache_hit", cached=cached, drift=interest_drift, signature=signature)

    if current is None:
        if prof.events_total < settings.rec_min_events:
            return _skip("warming_up", drift=interest_drift, signature=signature)
        blocked = _budget_block(db, user_id)
        if blocked:
            return _skip(blocked, drift=interest_drift, signature=signature)
        return Decision(True, "first_recommendation", drift=interest_drift, signature=signature)

    if prof.events_since_rec < settings.rec_min_events:
        return _skip("too_few_new_events", drift=interest_drift, signature=signature)

    last_at = ensure_utc(prof.last_rec_at)
    if last_at and utcnow() - last_at < timedelta(seconds=settings.rec_cooldown_seconds):
        return _skip("cooldown", drift=interest_drift, signature=signature)

    expired = current.is_expired()
    drifted = interest_drift >= settings.rec_drift_threshold
    stale = prof.events_since_rec >= settings.rec_staleness_events

    if not (drifted or stale or expired):
        # The user is active but circling the same topic. A new call would produce
        # near-identical copy, so the honest thing is not to spend it.
        return _skip("interests_unchanged", drift=interest_drift, signature=signature)

    blocked = _budget_block(db, user_id)
    if blocked:
        return _skip(blocked, drift=interest_drift, signature=signature)

    reason = "interest_drift" if drifted else ("expired" if expired else "staleness")
    return Decision(True, reason, drift=interest_drift, signature=signature)


def _budget_block(db: Session, user_id: int) -> str | None:
    if not settings.has_llm:
        return None  # offline mode still produces a deterministic recommendation
    if mesh.budget_remaining_usd <= 0:
        return "daily_budget_exhausted"
    if _calls_today(db, user_id) >= settings.llm_max_calls_per_user_per_day:
        return "user_daily_cap"
    if not mesh.available:
        return "circuit_open"
    return None


def efficiency_stats(db: Session | None = None) -> dict:
    """Powers the "LLM calls avoided" panel on the admin dashboard.

    Skip decisions are counted in-process, so the ratio is explicitly scoped to "since
    this process started" — mixing a volatile counter with an all-time database total
    would produce a number like "0 runs out of 1 opportunity", which is worse than
    useless because it looks authoritative. Durable totals are reported separately.
    """
    snapshot = metrics.snapshot()
    skipped_by_reason = {
        key.split('reason="')[1].rstrip('"}'): int(value)
        for key, value in snapshot.items()
        if key.startswith("smartreco_llm_calls_skipped_total{")
    }
    skipped = sum(skipped_by_reason.values())
    runs = int(sum(v for k, v in snapshot.items() if k.startswith("smartreco_agent_runs_total")))
    considered = skipped + runs

    stats = {
        "agent_runs": runs,
        "skipped": skipped,
        "considered": considered,
        "avoided_pct": round(100 * skipped / considered, 1) if considered else 0.0,
        "by_reason": dict(sorted(skipped_by_reason.items(), key=lambda kv: -kv[1])),
        "budget_remaining_usd": round(mesh.budget_remaining_usd, 4),
        "spend_today_usd": round(settings.llm_daily_budget_usd - mesh.budget_remaining_usd, 4),
        "runs_all_time": None,
        "spend_all_time_usd": None,
    }

    if db is not None:
        stats["runs_all_time"] = int(
            db.scalar(select(func.count()).select_from(AgentRun).where(AgentRun.status == "ok")) or 0
        )
        stats["spend_all_time_usd"] = round(
            float(db.scalar(select(func.coalesce(func.sum(AgentRun.cost_usd), 0.0))) or 0.0), 6
        )
    return stats
