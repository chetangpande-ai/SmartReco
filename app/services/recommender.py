"""Orchestration: should we run the agent, run it, persist what it produced.

Kept separate from the graph so the graph stays a pure reasoning workflow and the
policy about *when* to spend money lives in one readable place.
"""

import logging
import time

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app import metrics
from app.agent.graph import graph
from app.agent.state import new_state
from app.config import settings
from app.db import session_scope
from app.logging_conf import new_request_id, request_id_var
from app.models import (
    AgentRun,
    Product,
    Recommendation,
    RecommendationItem,
    UserProfile,
    rec_expiry,
    utcnow,
)
from app.observability import configure as configure_tracing
from app.observability import current_run_url, traced
from app.services import triggers

log = logging.getLogger(__name__)


def get_current(db: Session, user_id: int) -> Recommendation | None:
    return triggers.current_recommendation(db, user_id)


def items_with_products(db: Session, rec: Recommendation | None) -> list[dict]:
    """A recommendation's picks joined to their products, in rank order.

    An item whose product has since been deleted is dropped rather than rendered as a
    hole. Both surfaces that show a recommendation — the dashboard and the digest
    email — need exactly this, so it lives here rather than in either of them.
    """
    if rec is None:
        return []
    products = {
        p.id: p
        for p in db.scalars(
            select(Product).where(Product.id.in_([i.product_id for i in rec.items]))
        )
    }
    return [
        {"product": products[i.product_id], "reason": i.reason, "rank": i.rank}
        for i in rec.items
        if i.product_id in products
    ]


def generate_for_user(user_id: int, *, force: bool = False) -> Recommendation | None:
    """Evaluate the trigger policy and, if it says go, run the agent and store the result.

    Returns whatever the user should currently be shown, which may be a cached or
    existing recommendation when no new call was warranted.
    """
    request_id = request_id_var.get()
    if request_id == "-":
        request_id = new_request_id()
        request_id_var.set(request_id)

    with session_scope() as db:
        prof = db.get(UserProfile, user_id)
        decision = triggers.evaluate(db, user_id, prof, force=force)

        if decision.cached is not None:
            _make_current(db, user_id, decision.cached.id)
            log.info("served cached recommendation",
                     extra={"user_id": user_id, "rec_id": decision.cached.id})
            return decision.cached

        if not decision.run:
            log.info("skipped agent run",
                     extra={"user_id": user_id, "reason": decision.reason,
                            "drift": round(decision.drift, 4)})
            return triggers.current_recommendation(db, user_id)

        signature = decision.signature
        centroid = prof.centroid if prof else None

    configure_tracing()
    state = new_state(user_id, decision.reason, request_id, settings.rec_top_k)

    started = time.perf_counter()
    try:
        final, trace_url = _invoke(state)
        error = final.get("error", "")
    except Exception as exc:
        log.exception("agent run failed", extra={"user_id": user_id})
        return _failed_run(
            user_id, decision.reason, {}, int((time.perf_counter() - started) * 1000),
            error=str(exc)[:1000], request_id=request_id, trace_url="",
        )

    latency_ms = int((time.perf_counter() - started) * 1000)

    if error or not final.get("picks"):
        log.warning("agent produced nothing usable",
                    extra={"user_id": user_id, "error": error, "path": final.get("node_path")})
        return _failed_run(
            user_id, decision.reason, final, latency_ms,
            error=error or "no picks", request_id=request_id, trace_url=trace_url,
        )

    rec_id = _persist(user_id, final, signature, centroid)
    _record_run(user_id, rec_id, decision.reason, final, latency_ms,
                status="ok", error="", request_id=request_id, trace_url=trace_url)
    metrics.inc("smartreco_agent_runs_total", status="ok")

    log.info(
        "recommendation generated",
        extra={
            "user_id": user_id, "rec_id": rec_id, "trigger": decision.reason,
            "strategy": final.get("strategy"), "picks": len(final["picks"]),
            "grade": final.get("grade_score"), "refines": final.get("refine_count", 0),
            "llm_calls": final.get("llm_calls", 0),
            "cost_usd": round(final.get("cost_usd", 0.0), 6), "ms": latency_ms,
            "path": "->".join(final.get("node_path", [])),
        },
    )

    with session_scope() as db:
        return db.get(Recommendation, rec_id)


@traced("smartreco.recommend", run_type="chain")
def _invoke(state: dict) -> tuple[dict, str]:
    """One LangSmith run wrapping the whole graph, and the deep link read from inside it.

    The link has to be captured here: `get_current_run_tree()` returns None the moment
    invoke() returns, so reading it later — where it used to be read, next to the other
    AgentRun fields — silently stored an empty string every time.
    """
    return graph.invoke(state), current_run_url()


def _make_current(db: Session, user_id: int, rec_id: int) -> None:
    db.execute(
        update(Recommendation)
        .where(Recommendation.user_id == user_id, Recommendation.id != rec_id)
        .values(is_current=False)
    )
    db.execute(
        update(Recommendation).where(Recommendation.id == rec_id).values(is_current=True)
    )


def _persist(user_id: int, final: dict, signature: str, centroid: bytes | None) -> int:
    with session_scope() as db:
        db.execute(
            update(Recommendation)
            .where(Recommendation.user_id == user_id, Recommendation.is_current.is_(True))
            .values(is_current=False)
        )

        rec = Recommendation(
            user_id=user_id,
            headline=final.get("headline", "")[:200],
            narrative=final.get("narrative", ""),
            cta=final.get("cta", "")[:120],
            strategy=final.get("strategy", "agentic"),
            model=final.get("model", ""),
            trigger_reason=final.get("trigger", ""),
            behavior_signature=signature,
            confidence=float(final.get("confidence", 0.0)),
            is_current=True,
            expires_at=rec_expiry(settings.rec_ttl_minutes),
        )
        db.add(rec)
        db.flush()

        seen: set[int] = set()
        rank = 0
        for pick in final["picks"]:
            pid = pick["product_id"]
            if pid in seen:
                continue  # the unique constraint would reject a duplicate anyway
            seen.add(pid)
            db.add(
                RecommendationItem(
                    recommendation_id=rec.id,
                    product_id=pid,
                    rank=rank,
                    score=float(final.get("confidence", 0.0)),
                    reason=pick.get("reason", ""),
                )
            )
            rank += 1

        prof = db.get(UserProfile, user_id)
        if prof is not None:
            # Reset the drift baseline: from here, "has anything changed?" is measured
            # against the interests this recommendation was actually built from.
            prof.last_rec_at = utcnow()
            prof.last_rec_signature = signature
            prof.last_rec_centroid = centroid
            prof.events_since_rec = 0

        return rec.id


def _failed_run(
    user_id: int,
    trigger: str,
    final: dict,
    latency_ms: int,
    *,
    error: str,
    request_id: str,
    trace_url: str,
) -> Recommendation | None:
    """Record a failed run and fall back to whatever the user already had.

    A failed agent run must not surface as a 500 on a page someone was just browsing:
    both failure paths — the graph raising, and the graph returning nothing usable —
    owe the caller the same thing, so they share this.
    """
    _record_run(user_id, None, trigger, final, latency_ms,
                status="error", error=error, request_id=request_id, trace_url=trace_url)
    metrics.inc("smartreco_agent_runs_total", status="error")
    with session_scope() as db:
        return triggers.current_recommendation(db, user_id)


def _record_run(
    user_id: int,
    rec_id: int | None,
    trigger: str,
    final: dict,
    latency_ms: int,
    *,
    status: str,
    error: str,
    request_id: str,
    trace_url: str,
) -> None:
    with session_scope() as db:
        db.add(
            AgentRun(
                user_id=user_id,
                recommendation_id=rec_id,
                trigger=trigger,
                status=status,
                node_path=final.get("node_path", []),
                retrieval_stats=final.get("retrieval_stats", {}),
                grade_score=float(final.get("grade_score", 0.0)),
                refine_loops=int(final.get("refine_count", 0)),
                prompt_tokens=int(final.get("prompt_tokens", 0)),
                completion_tokens=int(final.get("completion_tokens", 0)),
                llm_calls=int(final.get("llm_calls", 0)),
                cost_usd=float(final.get("cost_usd", 0.0)),
                latency_ms=latency_ms,
                request_id=request_id,
                langsmith_url=trace_url,
                error=error,
            )
        )


def recent_runs(db: Session, limit: int = 25) -> list[AgentRun]:
    return list(
        db.scalars(select(AgentRun).order_by(AgentRun.created_at.desc()).limit(limit))
    )
