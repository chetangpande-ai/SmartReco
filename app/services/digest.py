"""Build and send the daily digest.

The brief's example is "an email in the afternoon recapping the morning's interests" —
so the digest is not a generic newsletter. It only goes to people who were actually
active today, it recaps what they specifically did, and the recommendation inside it is
generated from that same day's behaviour.
"""

import logging
from datetime import timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import session_scope
from app.models import Event, Product, Recommendation, User, utcnow
from app.services import notify, recommender
from app.services import profile as profile_service
from app.templating import templates

log = logging.getLogger(__name__)


def active_users_today(db: Session, since_hours: int = 14) -> list[User]:
    """Users with enough activity in the recent window to be worth emailing."""
    cutoff = utcnow() - timedelta(hours=since_hours)
    rows = db.execute(
        select(Event.user_id, func.count())
        .where(Event.user_id.isnot(None), Event.server_ts >= cutoff)
        .group_by(Event.user_id)
        .having(func.count() >= settings.digest_min_events)
    ).all()
    user_ids = [uid for uid, _ in rows]
    if not user_ids:
        return []
    return list(
        db.scalars(
            select(User).where(User.id.in_(user_ids), User.digest_opt_in.is_(True))
        )
    )


def render_digest(db: Session, user: User, rec: Recommendation) -> tuple[str, str, str]:
    products = {
        p.id: p
        for p in db.scalars(
            select(Product).where(Product.id.in_([i.product_id for i in rec.items]))
        )
    }
    items = [
        {"product": products[i.product_id], "reason": i.reason}
        for i in rec.items
        if i.product_id in products
    ]
    context = {
        "rec": rec,
        "items": items,
        "evidence": profile_service.evidence(db, user.id, limit=5),
        "base_url": settings.base_url.rstrip("/"),
        "user": user,
    }
    subject = rec.headline or "Your picks from SmartReco"
    html = templates.get_template("email/digest.html").render(subject=subject, **context)
    text = templates.get_template("email/digest.txt").render(**context)
    return subject, html, text


def send_daily_digests() -> dict:
    """One scheduler tick. Safe to run twice — the dedupe key makes it a no-op."""
    today = utcnow().date().isoformat()
    sent = skipped = failed = 0

    with session_scope() as db:
        users = active_users_today(db)
    log.info("digest run starting", extra={"candidates": len(users)})

    for user in users:
        # Refresh first so the email reflects today's browsing rather than whatever was
        # generated hours ago. The trigger policy still decides whether that costs a call.
        try:
            recommender.generate_for_user(user.id)
        except Exception:
            log.exception("digest pre-refresh failed", extra={"user_id": user.id})

        with session_scope() as db:
            fresh = db.get(User, user.id)
            rec = recommender.get_current(db, user.id)
            if rec is None or not rec.items:
                skipped += 1
                continue

            subject, html, text = render_digest(db, fresh, rec)
            delivered = notify.send_once(
                db,
                fresh,
                dedupe_key=f"digest:{fresh.id}:{today}",
                subject=subject,
                html=html,
                text=text,
                recommendation_id=rec.id,
            )
            if delivered:
                fresh.last_digest_at = utcnow()
                sent += 1
            else:
                skipped += 1

    result = {"candidates": len(users), "sent": sent, "skipped": skipped, "failed": failed}
    log.info("digest run complete", extra=result)
    return result
