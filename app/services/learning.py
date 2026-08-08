"""What the learner has actually started, finished and earned.

Enrollment state is mutable and small, which is why it is its own table rather than a
projection over the event log: "62% through, two lessons left" changes on every session
and replaying an append-only log to find it on each dashboard render would be absurd.

The `enroll` *event* still fires alongside — the two answer different questions. The
event says a click happened at a moment in time and feeds the behaviour profile; the row
says where they are now. Deleting the row must not rewrite history, so neither derives
from the other.
"""

import logging
import secrets

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app import taxonomy
from app.models import Enrollment, Product, utcnow

log = logging.getLogger(__name__)

ACTIVE, COMPLETED, SAVED = "active", "completed", "saved"


def get(db: Session, user_id: int, product_id: int) -> Enrollment | None:
    return db.scalar(
        select(Enrollment).where(
            Enrollment.user_id == user_id, Enrollment.product_id == product_id
        )
    )


def start(db: Session, user_id: int, product: Product) -> Enrollment:
    """Begin, or resume, a course. Idempotent — the button is a link people re-click."""
    row = get(db, user_id, product.id)
    if row is None:
        row = Enrollment(user_id=user_id, product_id=product.id, status=ACTIVE)
        db.add(row)
    elif row.status == SAVED:
        row.status = ACTIVE
    row.last_activity_at = utcnow()
    db.flush()
    log.info("enrollment started", extra={"user_id": user_id, "product_id": product.id})
    return row


def save(db: Session, user_id: int, product: Product) -> Enrollment:
    """Wishlist. A saved course is an enrollment that has not begun, not a separate
    concept — which is what makes "I saved this then started it" one row, not two."""
    row = get(db, user_id, product.id)
    if row is None:
        row = Enrollment(user_id=user_id, product_id=product.id, status=SAVED)
        db.add(row)
        db.flush()
    return row


def set_progress(db: Session, user_id: int, product: Product, pct: int) -> Enrollment:
    """Move a course along. Reaching 100 completes it and issues the certificate.

    Completion is handled here rather than by the caller so there is exactly one place
    that can mint a certificate code, and so it cannot be issued twice.
    """
    row = start(db, user_id, product)
    row.progress_pct = max(0, min(int(pct), 100))
    row.last_activity_at = utcnow()

    if row.progress_pct >= 100 and row.status != COMPLETED:
        row.status = COMPLETED
        row.completed_at = utcnow()
        if product.certificate and not row.certificate_code:
            row.certificate_code = f"SR-{secrets.token_hex(4).upper()}"
        log.info(
            "course completed",
            extra={"user_id": user_id, "product_id": product.id,
                   "certificate": bool(row.certificate_code)},
        )
    elif row.progress_pct < 100 and row.status == COMPLETED:
        # Reopening a finished course keeps the certificate: it was earned, and taking
        # it back because someone rewatched a lesson would be absurd.
        row.status = ACTIVE
        row.completed_at = None

    db.flush()
    return row


def _rows(db: Session, user_id: int, status: str | None = None) -> list[Enrollment]:
    stmt = (
        select(Enrollment)
        .where(Enrollment.user_id == user_id)
        .options(selectinload(Enrollment.product).selectinload(Product.skills))
        .order_by(Enrollment.last_activity_at.desc())
    )
    if status:
        stmt = stmt.where(Enrollment.status == status)
    return list(db.scalars(stmt))


def dashboard(db: Session, user_id: int) -> dict:
    """Everything My Learning shows, in one pass over the learner's enrollments.

    One query rather than five: the dashboard needs all of these at once, and the row
    count per learner is small enough that filtering in Python beats four more
    round trips.
    """
    rows = _rows(db, user_id)
    in_progress = [r for r in rows if r.status == ACTIVE]
    completed = [r for r in rows if r.status == COMPLETED]
    return {
        "all": rows,
        "in_progress": in_progress,
        "completed": completed,
        "saved": [r for r in rows if r.status == SAVED],
        "certificates": [r for r in completed if r.certificate_code],
        "projects": [
            r for r in rows if r.product.projects or r.product.format.endswith("Project")
        ],
        "assessments": [r for r in rows if r.product.assessments],
        "hours_done": sum(
            round(r.product.duration_hours * r.progress_pct / 100) for r in rows
        ),
        "skills_earned": skills_from(completed),
    }


def skills_from(enrollments) -> list[str]:
    """Skills a set of completed courses proves, including what those imply."""
    earned: list[str] = []
    for row in enrollments:
        earned.extend(s for s in row.product.teaches if s not in earned)
    return taxonomy.expand_skills(earned)


def counts(db: Session, user_id: int) -> dict[str, int]:
    rows = db.execute(
        select(Enrollment.status, func.count())
        .where(Enrollment.user_id == user_id)
        .group_by(Enrollment.status)
    ).all()
    return dict(rows)
