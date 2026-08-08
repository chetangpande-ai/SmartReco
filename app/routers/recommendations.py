"""The learner's own page: where they are, what they are part-way through, what next.

One page, not two. There were briefly separate "My picks" and "My learning" routes, and
a learner had no way to know which of them was the one about them — both were. Progress
against a target role and the recommendation of what to do next are the same question
asked twice, so they belong in one column.

Generation never happens inside the request. The agent takes seconds; a page that
blocks on it is a page that feels broken. Instead the route renders immediately with
whatever exists, queues a background run when the trigger policy allows one, and a
small poller swaps in the result when it lands.
"""

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from app import taxonomy
from app.db import get_db
from app.deps import require_user, verify_csrf
from app.models import CareerProfile, User, UserProfile
from app.ratelimit import recommend_limiter
from app.services import advisor, catalog, learning, recommender, shelves, triggers
from app.services import profile as profile_service
from app.templating import render

log = logging.getLogger(__name__)
router = APIRouter(tags=["recommendations"])


@router.get("/me")
def my_page(
    request: Request,
    background: BackgroundTasks,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    prof = db.get(UserProfile, user.id)
    rec = recommender.get_current(db, user.id)
    items = recommender.items_with_products(db, rec)
    decision = triggers.evaluate(db, user.id, prof)

    if decision.run:
        background.add_task(recommender.generate_for_user, user.id)

    board = learning.dashboard(db, user.id)
    career = db.get(CareerProfile, user.id)

    # Skill progress is the honest version of a progress bar: not "you are 40% through
    # three courses" but "you can now do four of the nine things the job asks for".
    held = set(advisor.held_skills(db, career)) if career else set()
    target = taxonomy.role(career.target_role) if career else None
    skill_rows = [
        {"slug": s, "name": taxonomy.skill_name(s), "held": s in held}
        for s in (target.skills if target else [])
    ]

    # Everything already on the page above the browse row: the agent's picks and the
    # courses they are part-way through. Without this the row repeats them.
    shown = {e["product"].id for e in items} | {r.product_id for r in board["all"]}

    return render(
        request,
        "dashboard.html",
        rec=rec,
        items=items,
        shelves=shelves.build(db, user.id, shown),
        board=board,
        plan=advisor.current_plan(db, user.id),
        target=target,
        skill_rows=skill_rows,
        held_count=sum(1 for r in skill_rows if r["held"]),
        categories=catalog.categories(db),
        evidence=profile_service.evidence(db, user.id) if prof else [],
        profile=prof,
        summary=profile_service.summarise(prof) if prof else "",
        decision=decision,
        pending=decision.run and rec is None,
    )


@router.get("/me/learning")
def my_learning_moved():
    """Folded into /me. Kept because the split shipped in a demo build and someone will
    have the old URL open."""
    return RedirectResponse("/me", status_code=status.HTTP_301_MOVED_PERMANENTLY)


@router.post("/me/refresh", dependencies=[Depends(verify_csrf)])
def refresh(
    request: Request,
    background: BackgroundTasks,
    user: User = Depends(require_user),
):
    if not recommend_limiter.allow(f"rec:{user.id}"):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "please wait a moment")
    background.add_task(recommender.generate_for_user, user.id, force=True)
    return RedirectResponse("/me?refreshing=1", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/api/recommendations/current")
def current_json(user: User = Depends(require_user), db: Session = Depends(get_db)) -> dict:
    """Polled by the dashboard while a background run is in flight."""
    rec = recommender.get_current(db, user.id)
    if rec is None:
        return {"ready": False}
    return {
        "ready": True,
        "id": rec.id,
        "headline": rec.headline,
        "narrative": rec.narrative,
        "cta": rec.cta,
        "strategy": rec.strategy,
        "confidence": rec.confidence,
        "created_at": rec.created_at.isoformat(),
        "items": [
            {
                "product_id": entry["product"].id,
                "slug": entry["product"].slug,
                "title": entry["product"].title,
                "price_cents": entry["product"].price_cents,
                "tier": entry["product"].tier,
                "brand": entry["product"].brand,
                "category": entry["product"].category,
                "rating": entry["product"].rating,
                "reason": entry["reason"],
            }
            for entry in recommender.items_with_products(db, rec)
        ],
    }
