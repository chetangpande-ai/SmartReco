"""The learner's recommendation page.

Generation never happens inside the request. The agent takes seconds; a page that
blocks on it is a page that feels broken. Instead the route renders immediately with
whatever exists, queues a background run when the trigger policy allows one, and a
small poller swaps in the result when it lands.
"""

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.deps import require_user, verify_csrf
from app.models import Product, Recommendation, User, UserProfile
from app.ratelimit import recommend_limiter
from app.services import profile as profile_service
from app.services import recommender, triggers
from app.templating import render

log = logging.getLogger(__name__)
router = APIRouter(tags=["recommendations"])


def _items_with_products(db: Session, rec: Recommendation | None) -> list[dict]:
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


@router.get("/me")
def my_page(
    request: Request,
    background: BackgroundTasks,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    prof = db.get(UserProfile, user.id)
    rec = recommender.get_current(db, user.id)
    decision = triggers.evaluate(db, user.id, prof)

    if decision.run:
        background.add_task(recommender.generate_for_user, user.id)

    return render(
        request,
        "dashboard.html",
        rec=rec,
        items=_items_with_products(db, rec),
        evidence=profile_service.evidence(db, user.id) if prof else [],
        profile=prof,
        summary=profile_service.summarise(prof) if prof else "",
        decision=decision,
        pending=decision.run and rec is None,
    )


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
            for entry in _items_with_products(db, rec)
        ],
    }
