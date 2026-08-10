"""Admin: catalogue management plus the operational view of the system.

The ops page exists because the two claims this project makes that are easiest to fake
— "the stores really are in sync" and "we really do avoid redundant LLM calls" — should
be readable off a dashboard rather than taken on trust.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from pydantic import ValidationError
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, selectinload

from app import scheduler, taxonomy
from app.agent.graph import describe
from app.config import settings
from app.db import get_db
from app.deps import require_admin, verify_csrf
from app.models import (
    AgentRun,
    CareerProfile,
    Event,
    Notification,
    Product,
    Recommendation,
    RecommendationItem,
    User,
    UserProfile,
    VectorOutbox,
)
from app.schemas import ProductIn
from app.services import catalog, notify, outbox, profile, recommender, triggers
from app.services.mesh import mesh
from app.templating import render

log = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])


def _form_to_product(form) -> ProductIn:
    return ProductIn(
        title=form.get("title", ""),
        description=form.get("description", ""),
        category=form.get("category", "").strip().lower(),
        tier=form.get("tier", "beginner"),
        tags=[t.strip() for t in form.get("tags", "").split(",") if t.strip()],
        price_cents=int(float(form.get("price", 0) or 0) * 100),
        brand=form.get("brand", ""),
        spec=form.get("spec", ""),
        rating=float(form.get("rating", 0) or 0),
        is_published=form.get("is_published") == "on",
    )


@router.get("")
def ops_dashboard(request: Request, db: Session = Depends(get_db)):
    counts = {
        "users": db.scalar(select(func.count()).select_from(User)),
        "courses": db.scalar(select(func.count()).select_from(Product)),
        "events": db.scalar(select(func.count()).select_from(Event)),
        "recommendations": db.scalar(select(func.count()).select_from(Recommendation)),
    }
    events_by_type = dict(
        db.execute(
            select(Event.type, func.count()).group_by(Event.type).order_by(func.count().desc())
        ).all()
    )
    return render(
        request,
        "admin/ops.html",
        counts=counts,
        events_by_type=events_by_type,
        sync=outbox.health(),
        efficiency=triggers.efficiency_stats(db),
        llm=mesh.status(),
        runs=recommender.recent_runs(db, 15),
        graph=describe(),
        jobs=scheduler.job_status(),
        notifications=list(
            db.scalars(select(Notification).order_by(Notification.created_at.desc()).limit(8))
        ),
        mail_backend=notify.get_notifier().backend,
        dead_rows=list(
            db.scalars(
                select(VectorOutbox).where(VectorOutbox.status == "dead").limit(10)
            )
        ),
    )


@router.get("/products")
def list_products(request: Request, q: str = "", db: Session = Depends(get_db)):
    products = catalog.list_products(db, q=q or None, published_only=False, limit=300)
    return render(request, "admin/products.html", products=products, q=q)


@router.get("/products/new")
def new_product_form(request: Request):
    return render(request, "admin/product_form.html", product=None, error=None)


@router.post("/products", dependencies=[Depends(verify_csrf)])
async def create(request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    try:
        data = _form_to_product(form)
    except (ValidationError, ValueError) as exc:
        return render(
            request, "admin/product_form.html", product=None, error=str(exc)[:300], status_code=400
        )
    product = catalog.create_product(db, data)
    # Nudge the outbox now so the admin sees "in sync" immediately rather than waiting
    # for the scheduler's next tick. Correctness does not depend on this — the worker
    # would pick it up anyway — it is purely for the feedback loop.
    outbox.drain_all()
    return RedirectResponse(f"/admin/products?q={product.title[:20]}", status_code=303)


@router.get("/products/{product_id}/edit")
def edit_form(request: Request, product_id: int, db: Session = Depends(get_db)):
    product = db.get(Product, product_id)
    if product is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Product not found")
    return render(request, "admin/product_form.html", product=product, error=None)


@router.post("/products/{product_id}", dependencies=[Depends(verify_csrf)])
async def update(request: Request, product_id: int, db: Session = Depends(get_db)):
    product = db.get(Product, product_id)
    if product is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Product not found")
    form = await request.form()
    try:
        data = _form_to_product(form)
    except (ValidationError, ValueError) as exc:
        return render(
            request, "admin/product_form.html", product=product, error=str(exc)[:300], status_code=400
        )
    catalog.update_product(db, product, data)
    outbox.drain_all()
    return RedirectResponse("/admin/products", status_code=303)


@router.post("/products/{product_id}/delete", dependencies=[Depends(verify_csrf)])
def delete(product_id: int, db: Session = Depends(get_db)):
    product = db.get(Product, product_id)
    if product is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Product not found")
    catalog.delete_product(db, product)
    outbox.drain_all()
    return RedirectResponse("/admin/products", status_code=303)


@router.post("/reconcile", dependencies=[Depends(verify_csrf)])
def reconcile():
    result = outbox.reconcile()
    log.info("manual reconcile", extra=result)
    return RedirectResponse("/admin", status_code=303)


@router.post("/reindex", dependencies=[Depends(verify_csrf)])
def reindex():
    result = outbox.reindex_all()
    log.info("manual reindex", extra={"result": str(result)[:200]})
    return RedirectResponse("/admin", status_code=303)


@router.post("/send-digest", dependencies=[Depends(verify_csrf)])
def send_digest_now():
    """Manual trigger for the same job the scheduler runs. Useful for a demo, and it
    exercises the identical code path — including the once-per-day dedupe key, so
    pressing it twice sends one email, not two."""
    from app.services.digest import send_daily_digests

    result = send_daily_digests()
    log.info("manual digest run", extra=result)
    return RedirectResponse("/admin", status_code=303)


@router.get("/users")
def list_users(request: Request, q: str = "", db: Session = Depends(get_db)):
    stmt = select(User).order_by(User.created_at.desc()).limit(200)
    if q.strip():
        like = f"%{q.strip()}%"
        stmt = stmt.where(or_(User.name.ilike(like), User.email.ilike(like)))
    users = list(db.scalars(stmt))

    # Two grouped queries rather than a count per row. The seed alone makes 24 synthetic
    # learners, so a per-user count inside the loop is quadratic on the page that exists
    # to survey everyone.
    ids = [u.id for u in users] or [-1]
    def totals(column) -> dict[int, int]:
        return dict(
            db.execute(
                select(column, func.count()).where(column.in_(ids)).group_by(column)
            ).all()
        )

    return render(
        request,
        "admin/users.html",
        users=users,
        q=q,
        event_counts=totals(Event.user_id),
        rec_counts=totals(Recommendation.user_id),
    )


@router.get("/users/{user_id}")
def user_detail(request: Request, user_id: int, db: Session = Depends(get_db)):
    """Everything the system captured about one learner, and what it recommended from it.

    The evidence list is read through `profile.evidence()` — the same call the agent
    makes — rather than rebuilt here. A second reconstruction would be free to drift from
    what the model was actually handed, which is the one thing this page is for.
    """
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such user")

    # A target role is a role slug, not a skill slug — the `skill` filter would fall back
    # to title-casing it and render "Ai Engineer".
    career = db.get(CareerProfile, user_id)
    target_role = taxonomy.role(career.target_role) if career else None

    return render(
        request,
        "admin/user_detail.html",
        user=user,
        user_profile=db.get(UserProfile, user_id),
        career=career,
        target_role=target_role,
        evidence=profile.evidence(db, user_id),
        events_by_type=dict(
            db.execute(
                select(Event.type, func.count())
                .where(Event.user_id == user_id)
                .group_by(Event.type)
                .order_by(func.count().desc())
            ).all()
        ),
        events=list(
            db.scalars(
                select(Event)
                .where(Event.user_id == user_id)
                .options(selectinload(Event.product))
                .order_by(Event.server_ts.desc())
                .limit(30)
            )
        ),
        recommendations=list(
            db.scalars(
                select(Recommendation)
                .where(Recommendation.user_id == user_id)
                .options(selectinload(Recommendation.items).selectinload(RecommendationItem.product))
                .order_by(Recommendation.created_at.desc())
                .limit(10)
            )
        ),
        runs=list(
            db.scalars(
                select(AgentRun)
                .where(AgentRun.user_id == user_id)
                .order_by(AgentRun.created_at.desc())
                .limit(20)
            )
        ),
    )


@router.get("/agent-runs")
def agent_runs(request: Request, db: Session = Depends(get_db)):
    runs = recommender.recent_runs(db, 100)
    totals = {
        "runs": len(runs),
        "llm_calls": sum(r.llm_calls for r in runs),
        "tokens": sum(r.prompt_tokens + r.completion_tokens for r in runs),
        "cost_usd": round(sum(r.cost_usd for r in runs), 6),
        "avg_latency_ms": int(sum(r.latency_ms for r in runs) / len(runs)) if runs else 0,
        "errors": sum(1 for r in runs if r.status != "ok"),
    }
    return render(
        request, "admin/agent_runs.html", runs=runs, totals=totals, settings=settings
    )
