"""Admin: catalogue management plus the operational view of the system.

The ops page exists because the two claims this project makes that are easiest to fake
— "the stores really are in sync" and "we really do avoid redundant LLM calls" — should
be readable off a dashboard rather than taken on trust.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import scheduler
from app.agent.graph import describe
from app.config import settings
from app.db import get_db
from app.deps import require_admin, verify_csrf
from app.models import Event, Notification, Product, Recommendation, User, VectorOutbox
from app.schemas import ProductIn
from app.services import catalog, notify, outbox, recommender, triggers
from app.services.mesh import mesh
from app.templating import render

log = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])


def _form_to_product(form) -> ProductIn:
    return ProductIn(
        title=form.get("title", ""),
        description=form.get("description", ""),
        category=form.get("category", "").strip().lower(),
        tier=form.get("tier", "entry"),
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
        "products": db.scalar(select(func.count()).select_from(Product)),
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


@router.get("/agent-runs")
def agent_runs(request: Request, db: Session = Depends(get_db)):
    runs = recommender.recent_runs(db, 100)
    users = {
        u.id: u
        for u in db.scalars(select(User).where(User.id.in_([r.user_id for r in runs] or [-1])))
    }
    totals = {
        "runs": len(runs),
        "llm_calls": sum(r.llm_calls for r in runs),
        "tokens": sum(r.prompt_tokens + r.completion_tokens for r in runs),
        "cost_usd": round(sum(r.cost_usd for r in runs), 6),
        "avg_latency_ms": int(sum(r.latency_ms for r in runs) / len(runs)) if runs else 0,
        "errors": sum(1 for r in runs if r.status != "ok"),
    }
    return render(
        request, "admin/agent_runs.html", runs=runs, users=users, totals=totals, settings=settings
    )
