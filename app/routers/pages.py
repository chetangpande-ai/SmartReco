"""Server-rendered pages. Tracking is attached declaratively via data- attributes so
tracker.js needs no per-page glue."""

import logging

from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import taxonomy
from app.db import get_db
from app.deps import get_current_user, require_user, verify_csrf
from app.models import CareerProfile, Product, User
from app.services import advisor, catalog, learning, shelves
from app.templating import render

log = logging.getLogger(__name__)
router = APIRouter(tags=["pages"])

# What the signed-out home page leads with. Enough to show the shape of the platform
# without pretending to know anything about the visitor.
HOME_PATHS = ("qa-to-ai", "ai-engineer", "data-scientist", "devops-engineer")


@router.get("/")
def home(
    request: Request,
    category: str | None = None,
    tier: str | None = None,
    user: User | None = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    products = catalog.list_products(db, category=category, tier=tier)
    profile = db.get(CareerProfile, user.id) if user else None
    return render(
        request,
        "catalog.html",
        products=products,
        categories=catalog.categories(db),
        active_category=category,
        active_tier=tier,
        career_profile=profile,
        plan=advisor.current_plan(db, user.id) if user else None,
        home_paths=[taxonomy.path(s) for s in HOME_PATHS if taxonomy.path(s)],
        counts=catalog.counts_by_category(db),
    )


@router.get("/search")
def search(request: Request, q: str = "", db: Session = Depends(get_db)):
    query = q.strip()[:120]
    products = catalog.list_products(db, q=query) if query else []
    return render(
        request,
        "search.html",
        products=products,
        query=query,
        categories=catalog.categories(db),
    )


@router.get("/support")
def support(request: Request):
    return render(request, "support.html")


def _product_or_404(db: Session, slug: str) -> Product:
    product = db.scalar(select(Product).where(Product.slug == slug))
    if product is None or not product.is_published:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Product not found")
    return product


def _detail_context(db: Session, product: Product, user: User | None) -> dict:
    """Everything the course page shows besides the course itself."""
    related_paths = [
        p
        for p in taxonomy.paths().values()
        if set(p.skills) & set(product.teaches)
    ][:4]
    return {
        "product": product,
        # Content-similar rather than same-category: the hybrid retriever already knows
        # what "similar" means here, and "more in Cloud Computing" is a weaker answer.
        "related": shelves.similar_to(db, product, limit=4),
        "enrollment": learning.get(db, user.id, product.id) if user else None,
        "roles": [
            r for r in taxonomy.roles().values() if set(r.skills) & set(product.teaches)
        ][:6],
        "paths": related_paths,
        "prerequisites": [
            {"slug": s, "name": taxonomy.skill_name(s)} for s in product.requires
        ],
    }


@router.get("/products/{slug}")
def product_detail(
    request: Request,
    slug: str,
    user: User | None = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    product = _product_or_404(db, slug)
    return render(
        request,
        "product.html",
        categories=catalog.categories(db),
        active_category=product.category,
        **_detail_context(db, product, user),
    )


@router.post("/products/{slug}/fit", dependencies=[Depends(verify_csrf)])
def course_fit(
    request: Request,
    slug: str,
    user: User | None = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """"Ask AI if this course is right for me."

    Answered inline on a re-render rather than through a fetch: the strict CSP forbids
    inline script, the answer is not worth a JS bundle, and a form post works with the
    back button and with scripting off.
    """
    product = _product_or_404(db, slug)
    return render(
        request,
        "product.html",
        categories=catalog.categories(db),
        active_category=product.category,
        fit=advisor.course_fit(db, product, user.id if user else None),
        **_detail_context(db, product, user),
    )


@router.post("/products/{slug}/enroll", dependencies=[Depends(verify_csrf)])
def enroll(
    slug: str,
    action: str = Form("start"),
    progress: int = Form(0),
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    """Start Learning, Save for later, and the progress control on My Learning."""
    product = _product_or_404(db, slug)
    if action == "save":
        learning.save(db, user.id, product)
    elif action == "progress":
        learning.set_progress(db, user.id, product, progress)
    else:
        learning.start(db, user.id, product)
    db.commit()

    back = "/me" if action == "progress" else f"/products/{slug}"
    return RedirectResponse(back, status_code=status.HTTP_303_SEE_OTHER)
