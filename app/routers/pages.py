"""Server-rendered pages. Tracking is attached declaratively via data- attributes so
tracker.js needs no per-page glue."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Product
from app.services import catalog
from app.templating import render

log = logging.getLogger(__name__)
router = APIRouter(tags=["pages"])


@router.get("/")
def home(
    request: Request,
    category: str | None = None,
    tier: str | None = None,
    db: Session = Depends(get_db),
):
    products = catalog.list_products(db, category=category, tier=tier)
    return render(
        request,
        "catalog.html",
        products=products,
        categories=catalog.categories(db),
        active_category=category,
        active_tier=tier,
    )


@router.get("/search")
def search(request: Request, q: str = "", db: Session = Depends(get_db)):
    query = q.strip()[:120]
    products = catalog.list_products(db, q=query) if query else []
    return render(request, "search.html", products=products, query=query)


@router.get("/products/{slug}")
def product_detail(request: Request, slug: str, db: Session = Depends(get_db)):
    product = db.scalar(select(Product).where(Product.slug == slug))
    if product is None or not product.is_published:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Product not found")

    related = [
        p
        for p in catalog.list_products(db, category=product.category, limit=4)
        if p.id != product.id
    ][:3]
    return render(request, "product.html", product=product, related=related)
