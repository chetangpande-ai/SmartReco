"""Product writes. Every mutation lands in SQL and the vector outbox in one commit.

The invariant this file maintains: **the vector index contains exactly the published
products, and nothing else.** Unpublishing enqueues a delete rather than setting a flag
the retrieval layer has to remember to filter on — an invariant you can't forget beats
a filter you can.
"""

import hashlib
import json
import logging
import re
import unicodedata

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.models import Product, VectorOutbox
from app.schemas import ProductIn

log = logging.getLogger(__name__)


def slugify(title: str) -> str:
    ascii_only = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_only.lower()).strip("-")
    return slug[:140] or "product"


def unique_slug(db: Session, base: str, exclude_id: int | None = None) -> str:
    slug, n = base, 1
    while True:
        stmt = select(Product.id).where(Product.slug == slug)
        if exclude_id is not None:
            stmt = stmt.where(Product.id != exclude_id)
        if db.scalar(stmt) is None:
            return slug
        n += 1
        slug = f"{base}-{n}"


def _vector_meta_core(p: Product) -> dict:
    """Everything we mirror into the vector store except the hash itself.

    Chroma and Pinecone both reject non-scalar metadata values, so tags are flattened
    to a comma string here rather than at each call site.
    """
    return {
        "product_id": p.id,
        "slug": p.slug,
        "title": p.title,
        "brand": p.brand,
        "category": p.category,
        "tier": p.tier,
        "price_cents": p.price_cents,
        "rating": p.rating,
        "spec": p.spec,
        "tags": ",".join(p.tags or []),
    }


def compute_content_hash(p: Product) -> str:
    """Hash of the vector store's *desired state* for this product.

    That means the embedded text, the mirrored metadata, and — easy to miss —
    `is_published`, because whether the row should exist in the index at all is part of
    the desired state. Leaving it out makes unpublishing hash-identical to publishing,
    so the update is treated as a no-op and the product silently stays recommendable.
    """
    payload = {
        "text": p.embedding_text(),
        "meta": _vector_meta_core(p),
        "published": bool(p.is_published),
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def vector_metadata(p: Product) -> dict:
    return {**_vector_meta_core(p), "content_hash": p.content_hash}


def enqueue_sync(db: Session, p: Product, op: str) -> None:
    """Stage an outbox row. Caller commits, so SQL and outbox move together or not at all.

    VectorOutbox.product_id is a plain Integer, not a ForeignKey — a delete row has to
    outlive the product it refers to.
    """
    db.add(
        VectorOutbox(
            product_id=p.id,
            op=op,
            payload={"slug": p.slug, "title": p.title, "hash": p.content_hash},
        )
    )


def create_product(db: Session, data: ProductIn) -> Product:
    p = Product(**data.model_dump(), slug=unique_slug(db, slugify(data.title)))
    db.add(p)
    db.flush()  # assign p.id before the outbox row references it

    p.content_hash = compute_content_hash(p)
    enqueue_sync(db, p, "upsert" if p.is_published else "delete")
    db.commit()

    log.info("product created", extra={"id": p.id, "slug": p.slug})
    return p


def update_product(db: Session, p: Product, data: ProductIn) -> Product:
    for field, value in data.model_dump().items():
        setattr(p, field, value)
    p.slug = unique_slug(db, slugify(data.title), exclude_id=p.id)
    db.flush()

    new_hash = compute_content_hash(p)
    if new_hash == p.content_hash:
        # Nothing about the vector store's desired state changed, so re-embedding here
        # would be pure spend. A visibility flip does change the hash — see above.
        db.commit()
        log.info("product updated, vector unchanged", extra={"id": p.id})
        return p

    p.content_hash = new_hash
    p.vector_synced_at = None
    enqueue_sync(db, p, "upsert" if p.is_published else "delete")
    db.commit()

    log.info("product updated, resync queued", extra={"id": p.id, "published": p.is_published})
    return p


def delete_product(db: Session, p: Product) -> int:
    product_id = p.id
    enqueue_sync(db, p, "delete")
    db.delete(p)
    db.commit()
    log.info("product deleted", extra={"id": product_id})
    return product_id


def list_products(
    db: Session,
    *,
    category: str | None = None,
    tier: str | None = None,
    q: str | None = None,
    published_only: bool = True,
    limit: int = 60,
    offset: int = 0,
) -> list[Product]:
    stmt = select(Product)
    if published_only:
        stmt = stmt.where(Product.is_published.is_(True))
    if category:
        stmt = stmt.where(Product.category == category)
    if tier:
        stmt = stmt.where(Product.tier == tier)
    if q:
        like = f"%{q.lower()}%"
        stmt = stmt.where(
            or_(
                func.lower(Product.title).like(like),
                func.lower(Product.description).like(like),
                func.lower(Product.category).like(like),
                func.lower(Product.brand).like(like),
            )
        )
    stmt = stmt.order_by(Product.rating.desc(), Product.id).limit(limit).offset(offset)
    return list(db.scalars(stmt))


def categories(db: Session) -> list[str]:
    stmt = (
        select(Product.category)
        .where(Product.is_published.is_(True))
        .group_by(Product.category)
        .order_by(func.count(Product.id).desc())
    )
    return list(db.scalars(stmt))


def published_hashes(db: Session) -> dict[str, str]:
    """{vector_id: content_hash} for the published set — one side of the drift diff."""
    rows = db.execute(
        select(Product.id, Product.content_hash).where(Product.is_published.is_(True))
    ).all()
    return {str(pid): h for pid, h in rows}
