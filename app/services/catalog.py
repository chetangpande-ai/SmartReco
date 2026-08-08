"""Catalogue writes. Every mutation lands in SQL and the vector outbox in one commit.

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
from dataclasses import dataclass
from urllib.parse import quote_plus

from sqlalchemy import delete, func, or_, select
from sqlalchemy.orm import Session, selectinload

from app.models import CourseSkill, Product, VectorOutbox
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


# Held on ProductIn but written to `course_skills`, not to a Product column.
_SKILL_FIELDS = ("teaches", "requires")


def _columns(data: ProductIn) -> dict:
    return {k: v for k, v in data.model_dump().items() if k not in _SKILL_FIELDS}


def _write_skills(db: Session, p: Product, data: ProductIn) -> None:
    """Replace this product's skill edges wholesale.

    A diff would be cheaper and is not worth it: a course has a handful of skills, they
    change only when an admin edits the course, and "delete then insert" cannot leave a
    stale edge behind the way a partial diff can.
    """
    db.execute(delete(CourseSkill).where(CourseSkill.product_id == p.id))
    for kind in _SKILL_FIELDS:
        for position, slug in enumerate(getattr(data, kind)):
            db.add(
                CourseSkill(
                    product_id=p.id,
                    skill=slug,
                    # ProductIn calls them teaches/requires because that reads naturally
                    # at the call site; the column stores the same two values.
                    kind="teaches" if kind == "teaches" else "requires",
                    position=position,
                )
            )


def create_product(db: Session, data: ProductIn) -> Product:
    p = Product(**_columns(data), slug=unique_slug(db, slugify(data.title)))
    db.add(p)
    db.flush()  # assign p.id before the outbox row and skill rows reference it

    _write_skills(db, p, data)
    p.content_hash = compute_content_hash(p)
    enqueue_sync(db, p, "upsert" if p.is_published else "delete")
    db.commit()

    log.info("product created", extra={"id": p.id, "slug": p.slug})
    return p


def update_product(db: Session, p: Product, data: ProductIn) -> Product:
    for field, value in _columns(data).items():
        setattr(p, field, value)
    p.slug = unique_slug(db, slugify(data.title), exclude_id=p.id)
    _write_skills(db, p, data)
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


def by_skill(db: Session, skills, kind: str = "teaches") -> dict[str, list[Product]]:
    """{skill slug: published courses with that edge}. One query, not one per skill.

    The advisor resolves a whole gap list at once, and a per-skill query there is a loop
    over the database in the hot path of the page people came to see.
    """
    wanted = list(skills)
    if not wanted:
        return {}

    rows = db.execute(
        select(CourseSkill.skill, Product)
        .join(Product, CourseSkill.product_id == Product.id)
        .where(
            CourseSkill.skill.in_(wanted),
            CourseSkill.kind == kind,
            Product.is_published.is_(True),
        )
        # Callers rank on where a skill sits in the course's own list, so every returned
        # product gets its full skill set in one extra query rather than a lazy load per
        # row in the middle of ranking.
        .options(selectinload(Product.skills))
    ).all()

    found: dict[str, list[Product]] = {s: [] for s in wanted}
    for skill, product in rows:
        found[skill].append(product)
    return found


def with_format(db: Session, formats, skills=None, limit: int = 12) -> list[Product]:
    """Published courses of the given program types, optionally narrowed to ones
    touching a set of skills. Backs the marketplace's Bootcamps / Certifications /
    Projects shelves and the roadmap's project and certification stages."""
    stmt = select(Product).where(
        Product.is_published.is_(True), Product.format.in_(list(formats))
    )
    if skills:
        stmt = stmt.where(
            Product.id.in_(
                select(CourseSkill.product_id).where(CourseSkill.skill.in_(list(skills)))
            )
        )
    stmt = stmt.order_by(Product.rating.desc(), Product.reviews.desc()).limit(limit)
    return list(db.scalars(stmt))


def categories(db: Session) -> list[str]:
    stmt = (
        select(Product.category)
        .where(Product.is_published.is_(True))
        .group_by(Product.category)
        .order_by(func.count(Product.id).desc())
    )
    return list(db.scalars(stmt))


# Duration buckets, as the taxonomy's filter list writes them, mapped to the hour range
# they mean. Keyed by the label so the filter chip, the query string and the SQL all use
# one string and cannot drift apart.
DURATION_BUCKETS = {
    "0-2 hours": (0, 2),
    "2-10 hours": (2, 10),
    "10-30 hours": (10, 30),
    "30-60 hours": (30, 60),
    "60+ hours": (60, 100_000),
}

SORTS = {
    "popular": (Product.learners.desc(), Product.rating.desc()),
    "rating": (Product.rating.desc(), Product.reviews.desc()),
    "newest": (Product.created_at.desc(),),
    "shortest": (Product.duration_hours.asc(), Product.rating.desc()),
    "price-low": (Product.price_cents.asc(), Product.rating.desc()),
}
DEFAULT_SORT = "popular"


@dataclass
class CourseFilters:
    """Everything the marketplace can narrow by. One object so the route, the query and
    the template's "clear this filter" links all agree on the field names."""

    q: str = ""
    category: str = ""
    subcategory: str = ""
    skill: str = ""
    role: str = ""  # taxonomy role slug — expands to the skills that role needs
    level: str = ""
    fmt: str = ""
    duration: str = ""
    price: str = ""  # free | paid
    rating: str = ""  # "4.0" | "4.5"
    certificate: bool = False
    sort: str = DEFAULT_SORT

    # Query-string name -> attribute name. The template builds every filter link from
    # this and reads the current value back through it, so the two can never drift into
    # a chip that highlights one facet and toggles another. Deliberately un-annotated:
    # an annotation here would make dataclass treat it as a field.
    QUERY_FIELDS = {
        "q": "q", "category": "category", "sub": "subcategory", "skill": "skill",
        "role": "role", "level": "level", "format": "fmt", "duration": "duration",
        "price": "price", "rating": "rating", "sort": "sort",
    }

    def get(self, field: str):
        if field == "certificate":
            return "1" if self.certificate else ""
        return getattr(self, self.QUERY_FIELDS[field], "")

    def is_on(self, field: str, value: str) -> bool:
        return self.get(field) == value

    def query(self, **changes) -> str:
        """The current filter set as a query string, with some facets replaced.

        Empty values drop out, which is what makes a chip a toggle: setting a facet to
        "" is the same link as removing it.
        """
        parts = {name: self.get(name) for name in self.QUERY_FIELDS}
        parts["certificate"] = self.get("certificate")
        parts.update({k: "" if v is None else str(v) for k, v in changes.items()})
        return "&".join(f"{k}={quote_plus(v)}" for k, v in parts.items() if v)

    @property
    def active(self) -> list[tuple[str, str]]:
        """(query field, human label) for each applied filter, for the removable chips.

        Labelled here rather than in the template because only this side knows that
        `category=ai-ml` should read "AI & Machine Learning" — a chip showing the slug
        looks like a bug even though it is the correct value. Sort is excluded: it is
        always set, so a chip offering to clear it is noise.
        """
        from app import taxonomy

        label_for = {
            "category": lambda v: getattr(taxonomy.category(v), "name", v),
            "sub": lambda v: getattr(taxonomy.subcategory(v), "name", v),
            "skill": taxonomy.skill_name,
            "role": lambda v: getattr(taxonomy.role(v), "name", v),
            "rating": lambda v: f"{v} and up",
        }
        out = []
        for name in self.QUERY_FIELDS:
            value = self.get(name)
            if value and name != "sort":
                out.append((name, label_for.get(name, str)(value)))
        if self.certificate:
            out.append(("certificate", "Certificate available"))
        return out


def search(db: Session, f: CourseFilters, limit: int = 24, offset: int = 0):
    """(page of courses, total matching). Every filter is optional and they compose."""
    stmt = select(Product).where(Product.is_published.is_(True))

    if f.q:
        like = f"%{f.q.lower()}%"
        stmt = stmt.where(
            or_(
                func.lower(Product.title).like(like),
                func.lower(Product.description).like(like),
                func.lower(Product.brand).like(like),
                func.lower(Product.instructor).like(like),
                func.lower(Product.spec).like(like),
            )
        )
    if f.category:
        stmt = stmt.where(Product.category == f.category)
    if f.subcategory:
        stmt = stmt.where(Product.subcategory == f.subcategory)
    if f.level:
        stmt = stmt.where(Product.tier == f.level)
    if f.fmt:
        stmt = stmt.where(Product.format == f.fmt)
    if f.certificate:
        stmt = stmt.where(Product.certificate.is_(True))
    if f.price == "free":
        stmt = stmt.where(Product.price_cents == 0)
    elif f.price == "paid":
        stmt = stmt.where(Product.price_cents > 0)
    if f.rating:
        stmt = stmt.where(Product.rating >= float(f.rating))
    if f.duration in DURATION_BUCKETS:
        low, high = DURATION_BUCKETS[f.duration]
        stmt = stmt.where(Product.duration_hours > low, Product.duration_hours <= high)

    # Skill and role are the same filter at two granularities: a role is the set of
    # skills it needs. Expressed as one EXISTS over course_skills so a course matching
    # three of the role's skills still comes back once.
    wanted_skills: list[str] = []
    if f.skill:
        wanted_skills.append(f.skill)
    if f.role:
        from app import taxonomy

        role = taxonomy.role(f.role)
        if role:
            wanted_skills.extend(role.skills)
    if wanted_skills:
        stmt = stmt.where(
            Product.id.in_(
                select(CourseSkill.product_id).where(
                    CourseSkill.skill.in_(wanted_skills), CourseSkill.kind == "teaches"
                )
            )
        )

    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    ordering = SORTS.get(f.sort, SORTS[DEFAULT_SORT])
    page = db.scalars(
        stmt.order_by(*ordering, Product.id)
        .limit(limit)
        .offset(offset)
        .options(selectinload(Product.skills))
    )
    return list(page), total


def counts_by_category(db: Session) -> dict[str, int]:
    rows = db.execute(
        select(Product.category, func.count())
        .where(Product.is_published.is_(True))
        .group_by(Product.category)
    ).all()
    return dict(rows)


def counts_by_format(db: Session) -> dict[str, int]:
    rows = db.execute(
        select(Product.format, func.count())
        .where(Product.is_published.is_(True))
        .group_by(Product.format)
    ).all()
    return dict(rows)


def published_hashes(db: Session) -> dict[str, str]:
    """{vector_id: content_hash} for the published set — one side of the drift diff."""
    rows = db.execute(
        select(Product.id, Product.content_hash).where(Product.is_published.is_(True))
    ).all()
    return {str(pid): h for pid, h in rows}
