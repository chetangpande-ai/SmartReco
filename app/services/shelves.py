"""The one browse row on the learner's home page that does not need the agent.

The agent's picks are the expensive, reasoned answer to "what should I learn next?" —
they cost tokens and land seconds later on a background run. This is the cheap row
beside them: content-similar to the last course the learner opened, answerable from the
behaviour profile alone, inside the request that renders the page.

There were four rows here. Three are gone, and the reasons are worth keeping:

  * **"Let's start learning"** is now the enrollment list on the dashboard, which knows
    actual progress. Deriving the same row from `enroll` *events* put two versions of
    "where was I?" on one page, and the events one could not show a progress bar.
  * **"Recommended based on ratings"** was the catalogue's popularity with a personalised
    label on it. Next to a real behavioural recommendation it read as filler.
  * **Topic chips** duplicated the category bar in the header.

Two rules still hold and are why this is a module rather than a list comprehension in
the router: nothing repeats what the page already showed above it, and nothing the
learner dismissed comes back — "not interested" is the only direct control they have.
"""

import logging
from dataclasses import dataclass, field

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Event, Product
from app.services import retrieval
from app.services.profile import DISMISSED_EVENTS

log = logging.getLogger(__name__)

ROW_SIZE = 8

# Below this a row reads as a page that failed to load the rest rather than as a shelf.
MIN_ROW = 3

# Opening a course. `rec_impression` is deliberately absent: a card scrolling past is
# exposure we manufactured, not interest the learner expressed, and anchoring "because
# you viewed" on it would recommend against our own suggestions.
VIEWED_EVENTS = ("product_view", "product_click", "search_result_click", "rec_click")


@dataclass
class Shelf:
    """One horizontal row of courses.

    `key` is both the DOM anchor and the `source` recorded on every click inside the
    row, so click-through is attributable per row instead of lumped together as "the
    dashboard".
    """

    key: str
    title: str
    subtitle: str
    products: list[Product] = field(default_factory=list)


def build(db: Session, user_id: int, exclude_ids: set[int] | None = None) -> list[Shelf]:
    """The browse rows for one learner, in display order.

    `exclude_ids` is what the page already shows above these rows — the agent's own
    picks and whatever the learner is part-way through. Passing them in is what stops
    the page repeating itself under a second heading.
    """
    used = set(exclude_ids or ()) | _dismissed(db, user_id)
    shelf = _because_you_viewed(db, user_id, used)
    return [shelf] if shelf else []


def similar_to(db: Session, anchor: Product, limit: int = 4) -> list[Product]:
    """Content-similar courses, using the same hybrid retrieval the agent uses.

    Shared by the "because you viewed" row and the course page's related list. Both mean
    the same thing by "similar", so both go through the same code — the alternative, a
    same-category SQL query on the product page, quietly answers a different question.
    """
    result = retrieval.retrieve(
        db,
        query_text=anchor.embedding_text(),
        exclude_ids={anchor.id},
        top_n=limit,
        mmr_lambda=0.85,
    )
    return _products(db, result.ids())


def _because_you_viewed(db: Session, user_id: int, exclude: set[int]) -> Shelf | None:
    """More like the last course the learner opened.

    Runs the same hybrid retrieval the agent uses, with the course itself as the query,
    so "similar" means here what it means everywhere else in the system — and degrades
    to BM25 rather than to nothing when the vector index is cold.
    """
    anchors = _recent_ids(db, user_id, VIEWED_EVENTS, 1)
    if not anchors:
        return None
    anchor = db.get(Product, anchors[0])
    if anchor is None or not anchor.is_published:
        return None

    result = retrieval.retrieve(
        db,
        query_text=anchor.embedding_text(),
        exclude_ids=exclude | {anchor.id},
        top_n=ROW_SIZE,
        # Above the agent's 0.65 default: this row makes a promise about closeness to
        # one named course, so relevance should outweigh diversity. The agent wants the
        # opposite — its picks should span a journey rather than repeat a point.
        mmr_lambda=0.85,
    )
    products = _products(db, result.ids())
    if len(products) < MIN_ROW:
        return None
    return Shelf(
        key="because-viewed",
        title=f"Because you viewed “{anchor.title}”",
        subtitle="Closest matches in the catalogue, by content rather than by track.",
        products=products,
    )


def _dismissed(db: Session, user_id: int) -> set[int]:
    return set(
        db.scalars(
            select(Event.product_id).where(
                Event.user_id == user_id,
                Event.type.in_(DISMISSED_EVENTS),
                Event.product_id.isnot(None),
            )
        )
    )


def _recent_ids(db: Session, user_id: int, types: tuple[str, ...], limit: int) -> list[int]:
    """Distinct products from this learner's events of `types`, most recent first.

    Grouped in SQL rather than deduplicated in Python: someone who opened one course
    nine times would otherwise fill the entire row with it.
    """
    return list(
        db.scalars(
            select(Event.product_id)
            .where(
                Event.user_id == user_id,
                Event.type.in_(types),
                Event.product_id.isnot(None),
            )
            .group_by(Event.product_id)
            .order_by(func.max(Event.server_ts).desc())
            .limit(limit)
        )
    )


def _products(db: Session, ids: list[int]) -> list[Product]:
    """Published products for `ids`, in the order given — `IN` does not preserve it, and
    the order is the whole point, since the row is a ranked list."""
    if not ids:
        return []
    found = {
        p.id: p
        for p in db.scalars(
            select(Product).where(Product.id.in_(ids), Product.is_published.is_(True))
        )
    }
    return [found[i] for i in ids if i in found]
