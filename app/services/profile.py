"""Turn raw events into a behaviour profile the agent can reason about.

Three ideas do the work:

  * **Recency decay.** Every event's weight halves every `profile_halflife_hours`
    (48h by default). Interest is a moving thing — a topic someone abandoned last week
    should not outvote what they are reading right now, but one session should not
    erase a month of consistent behaviour either. A half-life expresses that trade-off
    in one number you can tune.

  * **Intent weighting.** Not all events mean the same thing. A search is someone
    telling you what they want; a page view is them wandering past. A purchase is worth
    thirty page views. The weights below encode that ordering.

  * **An interest centroid.** A single vector summarising what the user engages with,
    built from the same embedding space as the catalogue. It is what makes "have this
    person's interests actually *changed*?" a number rather than a guess — which is how
    the trigger policy avoids re-running the agent for no reason.
"""

import hashlib
import logging
from collections import defaultdict
from datetime import timedelta

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Event, Product, UserProfile, ensure_utc, utcnow
from app.services.embeddings import embedder
from app.services.retrieval import tokenize

log = logging.getLogger(__name__)

# Relative worth of each signal. Explicit intent (search, wishlist, enrol) beats passive
# exposure (page_view, impression) by design.
#
# `purchase` and `add_to_cart` are kept as aliases of `enroll` and `wishlist`: events
# already in the database under the old names must keep counting, and a tracker.js still
# sitting in someone's browser cache will go on sending them for a while.
EVENT_WEIGHTS = {
    "enroll": 5.0,
    "purchase": 5.0,
    "wishlist": 3.0,
    "add_to_cart": 3.0,
    "search": 2.0,
    "search_result_click": 1.8,
    "rec_click": 1.6,
    "product_click": 1.2,
    "product_view": 1.0,
    "filter": 0.5,
    "scroll_depth": 0.25,
    "page_view": 0.15,
    "rec_impression": 0.05,
}
COMMITTED_EVENTS = ("enroll", "purchase", "wishlist", "add_to_cart")


def _dwell_weight(dwell_ms: int) -> float:
    """A minute of attention is worth about one product view; three minutes caps it.

    Capped because dwell is the easiest signal to corrupt — a tab left open, a user who
    walked away. Without a ceiling one stale tab dominates the whole profile.
    """
    return min(dwell_ms / 60_000.0, 3.0)


def refresh(db: Session, user_id: int) -> UserProfile | None:
    profile = db.get(UserProfile, user_id)
    if profile is None:
        profile = UserProfile(user_id=user_id)
        db.add(profile)

    # Sessions are autoflush=False, so any events the caller staged but has not
    # committed would be invisible to the queries below and silently under-count.
    db.flush()

    since = utcnow() - timedelta(days=settings.profile_lookback_days)
    rows = db.execute(
        select(
            Event.type,
            Event.query,
            Event.dwell_ms,
            Event.server_ts,
            Event.product_id,
            Product.category,
            Product.tier,
            Product.price_cents,
            Product.tags,
            Product.title,
            Product.brand,
        )
        .join(Product, Event.product_id == Product.id, isouter=True)
        .where(Event.user_id == user_id, Event.server_ts >= since)
        .order_by(Event.server_ts.desc())
        .limit(settings.profile_max_events)
    ).all()

    if not rows:
        profile.updated_at = utcnow()
        return profile

    now = utcnow()
    half_life = settings.profile_halflife_hours

    interests: dict[str, float] = defaultdict(float)
    tag_scores: dict[str, float] = defaultdict(float)
    term_scores: dict[str, float] = defaultdict(float)
    tier_scores: dict[str, float] = defaultdict(float)
    brand_scores: dict[str, float] = defaultdict(float)
    price_weighted, price_weight_total = 0.0, 0.0
    queries: list[str] = []
    viewed: list[int] = []
    # text -> accumulated weight, deduplicated so the centroid pays for each distinct
    # string once no matter how many times it was seen.
    centroid_inputs: dict[str, float] = defaultdict(float)

    # Vocabulary comes from the whole catalogue, not from this user's own history.
    # Sourcing it from their events means someone who has only ever searched — no
    # product views at all — has an empty vocabulary and gets no category signal,
    # which is precisely the user whose intent is clearest.
    known_categories, known_tags = _catalog_vocabulary(db)

    for r in rows:
        age_hours = max(0.0, (now - ensure_utc(r.server_ts)).total_seconds() / 3600.0)
        decay = 0.5 ** (age_hours / half_life)
        base = _dwell_weight(r.dwell_ms) if r.type == "dwell" else EVENT_WEIGHTS.get(r.type, 0.1)
        weight = base * decay
        if weight <= 0.0005:
            continue  # older than ~11 half-lives; it cannot change any ranking

        if r.category:
            interests[r.category] += weight
            tier_scores[r.tier or ""] += weight
            if r.brand:
                brand_scores[r.brand] += weight
            for tag in r.tags or []:
                tag_scores[tag] += weight
            for tok in tokenize(r.title):
                term_scores[tok] += weight * 0.5
            if r.price_cents:
                price_weighted += r.price_cents * weight
                price_weight_total += weight
            if r.product_id and r.product_id not in viewed:
                viewed.append(r.product_id)
            centroid_inputs[
                f"{r.title}. {r.brand}. {r.category}. {' '.join(r.tags or [])}"
            ] += weight

        if r.type == "search" and r.query:
            q = r.query.strip()
            if q and q.lower() not in {x.lower() for x in queries}:
                queries.append(q)
            centroid_inputs[q] += weight
            for tok in tokenize(q):
                term_scores[tok] += weight
                # A search naming a category or tag is an explicit vote for it. Without
                # this a user who only ever searches has no category signal at all.
                if tok in known_categories:
                    interests[tok] += weight
                if tok in known_tags:
                    tag_scores[tok] += weight

    profile.interests = {k: round(v, 4) for k, v in _top(interests, 12)}
    profile.tag_scores = {k: round(v, 4) for k, v in _top(tag_scores, 20)}
    profile.top_terms = [k for k, _ in _top(term_scores, 15)]
    profile.recent_queries = queries[:10]
    profile.viewed_product_ids = viewed[:40]
    profile.tier_affinity = max(tier_scores, key=tier_scores.get) if tier_scores else ""
    profile.brand_scores = {k: round(v, 4) for k, v in _top(brand_scores, 8)}
    profile.price_affinity_cents = (
        int(price_weighted / price_weight_total) if price_weight_total else 0
    )
    profile.centroid = _centroid(centroid_inputs, db)
    profile.events_total = len(rows)
    profile.last_event_at = ensure_utc(rows[0].server_ts)
    profile.events_since_rec = _count_since(db, user_id, ensure_utc(profile.last_rec_at))
    profile.updated_at = now

    log.debug(
        "profile refreshed",
        extra={
            "user_id": user_id,
            "events": len(rows),
            "top": list(profile.interests)[:3],
            "since_rec": profile.events_since_rec,
        },
    )
    return profile


def _catalog_vocabulary(db: Session) -> tuple[set[str], set[str]]:
    """Category and tag words the published catalogue actually uses."""
    rows = db.execute(
        select(Product.category, Product.tags).where(Product.is_published.is_(True))
    ).all()
    categories = {c for c, _ in rows if c}
    tags = {t for _, tag_list in rows for t in (tag_list or [])}
    return categories, tags


def _top(scores: dict[str, float], n: int) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda kv: -kv[1])[:n]


def _centroid(weighted_texts: dict[str, float], db: Session) -> bytes | None:
    if not weighted_texts:
        return None
    texts = list(weighted_texts)
    vectors = embedder.embed_documents(texts, db=db)
    weights = np.array([weighted_texts[t] for t in texts], dtype=np.float32)[:, None]
    summed = (vectors * weights).sum(axis=0)
    norm = np.linalg.norm(summed)
    if norm == 0:
        return None
    return (summed / norm).astype(np.float32).tobytes()


def _count_since(db: Session, user_id: int, since) -> int:
    from sqlalchemy import func

    stmt = select(func.count()).select_from(Event).where(Event.user_id == user_id)
    if since is not None:
        stmt = stmt.where(Event.server_ts > since)
    return int(db.scalar(stmt) or 0)


def as_vector(blob: bytes | None) -> np.ndarray | None:
    if not blob:
        return None
    return np.frombuffer(blob, dtype=np.float32)


def drift(current: bytes | None, previous: bytes | None) -> float:
    """1 - cosine similarity between the interest vector now and when we last generated.

    0.0 means "identical interests, a new recommendation would say the same thing".
    Returns 1.0 when there is nothing to compare against, so a first run always fires.
    """
    a, b = as_vector(current), as_vector(previous)
    if a is None or b is None or a.shape != b.shape:
        return 1.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 1.0
    return float(np.clip(1.0 - (a @ b) / denom, 0.0, 2.0))


def signature(profile: UserProfile) -> str:
    """Stable fingerprint of *what the user is into*, used as the cache key.

    Deliberately built from the ranked order of the top terms rather than their scores:
    raw floats drift on every single event, which would make the cache never hit. The
    ordering only changes when interests genuinely reshuffle.
    """
    parts = [
        ",".join(list(profile.interests)[:3]),
        ",".join(list(profile.tag_scores)[:5]),
        profile.tier_affinity,
        ",".join(list(profile.brand_scores)[:2]),
        # Price bucketed: a few dollars' difference is not a different learner. Bands
        # widen with price, because $20 matters on a $50 course and not on a $2000 cohort.
        str(int((profile.price_affinity_cents / 100) ** 0.5) // 3),
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def summarise(profile: UserProfile) -> str:
    """One-line human summary, used in prompts and the admin dashboard."""
    if not profile.interests:
        return "no activity yet"
    cats = ", ".join(f"{k} ({v:.1f})" for k, v in list(profile.interests.items())[:3])
    bits = [f"studying: {cats}"]
    if profile.recent_queries:
        bits.append(f"searched: {'; '.join(profile.recent_queries[:3])}")
    if profile.brand_scores:
        bits.append(f"providers looked at: {', '.join(list(profile.brand_scores)[:3])}")
    if profile.tier_affinity:
        bits.append(f"level: {profile.tier_affinity}")
    if profile.price_affinity_cents:
        bits.append(f"typical price: ${profile.price_affinity_cents / 100:,.0f}")
    return " | ".join(bits)


def evidence(db: Session, user_id: int, limit: int = 8) -> list[str]:
    """Concrete, checkable statements about what this user actually did.

    This is the only behaviour the copy is allowed to cite. Handing the model a list of
    facts instead of raw events is what stops it inventing a plausible-sounding history
    — and it doubles as the "why am I seeing this?" explanation shown to the user.
    """
    since = utcnow() - timedelta(days=settings.profile_lookback_days)
    rows = db.execute(
        select(
            Event.type, Event.query, Event.dwell_ms, Event.product_id, Product.title,
            Product.category, Product.tier, Product.brand,
        )
        .join(Product, Event.product_id == Product.id, isouter=True)
        .where(Event.user_id == user_id, Event.server_ts >= since)
        .order_by(Event.server_ts.desc())
        .limit(settings.profile_max_events)
    ).all()

    searches: dict[str, int] = defaultdict(int)
    views: dict[str, int] = defaultdict(int)
    dwell: dict[str, int] = defaultdict(int)
    categories: dict[str, int] = defaultdict(int)
    tiers: dict[str, int] = defaultdict(int)
    brands: dict[str, int] = defaultdict(int)
    intents: list[str] = []

    for r in rows:
        if r.type == "search" and r.query:
            searches[r.query.strip().lower()] += 1
        elif r.type in ("product_view", "product_click") and r.title:
            views[r.title] += 1
            categories[r.category] += 1
            tiers[r.tier] += 1
            if r.brand:
                brands[r.brand] += 1
        elif r.type == "dwell" and r.title:
            dwell[r.title] += r.dwell_ms
        elif r.type in COMMITTED_EVENTS and r.title:
            verb = "enrolled in" if r.type in ("enroll", "purchase") else "saved"
            statement = f"{verb} {r.title}"
            if statement not in intents:
                intents.append(statement)

    facts: list[str] = list(intents)
    for query, count in sorted(searches.items(), key=lambda kv: -kv[1])[:3]:
        facts.append(
            f"searched for '{query}'" + (f" {count} times" if count > 1 else "")
        )
    for title, ms in sorted(dwell.items(), key=lambda kv: -kv[1])[:3]:
        if ms >= 20_000:
            minutes, seconds = ms // 60000, (ms % 60000) // 1000
            spent = f"{minutes}m {seconds}s" if minutes and seconds else (
                f"{minutes}m" if minutes else f"{seconds}s"
            )
            facts.append(f"spent {spent} reading '{title}'")
    for title, count in sorted(views.items(), key=lambda kv: -kv[1])[:3]:
        facts.append(f"looked at {title}" + (f" {count} times" if count > 1 else ""))
    if categories:
        top_cat, n = max(categories.items(), key=lambda kv: kv[1])
        facts.append(f"opened {n} {top_cat.replace('-', ' ')} course{'s' if n != 1 else ''}")
    if brands:
        top_brand, n = max(brands.items(), key=lambda kv: kv[1])
        if n >= 2:
            facts.append(f"keep coming back to {top_brand}")
    if tiers:
        top_tier, n = max(tiers.items(), key=lambda kv: kv[1])
        if n >= 2:
            facts.append(f"favour {top_tier} material")

    # Every fact is phrased as a bare past-tense predicate so callers can prefix it with
    # "You " and get a sentence. The fallback copy writer depends on that.
    return facts[:limit]


def refresh_users(user_ids: set[int]) -> None:
    """Called by the ingest worker after a batch lands."""
    from app.db import session_scope

    for user_id in user_ids:
        try:
            with session_scope() as db:
                refresh(db, user_id)
        except Exception:
            # One user's bad data must not stop the others' profiles from updating.
            log.exception("profile refresh failed", extra={"user_id": user_id})
