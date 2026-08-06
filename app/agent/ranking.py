"""Deterministic candidate scoring — a cheap alternative to an LLM call for ranking, and
the mechanism behind the exploration slot. Split out of nodes.py for the same reason
prompts.py is: a distinct concern from graph wiring.

The scorer's weights are hand-set, not fit. There is no labelled click/conversion data
at this catalogue's scale to train them from, and pretending otherwise would be dishonest
about what this is. What it buys is real: computed entirely from data already in memory
at grading time (no new DB query), it lets `grade()` skip an LLM call outright when its
own ranking is confident — the actual value being chased, cheaper and faster ranking,
without overclaiming ML sophistication the data can't back.
"""

import random

# Category/tag overlap weighted heaviest: it's the most direct signal of "this is what
# the learner has actually been studying." Price and rating are tie-breakers, not
# drivers — a course priced slightly outside someone's usual range can still be exactly
# right, which is why retrieval's own price filter is already generous (2.5x affinity).
_WEIGHT_CATEGORY = 0.35
_WEIGHT_TAGS = 0.30
_WEIGHT_TIER = 0.15
_WEIGHT_BRAND = 0.10
_WEIGHT_PRICE = 0.05
_WEIGHT_RATING = 0.05


def heuristic_score(candidate: dict, profile_features: dict) -> float:
    """Score one candidate against a learner's structured profile fields (interests,
    tag_scores, brand_scores, price_affinity_cents, tier_affinity — the raw UserProfile
    data, not the string-rendered summary/evidence the LLM prompts use). Roughly [0, 1];
    the exact ceiling doesn't matter since scores here are only ever compared to each
    other, never shown to a user or a model.
    """
    interests = profile_features.get("interests") or {}
    tag_scores = profile_features.get("tag_scores") or {}
    brand_scores = profile_features.get("brand_scores") or {}
    price_affinity = profile_features.get("price_affinity_cents") or 0
    tier_affinity = profile_features.get("tier_affinity") or ""

    peak_interest = max(interests.values(), default=0.0) or 1.0
    category_score = interests.get(candidate.get("category", ""), 0.0) / peak_interest

    tags = candidate.get("tags") or []
    peak_tag = max(tag_scores.values(), default=0.0) or 1.0
    tag_score = sum(tag_scores.get(t, 0.0) for t in tags) / (len(tags) * peak_tag) if tags else 0.0

    peak_brand = max(brand_scores.values(), default=0.0) or 1.0
    brand_score = brand_scores.get(candidate.get("brand", ""), 0.0) / peak_brand

    if tier_affinity:
        tier_score = 1.0 if candidate.get("tier") == tier_affinity else 0.0
    else:
        tier_score = 0.5  # no signal yet — neutral, not a penalty

    price_cents = candidate.get("price_cents", 0)
    if price_affinity:
        price_score = max(0.0, 1.0 - abs(price_cents - price_affinity) / max(price_affinity, 1))
    else:
        price_score = 0.5

    rating_score = (candidate.get("rating") or 0.0) / 5.0

    return (
        _WEIGHT_CATEGORY * category_score
        + _WEIGHT_TAGS * tag_score
        + _WEIGHT_TIER * tier_score
        + _WEIGHT_BRAND * brand_score
        + _WEIGHT_PRICE * price_score
        + _WEIGHT_RATING * rating_score
    )


def confidence_margin(scores: list[float]) -> float:
    """Gap between the best and second-best score — a proxy for "is the top pick
    clearly right, or is this close enough that the judgment call an LLM provides is
    worth paying for?" A list of 0 or 1 scores has nothing to be ambiguous about, so it's
    maximally confident by definition."""
    if len(scores) < 2:
        return 1.0
    ranked = sorted(scores, reverse=True)
    return ranked[0] - ranked[1]


def apply_exploration_slot(
    candidates: list[dict], exposure: dict[int, int], epsilon: float, top_k: int
) -> list[dict]:
    """With probability `epsilon`, swap the lowest-ranked kept slot for a candidate
    outside the top-k that's been shown the least — so a genuinely under-exposed course
    occasionally surfaces regardless of what the ranker/LLM thinks, instead of the same
    best-guess set compounding its own popularity forever. `epsilon<=0` and `epsilon>=1`
    are exact (no float-boundary flakiness), which is what lets tests drive this
    deterministically instead of mocking `random`.
    """
    if epsilon <= 0 or len(candidates) <= top_k or random.random() >= epsilon:
        return candidates

    kept, rest = candidates[:top_k], candidates[top_k:]
    if not rest:
        return candidates
    explorer = min(rest, key=lambda c: exposure.get(c["product_id"], 0))
    return kept[:-1] + [explorer]
