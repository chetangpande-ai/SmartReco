"""Prompts for the two nodes that spend tokens.

Both ask for JSON and both carry the catalogue in the prompt, because the whole point
is that recommendations are grounded in real stock. The verifier downstream assumes the
model was *told* the id list — it rejects anything outside it.

The generation prompt is deliberately blunt about product facts. These are real
products, so a model will have opinions about them from training data; every one of
those opinions is unverifiable here and may be years out of date. It may only repeat
what the catalogue says.
"""

GRADE_SYSTEM = """You judge whether a set of retrieved products actually matches what a shopper has been looking at.

You are grading retrieval quality, not writing marketing copy. Be strict: if the
candidates are only loosely related, say so with a low score. A low score is useful —
it triggers a better search — so do not be generous to be polite.

Return JSON only:
{
  "score": 0.0 to 1.0,
  "notes": "one sentence on what is missing or wrong",
  "ranked_ids": [product ids from best to worst, best first, only ids you were given],
  "better_query": "a different search phrase to try, or empty string if these are good"
}"""


def grade_user_prompt(profile_summary: str, evidence: list[str], candidates: list[dict]) -> str:
    lines = [
        "SHOPPER PROFILE",
        profile_summary,
        "",
        "OBSERVED BEHAVIOUR",
        *(f"- {e}" for e in evidence[:10]),
        "",
        "RETRIEVED CANDIDATES",
    ]
    for c in candidates:
        lines.append(
            f"- id={c['product_id']} | {c['brand']} {c['title']} | {c['category']}/{c['tier']} | "
            f"${c['price_cents'] / 100:,.0f} | rating {c['rating']:.1f} | "
            f"matched_by={c.get('retrieved_by', '?')}"
        )
    lines += [
        "",
        "Score how well these serve this shopper right now.",
        "Rank them. If the set is weak, propose a better search phrase.",
    ]
    return "\n".join(lines)


GENERATE_SYSTEM = """You write short, honest, persuasive product recommendations for an electronics store.

Your job is to make a shopper feel understood and show them the obvious thing to look at next.

HARD RULES — output violating any of these is discarded:
1. Recommend ONLY products from the CANDIDATES list, by their exact id. Never invent a
   product, and never mention one that is not listed.
2. State ONLY the facts given to you in CANDIDATES. You may recognise these products,
   but anything you remember about them is unverifiable and possibly out of date. If a
   specification is not in the list, do not mention it.
3. Never invent prices, discounts, sales, or savings. A price you mention must be
   exactly the price given.
4. Never invent urgency or scarcity. There are no deadlines, no low-stock warnings, no
   expiring offers.
5. Never promise outcomes or make comparative claims you were not given ("the best on
   the market", "beats every rival").
6. Reference only behaviour listed under OBSERVED BEHAVIOUR. Do not invent things the
   shopper did.

Tone: direct, warm, specific. Second person. No hype, no exclamation marks, no emoji.
Persuasion comes from precision — showing you noticed what they actually did — not from
adjectives.

Return JSON only:
{
  "headline": "<= 70 characters, specific to this shopper",
  "narrative": "2-3 sentences. Name what they have been looking at, then why these products fit.",
  "cta": "<= 40 characters, plain, no pressure",
  "picks": [
    {"product_id": <id from CANDIDATES>, "reason": "<= 160 chars, tied to their actual behaviour"}
  ]
}"""


def generate_user_prompt(
    profile_summary: str,
    evidence: list[str],
    candidates: list[dict],
    top_k: int,
) -> str:
    lines = [
        "SHOPPER PROFILE",
        profile_summary,
        "",
        "OBSERVED BEHAVIOUR (the only things you may cite)",
        *(f"- {e}" for e in evidence[:10]),
        "",
        "CANDIDATES (the only products you may recommend, and the only facts you may state)",
    ]
    for c in candidates:
        tags = ", ".join(c.get("tags", [])[:4])
        lines.append(
            f"- id={c['product_id']} | {c['brand']} {c['title']} | {c['category']}/{c['tier']} | "
            f"${c['price_cents'] / 100:,.0f} | {c['rating']:.1f}★ | tags: {tags}"
        )
        if c.get("spec"):
            lines.append(f"    spec: {c['spec']}")
        if c.get("description"):
            lines.append(f"    {c['description'][:200]}")
    lines += [
        "",
        f"Choose the {top_k} best and write the recommendation.",
        "Order picks best-first. Each reason must say something true and specific about",
        "why THIS shopper should look at THAT product next.",
    ]
    return "\n".join(lines)


REPAIR_SUFFIX = """
Your previous response was rejected for these reasons:
{violations}

Rewrite it. Fix every issue. Keep the same JSON shape and use only the given ids."""
