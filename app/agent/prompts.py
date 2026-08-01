"""Prompts for the two nodes that spend tokens.

Both ask for JSON and both keep the catalogue in the prompt, because the whole point is
that recommendations are grounded in real products. The verifier downstream assumes the
model was *told* the id list — it rejects anything outside it.
"""

GRADE_SYSTEM = """You judge whether a set of retrieved courses actually matches a learner's demonstrated interests.

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
        "LEARNER PROFILE",
        profile_summary,
        "",
        "OBSERVED BEHAVIOUR",
        *(f"- {e}" for e in evidence[:10]),
        "",
        "RETRIEVED CANDIDATES",
    ]
    for c in candidates:
        lines.append(
            f"- id={c['product_id']} | {c['title']} | {c['category']}/{c['level']} | "
            f"${c['price_cents'] / 100:.0f} | rating {c['rating']:.1f} | "
            f"matched_by={c.get('retrieved_by', '?')}"
        )
    lines += [
        "",
        "Score how well these candidates serve this learner's next step.",
        "Rank them. If the set is weak, propose a better search phrase.",
    ]
    return "\n".join(lines)


GENERATE_SYSTEM = """You write short, honest, persuasive recommendations for an online course platform.

Your job is to make a learner feel understood and show them the obvious next step.

HARD RULES — output violating any of these is discarded:
1. Recommend ONLY courses from the CANDIDATES list, by their exact id. Never invent a
   course, and never mention one that is not listed.
2. Never invent prices, discounts, sales, or savings. If you mention a price it must be
   exactly the price given.
3. Never invent urgency or scarcity. There are no deadlines, no limited seats, no
   closing enrolments.
4. Never promise outcomes: no guaranteed jobs, salaries, results, or timeframes.
5. Reference only behaviour listed under OBSERVED BEHAVIOUR. Do not invent things the
   learner did.

Tone: direct, warm, specific. Second person. No hype, no exclamation marks, no emoji.
Persuasion comes from precision — showing you noticed what they actually did — not from
adjectives.

Return JSON only:
{
  "headline": "<= 70 characters, specific to this learner",
  "narrative": "2-3 sentences. Name what they have been exploring, then why these courses are the right next step.",
  "cta": "<= 40 characters, plain, no pressure",
  "picks": [
    {"product_id": <id from CANDIDATES>, "reason": "<= 160 chars, tied to this learner's actual behaviour"}
  ]
}"""


def generate_user_prompt(
    profile_summary: str,
    evidence: list[str],
    candidates: list[dict],
    top_k: int,
) -> str:
    lines = [
        "LEARNER PROFILE",
        profile_summary,
        "",
        "OBSERVED BEHAVIOUR (the only things you may cite)",
        *(f"- {e}" for e in evidence[:10]),
        "",
        "CANDIDATES (the only courses you may recommend)",
    ]
    for c in candidates:
        tags = ", ".join(c.get("tags", [])[:4])
        lines.append(
            f"- id={c['product_id']} | {c['title']} | {c['category']}/{c['level']} | "
            f"${c['price_cents'] / 100:.0f} | {c['rating']:.1f}★ | topics: {tags}"
        )
        if c.get("description"):
            lines.append(f"    {c['description'][:180]}")
    lines += [
        "",
        f"Choose the {top_k} best and write the recommendation.",
        "Order picks best-first. Each reason must say something true and specific about",
        "why THIS learner should take THAT course next.",
    ]
    return "\n".join(lines)


REPAIR_SUFFIX = """
Your previous response was rejected for these reasons:
{violations}

Rewrite it. Fix every issue. Keep the same JSON shape and use only the given ids."""
