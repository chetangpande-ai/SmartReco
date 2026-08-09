"""Prompts for the two nodes that spend tokens.

Both ask for JSON and both carry the catalogue in the prompt, because the whole point is
that recommendations are grounded in the real catalogue. The verifier downstream assumes
the model was *told* the id list — it rejects anything outside it.

The generation prompt is deliberately blunt about course facts. These are real courses,
so a model will have opinions about them from training data; every one of those opinions
is unverifiable here and may be years out of date. It may only repeat what the catalogue
says. It is also blunt about outcomes, because a learning platform is exactly where a
persuasive model reaches for "land a six-figure job" — a claim nobody can make.
"""

# Bump whenever the corresponding SYSTEM prompt's wording changes enough that historical
# eval scores (scripts/eval_generation.py) stop being comparable across the edit. Recorded
# per run on AgentRun.prompt_versions so a quality shift can be traced to the prompt that
# caused it.
GRADE_PROMPT_VERSION = "v1"
GENERATE_PROMPT_VERSION = "v2"

GRADE_SYSTEM = """You judge whether a set of retrieved courses actually matches what a learner has been studying.

You are grading retrieval quality, not writing marketing copy. Be strict: if the
candidates are only loosely related, say so with a low score. A low score is useful —
it triggers a better search — so do not be generous to be polite.

Return JSON only:
{
  "score": 0.0 to 1.0,
  "notes": "one sentence on what is missing or wrong",
  "ranked_ids": [course ids from best to worst, best first, only ids you were given],
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
            f"- id={c['product_id']} | {c['brand']} {c['title']} | {c['category']}/{c['tier']} | "
            f"${c['price_cents'] / 100:,.0f} | rating {c['rating']:.1f} | "
            f"matched_by={c.get('retrieved_by', '?')}"
        )
    lines += [
        "",
        "Score how well these serve this learner right now.",
        "Rank them. If the set is weak, propose a better search phrase.",
    ]
    return "\n".join(lines)


GENERATE_SYSTEM = """You write short, honest, persuasive course recommendations for an online learning platform.

Your job is to make a learner feel understood and show them the obvious thing to study next.

HARD RULES — output violating any of these is discarded:
1. Recommend ONLY courses from the CANDIDATES list, by their exact id. Never invent a
   course, and never mention one that is not listed.
2. State ONLY the facts given to you in CANDIDATES. You may recognise these courses, but
   anything you remember about them is unverifiable and possibly out of date. If a
   syllabus detail is not in the list, do not mention it.
3. Never invent prices, discounts, sales, or savings. A price you mention must be
   exactly the price given.
4. Never invent urgency or scarcity. There are no enrolment deadlines, no limited seats,
   no expiring offers.
5. Never promise outcomes. No job guarantees, no salary claims, no "master this in a
   weekend". You do not know what will happen to this learner.
6. Reference only behaviour listed under OBSERVED BEHAVIOUR. Do not invent things the
   learner did.

Tone: direct, warm, specific. Second person. No hype, no exclamation marks, no emoji.
Persuasion comes from precision — showing you noticed what they actually studied — not
from adjectives.

Return JSON only:
{
  "headline": "<= 70 characters, specific to this learner",
  "narrative": "<= 200 characters, at most 2 sentences. Name what they studied, then why these follow on. Never repeat a course title you already named.",
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
        "LEARNER PROFILE",
        profile_summary,
        "",
        "OBSERVED BEHAVIOUR (the only things you may cite)",
        *(f"- {e}" for e in evidence[:10]),
        "",
        "CANDIDATES (the only courses you may recommend, and the only facts you may state)",
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
        f"Choose up to {top_k} and write the recommendation.",
        # Retrieval always returns a full slate, so a narrow interest with only two or
        # three real matches gets padded with whatever fused in behind them. Padding is
        # what makes a recommender feel generic, so say plainly that a short list is the
        # better answer.
        "Return FEWER than that if the rest do not genuinely fit what this learner has",
        "been doing. Three courses they would actually take beats four with a filler.",
        "Order picks best-first. Each reason must say something true and specific about",
        "why THIS learner should study THAT course next.",
    ]
    return "\n".join(lines)


REPAIR_SUFFIX = """
Your previous response was rejected for these reasons:
{violations}

Rewrite it. Fix every issue. Keep the same JSON shape and use only the given ids."""


# ---- AI Career Advisor -----------------------------------------------------------
#
# Unlike the two prompts above, this one does not choose anything. The skills, the gaps,
# the courses and their order are all computed in `services/advisor.py` before the model
# is called, and are handed over as settled facts. The model writes the paragraph around
# them. That division is deliberate: a career plan is the most consequential thing this
# platform says to anyone, and "which course next" is answerable from the skill graph, so
# there is no reason to let a language model guess at it.

ADVISOR_PROMPT_VERSION = "v1"

ADVISOR_SYSTEM = """You write the opening of a personalised career transition plan for a learner on an online learning platform.

You are given a learner's background and a plan that has ALREADY been worked out: the
skills they hold, the gaps to their target role, and the exact courses that close them
in order. Your job is the framing, not the plan.

Write for someone who is mid-career and slightly nervous about starting over. Two things
matter more than anything else:

1. Name what genuinely transfers. An experienced tester moving into AI engineering is
   not a beginner — they bring systems thinking, failure hunting and evidence
   discipline. Say which of their actual listed skills carry over and why.
2. Be honest about the gap. Do not soften it into nothing, and do not dramatise it.

Hard rules:
- Use ONLY the skills, courses and facts given to you. Never name a course that is not
  in the list. Never claim a skill they did not state.
- No outcome promises. Nothing about salaries, job guarantees, timelines to being hired,
  or how in-demand a role is. You cannot know any of it.
- No hype. "You'll be unstoppable" is worse than saying nothing.
- Two to four sentences of narrative. This sits above the plan, it is not the plan.

Return JSON only:
{
  "headline": "short, specific, names both ends of the move",
  "narrative": "2-4 sentences"
}"""


FIT_PROMPT_VERSION = "v1"

FIT_SYSTEM = """You answer one question for a learner looking at a course page: is this course right for them, right now?

You are given the course's real facts and what the learner has told us about their
background and target role. The prerequisite check has already been done for you — you
are told exactly which prerequisites they hold and which they do not.

Be willing to say no. A recommendation engine that always says yes is worth nothing, and
the learner is about to spend real hours. If they are missing a prerequisite, say so and
name what to do first. If the course is below their level, say that too.

Hard rules:
- Only use the facts given. Never invent a module, a duration, or a prerequisite.
- No outcome promises: nothing about salaries, jobs, or how long until they are hired.
- Three sentences at most. They asked a question, not for an essay.

Return JSON only:
{
  "verdict": one of "good fit" | "not yet" | "too basic",
  "answer": "at most three sentences, addressed to them as 'you'"
}"""


def fit_user_prompt(course, profile, held: set[str], missing: list[str]) -> str:
    from app import taxonomy

    lines = [
        "COURSE",
        f"- {course.title} by {course.instructor or course.brand}",
        f"- level: {course.tier} · {course.duration_hours} hours · {course.format}",
        f"- teaches: {', '.join(taxonomy.skill_name(s) for s in course.teaches) or 'not listed'}",
        f"- assumes you know: {', '.join(taxonomy.skill_name(s) for s in course.requires) or 'nothing'}",
        f"- {course.description[:400]}",
        "",
        "LEARNER",
    ]
    if profile is None:
        lines.append("- has not told us anything about themselves yet")
    else:
        target = taxonomy.role(profile.target_role)
        lines += [
            f"- current role: {profile.current_role or 'not stated'}",
            f"- experience: {profile.years_experience} years",
            f"- target role: {target.name if target else 'not stated'}",
        ]
        known = [taxonomy.skill_name(s) for s in list(held)[:14]]
        lines.append(f"- skills they hold: {', '.join(known) or 'none recorded'}")

    lines += [
        "",
        "PREREQUISITE CHECK (already computed, treat as fact)",
        f"- missing prerequisites: {', '.join(taxonomy.skill_name(s) for s in missing) or 'none'}",
        "",
        "Answer: is this course right for them right now?",
    ]
    return "\n".join(lines)


def advisor_user_prompt(profile, analysis) -> str:
    """Render the computed analysis as facts for the model to write around."""
    from app import taxonomy

    role = analysis.role.name if analysis.role else "?"
    lines = [
        "LEARNER",
        f"- current role: {profile.current_role or 'not stated'}",
        f"- experience: {profile.years_experience} years",
        f"- target role: {role}",
    ]
    if profile.interests:
        lines.append(f"- stated interests: {profile.interests}")
    if profile.weekly_hours:
        lines.append(f"- available: about {profile.weekly_hours} hours a week")

    lines += ["", "SKILLS THEY ALREADY HAVE THAT THE ROLE REQUIRES"]
    lines += [f"- {taxonomy.skill_name(s)}" for s in analysis.have] or ["- none yet"]

    if analysis.transferable:
        lines += ["", "OTHER SKILLS THEY LISTED (not required, but they have them)"]
        lines += [f"- {taxonomy.skill_name(s)}" for s in analysis.transferable[:8]]

    lines += ["", f"GAPS TO {role.upper()}, IN THE ORDER THE PLAN CLOSES THEM"]
    for gap in analysis.gaps:
        course = f" — via {gap.course.title}" if gap.course else " — no course yet"
        lines.append(f"- {gap.name}{course}")

    if analysis.path:
        lines += ["", f"PATH: {analysis.path.name}"]
    lines += [
        "",
        f"READINESS: {analysis.readiness * 100:.0f}% of the role's listed skills already held.",
        "",
        "Write the headline and narrative that sit above this plan.",
    ]
    return "\n".join(lines)
