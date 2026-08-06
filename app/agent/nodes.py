"""The nine graph nodes.

Only two of them spend tokens — `grade` and `generate`. `analyse` is deliberately
deterministic: turning a behaviour profile into a search query is arithmetic over
scores we already computed, and paying a model to restate its own inputs would add
latency, cost and a failure mode for no accuracy gain. The nodes that genuinely need
judgement (is this retrieval any good? how do I say this persuasively?) are the ones
that get a model.

Every node returns a partial state dict. Accumulator fields append; everything else
overwrites.
"""

import logging
import re

from sqlalchemy import select

from app.agent import prompts, ranking
from app.agent.state import AgentState
from app.config import settings
from app.db import session_scope
from app.models import Event, Product, UserProfile
from app.services import guardrails
from app.services import profile as profile_service
from app.services.mesh import MeshError, mesh
from app.services.retrieval import impression_counts, popular, retrieve
from app.services.vectorstore import Filter

log = logging.getLogger(__name__)

_QUOTED = re.compile(r"'([^']+)'")


def _candidate_dict(c) -> dict:
    return {
        "product_id": c.product_id,
        "title": c.title,
        "brand": c.brand,
        "category": c.category,
        "tier": c.tier,
        "price_cents": c.price_cents,
        "rating": c.rating,
        "spec": c.spec,
        "tags": c.tags,
        "description": c.description[:220],
        "retrieved_by": getattr(c, "retrieved_by", "popularity"),
        "fused_score": c.fused_score,
    }


def _account(result) -> dict:
    return {
        "llm_calls": 1,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "cost_usd": result.cost_usd,
    }


# --------------------------------------------------------------------------- 1
def analyze(state: AgentState) -> dict:
    """Behaviour profile -> a search query, metadata filters, and citable evidence."""
    with session_scope() as db:
        prof = db.get(UserProfile, state["user_id"])
        if prof is None:
            return {"node_path": ["analyze"], "strategy": "coldstart", "query": "", "evidence": []}

        summary = profile_service.summarise(prof)
        facts = profile_service.evidence(db, state["user_id"])

        # The query is weighted towards what the user *said* (searches) over what they
        # merely looked at, mirroring the event weights that built the profile.
        parts: list[str] = []
        parts.extend(prof.recent_queries[:3])
        parts.extend(list(prof.interests)[:3])
        parts.extend(list(prof.brand_scores)[:2])
        parts.extend(list(prof.tag_scores)[:5])
        parts.extend(prof.top_terms[:5])
        query = " ".join(dict.fromkeys(p for p in parts if p))[:300]

        # Don't re-recommend something they already enrolled in or saved for later.
        committed = {
            pid
            for (pid,) in db.execute(
                select(Event.product_id).where(
                    Event.user_id == state["user_id"],
                    Event.type.in_(profile_service.COMMITTED_EVENTS),
                    Event.product_id.isnot(None),
                )
            ).all()
        }

        # Or something they explicitly said "not interested" to. Kept as a separate
        # query from `committed` (rather than one combined `.in_()` over both tuples)
        # because the two mean different things even though both end up excluded here.
        dismissed = {
            pid
            for (pid,) in db.execute(
                select(Event.product_id).where(
                    Event.user_id == state["user_id"],
                    Event.type.in_(profile_service.DISMISSED_EVENTS),
                    Event.product_id.isnot(None),
                )
            ).all()
        }

        # Level filter as a progression signal: someone consistently opening advanced
        # material has outgrown the introductory shelf, and vice versa. Always spans two
        # rungs, because the next step up is exactly what a learner wants recommended.
        tiers = None
        if prof.tier_affinity == "advanced":
            tiers = ["intermediate", "advanced"]
        elif prof.tier_affinity == "beginner":
            tiers = ["beginner", "intermediate"]

        # Generous price ceiling — a cap tight enough to bind would silently hide the
        # best match over a few dollars.
        max_price = int(prof.price_affinity_cents * 2.5) if prof.price_affinity_cents else None

        filters = {
            "tiers": tiers,
            "max_price_cents": max_price,
            "exclude_ids": sorted(committed | dismissed),
            # Feeds the collaborative-filtering retrieval leg (retrieval._cf_hits) —
            # separate from exclude_ids, which is what should never be re-shown, not
            # what CF should reason from.
            "committed_ids": sorted(committed),
        }

        # The numeric form of the profile, for ranking.heuristic_score() at grade()
        # time — read here while the session is still open, not queried again later.
        # `dict(...)` copies rather than aliases: `prof` becomes detached the moment
        # this `with` block ends, and its JSON columns would raise on next access.
        profile_features = {
            "interests": dict(prof.interests),
            "tag_scores": dict(prof.tag_scores),
            "brand_scores": dict(prof.brand_scores),
            "price_affinity_cents": prof.price_affinity_cents,
            "tier_affinity": prof.tier_affinity,
        }

    log.debug("analyzed", extra={"query": query[:80], "facts": len(facts)})
    return {
        "node_path": ["analyze"],
        "profile_summary": summary,
        "evidence": facts,
        "profile_features": profile_features,
        "query": query,
        "filters": filters,
    }


# --------------------------------------------------------------------------- 2
def plan(state: AgentState) -> dict:
    """Decide whether there is enough signal to retrieve against at all."""
    if not state.get("query"):
        return {"node_path": ["plan"], "strategy": "coldstart"}
    return {"node_path": ["plan"], "strategy": "agentic"}


def route_after_plan(state: AgentState) -> str:
    return "coldstart" if state["strategy"] == "coldstart" else "retrieve"


# --------------------------------------------------------------------------- 3
def retrieve_node(state: AgentState) -> dict:
    f = state.get("filters") or {}
    flt = Filter(
        tiers=f.get("tiers"),
        max_price_cents=f.get("max_price_cents"),
    )
    exclude = set(f.get("exclude_ids") or [])
    committed = set(f.get("committed_ids") or [])

    with session_scope() as db:
        result = retrieve(
            db,
            query_text=state["query"],
            flt=flt,
            exclude_ids=exclude,
            committed_ids=committed,
            top_n=max(state["top_k"] * 2, 8),
        )
    return {
        "node_path": ["retrieve"],
        "candidates": [_candidate_dict(c) for c in result.candidates],
        "retrieval_stats": result.stats,
    }


# --------------------------------------------------------------------------- 3b
def coldstart(state: AgentState) -> dict:
    """No usable behaviour yet. Recommend what the catalogue itself rates highest
    rather than inventing an interest the user has not expressed."""
    # analyze() may still have computed committed/dismissed exclusions even when it
    # routed here (e.g. a profile with no query signal yet) — popular() already
    # supports exclude_ids, it just wasn't being passed before.
    exclude = set((state.get("filters") or {}).get("exclude_ids") or [])
    with session_scope() as db:
        top = popular(db, state["top_k"], exclude_ids=exclude)
    return {
        "node_path": ["coldstart"],
        "strategy": "coldstart",
        "candidates": [_candidate_dict(c) for c in top],
        "retrieval_stats": {"strategy": "popularity", "returned": len(top)},
    }


# --------------------------------------------------------------------------- 4
def grade(state: AgentState) -> dict:
    """LLM-as-judge on retrieval quality, doubling as a re-ranker — unless the
    deterministic heuristic ranker (app/agent/ranking.py) is already confident, in which
    case the LLM call is skipped outright.

    Grading and re-ranking are one call because they need identical context. Splitting
    them would double the spend to answer two halves of the same question.
    """
    candidates = state.get("candidates") or []
    if not candidates:
        return {"node_path": ["grade"], "grade_score": 0.0, "grade_notes": "no candidates"}

    # The heuristic score reorders `candidates` unconditionally, before anything below
    # touches it — this is what makes it "a lightweight tiebreaker on top of the LLM's
    # ranked_ids ordering" even when the LLM does run: any id the LLM's ranked_ids
    # omits falls back to heuristic order, not arbitrary retrieval order.
    profile_features = state.get("profile_features") or {}
    heuristic_scores = {
        c["product_id"]: ranking.heuristic_score(c, profile_features) for c in candidates
    }
    candidates = sorted(candidates, key=lambda c: -heuristic_scores[c["product_id"]])
    margin = ranking.confidence_margin(sorted(heuristic_scores.values(), reverse=True))

    if not mesh.available:
        # Without a model we cannot judge, so we accept the heuristic ranking as-is
        # rather than blocking the whole recommendation on an unavailable grader.
        return {
            "node_path": ["grade"],
            "grade_score": 0.7,
            "grade_notes": "grading skipped (llm unavailable)",
            "warnings": ["grade_skipped"],
            "candidates": candidates,
        }

    if margin >= settings.ranker_skip_margin:
        # The heuristic's top pick clearly separates from the rest — paying for an LLM
        # call to confirm what's already obvious is exactly the spending this project's
        # trigger policy exists to avoid one level up; this is the same idea one level
        # down, inside a single recommendation.
        return {
            "node_path": ["grade"],
            "grade_score": round(min(1.0, 0.5 + margin), 3),
            "grade_notes": f"heuristic ranker confident (margin {margin:.2f}); LLM grading skipped",
            "warnings": ["grade_skipped_heuristic_confident"],
            "candidates": candidates,
        }

    try:
        data, result = mesh.chat_json(
            [
                {"role": "system", "content": prompts.GRADE_SYSTEM},
                {
                    "role": "user",
                    "content": prompts.grade_user_prompt(
                        state["profile_summary"], state["evidence"], candidates
                    ),
                },
            ],
            model=settings.meshapi_model_fast,
            max_tokens=1200,
        )
    except MeshError as exc:
        log.warning("grade failed", extra={"error": str(exc)[:160]})
        return {
            "node_path": ["grade"],
            "grade_score": 0.7,
            "grade_notes": f"grading failed: {exc}",
            "warnings": ["grade_failed"],
            "candidates": candidates,
        }

    score = float(data.get("score", 0.0) or 0.0)
    ranked = [int(i) for i in (data.get("ranked_ids") or []) if str(i).isdigit()]

    # Re-order by the model's ranking, but only among ids it was actually given. An id
    # it invented here would sail straight past the verifier as a "candidate".
    by_id = {c["product_id"]: c for c in candidates}
    reordered = [by_id[i] for i in ranked if i in by_id]
    reordered += [c for c in candidates if c["product_id"] not in set(ranked)]

    return {
        "node_path": ["grade"],
        "grade_score": max(0.0, min(1.0, score)),
        "grade_notes": str(data.get("notes", ""))[:400],
        "better_query": str(data.get("better_query", ""))[:200],
        "candidates": reordered,
        "prompt_versions": {**state.get("prompt_versions", {}), "grade": prompts.GRADE_PROMPT_VERSION},
        **_account(result),
    }


def route_after_grade(state: AgentState) -> str:
    if (
        state.get("grade_score", 0.0) < settings.retrieval_grade_threshold
        and state.get("refine_count", 0) < settings.retrieval_max_refines
        and state.get("better_query")
    ):
        return "refine"
    return "generate"


# --------------------------------------------------------------------------- 5
def refine(state: AgentState) -> dict:
    """Retry retrieval with the grader's suggested phrasing and looser constraints.

    Widening the filters matters as much as rewording: a weak result set is often a
    filter that excluded the right answer, not a query that failed to describe it.
    """
    filters = dict(state.get("filters") or {})
    filters["tiers"] = None
    if filters.get("max_price_cents"):
        filters["max_price_cents"] = int(filters["max_price_cents"] * 2)

    new_query = state.get("better_query") or state["query"]
    log.info(
        "refining retrieval",
        extra={"attempt": state.get("refine_count", 0) + 1, "query": new_query[:80]},
    )
    return {
        "node_path": ["refine"],
        "query": new_query,
        "filters": filters,
        "refine_count": state.get("refine_count", 0) + 1,
        "better_query": "",
    }


# --------------------------------------------------------------------------- 6
def generate(state: AgentState) -> dict:
    candidates = state.get("candidates") or []
    if not candidates:
        return {"node_path": ["generate"], "error": "nothing to recommend"}

    # Bandit-style exploration: occasionally force a genuinely under-shown candidate
    # into consideration, regardless of what grade()/the heuristic ranker concluded.
    # Applied here rather than as its own graph node — every upstream path (LLM-graded,
    # heuristic-skipped, coldstart) already funnels through generate(), so this is the
    # one place that reaches all of them without adding a 10th node.
    with session_scope() as db:
        exposure = impression_counts(db, [c["product_id"] for c in candidates])
    candidates = ranking.apply_exploration_slot(
        candidates, exposure, settings.explore_epsilon, state["top_k"]
    )

    if not mesh.available:
        return {"node_path": ["generate"], **_deterministic_copy(state, candidates)}

    system = prompts.GENERATE_SYSTEM
    user = prompts.generate_user_prompt(
        state["profile_summary"], state["evidence"], candidates, state["top_k"]
    )
    if state.get("repair_notes"):
        user += prompts.REPAIR_SUFFIX.format(violations=state["repair_notes"])

    try:
        data, result = mesh.chat_json(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            model=settings.meshapi_model,
            max_tokens=settings.llm_max_output_tokens,
        )
    except MeshError as exc:
        log.warning("generation failed, using deterministic copy", extra={"error": str(exc)[:160]})
        return {
            "node_path": ["generate"],
            "warnings": [f"generate_failed:{type(exc).__name__}"],
            **_deterministic_copy(state, candidates),
        }

    picks = []
    for p in data.get("picks", []) or []:
        try:
            picks.append(
                {"product_id": int(p["product_id"]), "reason": str(p.get("reason", ""))[:400]}
            )
        except (KeyError, TypeError, ValueError):
            continue  # verify() reports the shortfall; one malformed pick is not fatal

    return {
        "node_path": ["generate"],
        "headline": str(data.get("headline", ""))[:180],
        "narrative": str(data.get("narrative", ""))[:2000],
        "cta": str(data.get("cta", ""))[:100],
        "picks": picks,
        "model": result.model,
        "strategy": state.get("strategy", "agentic"),
        "prompt_versions": {
            **state.get("prompt_versions", {}), "generate": prompts.GENERATE_PROMPT_VERSION
        },
        **_account(result),
    }


# --------------------------------------------------------------------------- 7
def verify(state: AgentState) -> dict:
    """Groundedness gate. Nothing reaches a user without passing here.

    Two independent failure modes, both real:
      * the model recommends a product id that does not exist, or that it was never
        offered — the classic RAG hallucination;
      * the copy is grounded but dishonest — invented discount, manufactured deadline,
        promised outcome. Guardrails catch that.
    """
    candidate_ids = {c["product_id"] for c in (state.get("candidates") or [])}
    picks = state.get("picks") or []

    kept, dropped = [], []
    for p in picks:
        if p["product_id"] in candidate_ids:
            kept.append(p)
        else:
            dropped.append(p["product_id"])

    problems: list[str] = []
    if dropped:
        problems.append(
            f"recommended product ids {dropped} were not in the candidate list"
        )

    # Confirm the survivors still exist and are published: the index can lag SQL.
    rows: dict[int, Product] = {}
    if kept:
        with session_scope() as db:
            rows = {
                p.id: p
                for p in db.scalars(
                    select(Product).where(Product.id.in_([k["product_id"] for k in kept]))
                )
            }
    live = {pid for pid, p in rows.items() if p.is_published}
    allowed_prices = {p.price_cents for p in rows.values()}

    unpublished = [p["product_id"] for p in kept if p["product_id"] not in live]
    if unpublished:
        problems.append(f"product ids {unpublished} are missing or unpublished")
        kept = [p for p in kept if p["product_id"] in live]

    copy_blob = " ".join(
        [state.get("headline", ""), state.get("narrative", ""), state.get("cta", "")]
        + [p["reason"] for p in kept]
    )
    report = guardrails.validate(copy_blob, allowed_prices_cents=allowed_prices or None)
    problems.extend(report.violations)

    if not kept:
        problems.append("no valid products left after verification")

    if not problems:
        confidence = round(
            0.5 * min(1.0, len(kept) / max(1, state["top_k"]))
            + 0.5 * max(state.get("grade_score", 0.0), 0.5),
            3,
        )
        return {
            "node_path": ["verify"],
            "picks": kept,
            "confidence": confidence,
            "repair_notes": "",
        }

    log.warning("verification failed", extra={"problems": problems[:4], "repaired": state.get("repaired")})
    return {
        "node_path": ["verify"],
        "picks": kept,
        "warnings": [f"verify:{p}" for p in problems],
        "repair_notes": "; ".join(problems)[:600],
        "repaired": True,
    }


def route_after_verify(state: AgentState) -> str:
    """One repair attempt, then take the best deterministic outcome available.

    An unbounded repair loop against a model that keeps making the same mistake is a
    way to burn a budget, not a way to get correct output.
    """
    if state.get("repair_notes") and state.get("node_path", []).count("generate") < 2:
        return "generate"
    return "finalize"


# --------------------------------------------------------------------------- 8
def finalize(state: AgentState) -> dict:
    """Last line of defence: if verification still fails, fall back to honest copy
    built from the catalogue rather than shipping something we know is wrong."""
    if not state.get("repair_notes"):
        return {"node_path": ["finalize"]}

    candidates = state.get("candidates") or []
    kept = state.get("picks") or []
    if not kept and candidates:
        fallback = _deterministic_copy(state, candidates)
        return {
            "node_path": ["finalize"],
            "warnings": ["fell_back_to_deterministic"],
            **fallback,
        }

    # Courses survived but the prose did not. Strip the offending sentences.
    cleaned = guardrails.scrub(state.get("narrative", ""))
    return {
        "node_path": ["finalize"],
        "narrative": cleaned or _plain_narrative(state, kept),
        "headline": state.get("headline") or "Picked for you",
        "warnings": ["scrubbed_narrative"],
        "confidence": 0.4,
    }


# --------------------------------------------------------------------------- fallback
def _plain_narrative(state: AgentState, picks: list[dict]) -> str:
    facts = state.get("evidence") or []
    lead = f"Based on what you've been studying — {facts[0]}" if facts else "Based on your recent activity"
    tail = "this course follows" if len(picks) == 1 else f"these {len(picks)} courses follow"
    return f"{lead} — {tail} on from that."


def _deterministic_copy(state: AgentState, candidates: list[dict]) -> dict:
    """Template copy used when the model is unavailable, over budget, or untrustworthy.

    Honest and specific rather than impressive: it cites the same evidence the model
    would have been given, and every claim is read straight off the catalogue.
    """
    top = candidates[: state["top_k"]]
    facts = state.get("evidence") or []
    interests = state.get("profile_summary", "")

    # profile.evidence() phrases every fact as a bare predicate ("searched for 'x'
    # 3 times", "spent 4m reading 'y'"), so "You " + fact is always a sentence.
    subject = next((m.group(1) for f in facts if (m := _QUOTED.search(f))), "")
    # Track comes off the retrieved set, not the filters: `analyze` deliberately does not
    # filter by track — soft matching beats a hard fence for discovery — so reading it
    # from there always produced an empty string.
    track = (top[0]["category"] if top else "").replace("-", " ")
    headline = (
        f"More like {subject}"[:70] if subject
        else (f"Next in {track}"[:70] if track else "Picked for you")
    )

    if facts:
        lead = f"You {facts[0]}"
        if len(facts) > 1:
            lead += f", and {facts[1]}"
    else:
        lead = "Based on what you've been studying"

    titles = ", ".join(c["title"] for c in top[:2])
    narrative = (
        f"{lead}. These are the closest matches in the catalogue — starting with {titles}"
        f"{', plus more below' if len(top) > 2 else ''}."
    )

    picks = [
        {
            "product_id": c["product_id"],
            "reason": f"{c['brand']}, {c['category'].replace('-', ' ')}, "
            f"{c['tier']} level, rated {c['rating']:.1f}"
            + (f" — {c['spec'].split(' · ')[0]}." if c.get("spec") else "."),
        }
        for c in top
    ]
    return {
        "headline": headline,
        "narrative": narrative,
        "cta": "Take a look",
        "picks": picks,
        "strategy": "fallback" if state.get("strategy") != "coldstart" else "coldstart",
        "model": "deterministic",
        "confidence": 0.3,
        "profile_summary": interests,
    }
