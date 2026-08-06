"""The state passed between graph nodes.

Accumulator fields use `operator.add` reducers so each node appends its own
contribution rather than overwriting — which is what makes `node_path` a real
execution trace and the token counters a real per-run total, including across
refine loops that visit the same node twice.
"""

from operator import add
from typing import Annotated, TypedDict


class AgentState(TypedDict, total=False):
    # ---- input
    user_id: int
    request_id: str
    trigger: str
    top_k: int

    # ---- analyse
    profile_summary: str
    query: str
    filters: dict
    evidence: list[str]  # the concrete behaviours the copy is allowed to cite
    strategy: str  # agentic | coldstart | fallback
    # Structured UserProfile fields (interests, tag_scores, brand_scores,
    # price_affinity_cents, tier_affinity), read once in analyze() while its DB session
    # is still open. profile_summary/evidence above are the string-rendered forms the
    # LLM prompts use; this is the numeric form app/agent/ranking.py's heuristic scorer
    # needs, threaded through state instead of a second query at grade() time.
    profile_features: dict

    # ---- retrieve / grade / refine
    candidates: list[dict]
    retrieval_stats: dict
    grade_score: float
    grade_notes: str
    better_query: str
    refine_count: int

    # ---- generate / verify
    headline: str
    narrative: str
    cta: str
    picks: list[dict]
    confidence: float
    repaired: bool
    repair_notes: str

    # ---- bookkeeping (accumulated)
    node_path: Annotated[list[str], add]
    warnings: Annotated[list[str], add]
    llm_calls: Annotated[int, add]
    prompt_tokens: Annotated[int, add]
    completion_tokens: Annotated[int, add]
    cost_usd: Annotated[float, add]

    # Which prompt version each token-spending node used, e.g. {"grade": "v1", "generate":
    # "v1"}. Not an operator.add accumulator — dicts don't support `+` — so grade/generate
    # must merge against the incoming state themselves (dict.update, not overwrite) or a
    # second pass through the refine/repair loop drops the other node's entry.
    prompt_versions: dict[str, str]

    # ---- terminal
    model: str
    error: str
    recommendation_id: int


def new_state(user_id: int, trigger: str, request_id: str, top_k: int) -> AgentState:
    return AgentState(
        user_id=user_id,
        request_id=request_id,
        trigger=trigger,
        top_k=top_k,
        strategy="agentic",
        refine_count=0,
        grade_score=0.0,
        repaired=False,
        node_path=[],
        warnings=[],
        llm_calls=0,
        prompt_tokens=0,
        completion_tokens=0,
        cost_usd=0.0,
        candidates=[],
        picks=[],
        evidence=[],
        filters={},
        error="",
        model="",
        prompt_versions={},
        profile_features={},
    )
