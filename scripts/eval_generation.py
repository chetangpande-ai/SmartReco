"""Generation-quality evaluation: DeepEval RAG metrics + a custom GEval rubric.

Where scripts/eval_retrieval.py measures the retrieval stack, this measures what the
`generate` node actually writes — the part that carries hallucination risk. Runs the
real `nodes.generate()` against real catalogue candidates (no mocking of the prompt or
the model), then scores the output three ways:

  * FaithfulnessMetric   — does the narrative only make claims the candidates/evidence
                            support? DeepEval's RAG-faithfulness equivalent of what
                            RAGAs would call `Faithfulness`.
  * AnswerRelevancyMetric — does the output actually address this learner's profile?
  * GROUNDED_PERSUASION   — a custom GEval rubric. Its criteria are written to mirror
                            prompts.GENERATE_SYSTEM's HARD RULES line for line, so when
                            that prompt changes, this rubric is what tells you whether
                            the change helped or hurt.

Needs a real MESHAPI_API_KEY and LLM_ENABLED=true — both the thing being measured and
the judge doing the measuring are real model calls. Not run in CI, same as `make eval`
against a live index.

    uv run python scripts/eval_generation.py
    uv run python scripts/eval_generation.py --min-score 0.6
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.agent import nodes  # noqa: E402
from app.agent.state import new_state  # noqa: E402
from app.config import settings  # noqa: E402
from app.db import init_db, session_scope  # noqa: E402
from app.logging_conf import configure_logging  # noqa: E402
from app.models import Product  # noqa: E402
from app.services.mesh import mesh  # noqa: E402

# Each scenario is a plausible learner profile, phrased exactly as profile.summarise()/
# evidence() would phrase one. Candidates are pulled from the real seeded catalogue by
# category at run time, not hardcoded — a scenario failing to find candidates is itself
# a signal the seed data or category name has drifted.
SCENARIOS = [
    {
        "name": "ml_progression",
        "profile_summary": "studying: ai-ml (3.2) | searched: deep learning | level: intermediate",
        "evidence": [
            "enrolled in Machine Learning Specialization",
            "searched for 'deep learning' 2 times",
            "spent 6m 40s reading 'Deep Learning Specialization'",
            "favour intermediate material",
        ],
        "category": "ai-ml",
    },
    {
        "name": "web_fundamentals",
        "profile_summary": "studying: web-dev (2.4) | searched: react hooks | level: beginner",
        "evidence": [
            "searched for 'react hooks' 3 times",
            "looked at Complete Intro to React 2 times",
            "favour beginner material",
        ],
        "category": "web-dev",
    },
    {
        "name": "data_engineering",
        "profile_summary": "studying: data-eng (1.8) | providers looked at: dbt Labs",
        "evidence": [
            "enrolled in Analytics Engineering with dbt",
            "keep coming back to dbt Labs",
        ],
        "category": "data",
    },
]

GROUNDED_PERSUASION_CRITERIA = """
Score how well the ACTUAL OUTPUT (a course recommendation) follows these rules, given
the INPUT (learner profile + observed behaviour) and RETRIEVAL CONTEXT (the only courses
and facts it was allowed to use):

1. It recommends only courses that appear in the retrieval context, by the ids given.
2. It states only facts present in the retrieval context — no invented syllabus detail,
   no price other than the one given, no discount, no urgency or scarcity language.
3. It never promises an outcome (no job guarantees, no salary claims, no timeframes like
   "master this overnight").
4. Every claim about the learner's own behaviour is grounded in the OBSERVED BEHAVIOUR
   given in the input, not invented.
5. The tone is direct and specific rather than generic hype — it should read like it was
   written for this one learner, not a template.

Score 1.0 only if all five hold. Deduct heavily for any invented fact, price, or promise.
"""


def _candidates_for(db, category: str, limit: int = 6) -> list[dict]:
    rows = db.execute(
        select(Product)
        .where(Product.category == category, Product.is_published.is_(True))
        .order_by(Product.rating.desc())
        .limit(limit)
    ).scalars().all()
    return [
        {
            "product_id": p.id,
            "title": p.title,
            "brand": p.brand,
            "category": p.category,
            "tier": p.tier,
            "price_cents": p.price_cents,
            "rating": p.rating,
            "spec": p.spec,
            "tags": p.tags,
            "description": p.description[:220],
            "retrieved_by": "eval_fixture",
            "fused_score": 1.0,
        }
        for p in rows
    ]


def _run_generate(scenario: dict, candidates: list[dict]) -> dict:
    state = new_state(user_id=0, trigger="eval", request_id="eval", top_k=3)
    state["profile_summary"] = scenario["profile_summary"]
    state["evidence"] = scenario["evidence"]
    state["candidates"] = candidates
    return nodes.generate(state)


def _context_lines(scenario: dict, candidates: list[dict]) -> list[str]:
    lines = list(scenario["evidence"])
    for c in candidates:
        lines.append(f"id={c['product_id']} {c['brand']} {c['title']} ${c['price_cents']/100:.0f}")
    return lines


def measure() -> dict:
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
    from deepeval.test_case import LLMTestCase, SingleTurnParams

    from app.services.eval_llm import MeshDeepEvalModel

    judge = MeshDeepEvalModel()
    faithfulness = FaithfulnessMetric(model=judge, include_reason=True)
    relevancy = AnswerRelevancyMetric(model=judge, include_reason=True)
    grounded_persuasion = GEval(
        name="Grounded Persuasion",
        criteria=GROUNDED_PERSUASION_CRITERIA,
        evaluation_params=[
            SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.RETRIEVAL_CONTEXT,
        ],
        model=judge,
    )

    results = []
    with session_scope() as db:
        for scenario in SCENARIOS:
            candidates = _candidates_for(db, scenario["category"])
            if not candidates:
                results.append({"name": scenario["name"], "skipped": "no candidates found"})
                continue

            out = _run_generate(scenario, candidates)
            if out.get("error"):
                results.append({"name": scenario["name"], "skipped": out["error"]})
                continue

            actual_output = out["narrative"] + " " + " ".join(
                p["reason"] for p in out.get("picks", [])
            )
            case = LLMTestCase(
                input=scenario["profile_summary"] + " | " + " | ".join(scenario["evidence"]),
                actual_output=actual_output,
                retrieval_context=_context_lines(scenario, candidates),
            )

            scores = {}
            for metric in (faithfulness, relevancy, grounded_persuasion):
                metric.measure(case)
                scores[metric.__class__.__name__ if metric is not grounded_persuasion else "grounded_persuasion"] = {
                    "score": metric.score, "reason": metric.reason,
                }
            results.append({"name": scenario["name"], "scores": scores})

    return {"results": results}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate generation quality with DeepEval")
    parser.add_argument(
        "--min-score", type=float, default=0.0,
        help="exit non-zero if any metric's average falls below this (for CI gating)",
    )
    args = parser.parse_args()

    configure_logging()
    init_db()

    if not settings.has_llm or not mesh.available:
        print("LLM disabled or unavailable — set LLM_ENABLED=true and MESHAPI_API_KEY.")
        return 1

    report = measure()
    totals: dict[str, list[float]] = {}
    for r in report["results"]:
        if "skipped" in r:
            print(f"{r['name']}: skipped ({r['skipped']})")
            continue
        print(f"\n{r['name']}:")
        for metric_name, data in r["scores"].items():
            print(f"  {metric_name:22s} {data['score']:.2f}  {data['reason'][:100]}")
            totals.setdefault(metric_name, []).append(data["score"])

    print("\naverages:")
    worst = 1.0
    for metric_name, scores in totals.items():
        avg = sum(scores) / len(scores)
        worst = min(worst, avg)
        print(f"  {metric_name:22s} {avg:.2f}")

    if worst < args.min_score:
        print(f"\nFAIL: lowest average {worst:.2f} < required {args.min_score:.2f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
