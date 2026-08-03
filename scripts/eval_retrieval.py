"""Offline retrieval evaluation.

Twenty probes written so their wording shares as little vocabulary as possible with the
course they should find — "make my graphs actually convince the leadership team" rather
than "data visualisation". BM25 cannot answer those; only embeddings can. That makes
this a measurement of the retrieval stack rather than of keyword matching.

    uv run python scripts/eval_retrieval.py              # measure current settings
    uv run python scripts/eval_retrieval.py --sweep      # tune the relevance floor

Reports recall@1/@3/@5 and MRR. Embeddings are cached, so re-running is free.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.db import init_db, session_scope  # noqa: E402
from app.logging_conf import configure_logging  # noqa: E402
from app.services.retrieval import reset_lexical_index, retrieve  # noqa: E402

# (query, expected course title). Each query avoids the course's own *title* words while
# still naming the subject — which is what a learner actually types. A first draft left
# the topic out entirely ("get something working in week one and explain it afterwards")
# and scored recall@1 0.50; the vector side's best hit was similarity 0.29, because a
# sentence describing only a teaching style could match any course in the catalogue.
# That measured the probes, not the retrieval.
PROBES = [
    ("implement backpropagation by hand before letting a framework do it",
     "Neural Networks: Zero to Hero"),
    ("train an image classifier in the first lesson and understand it afterwards",
     "Practical Deep Learning for Coders"),
    ("the standard first course in ML, regression through to clustering",
     "Machine Learning Specialization"),
    ("fine-tune and serve a language model without the latency killing me",
     "Natural Language Processing with Transformers"),
    ("chains, memory and retrieval over my own documents in python",
     "LangChain for LLM Application Development"),
    ("generics and conditional types are fighting me in my editor",
     "Total TypeScript"),
    ("stop writing javascript tests coupled to implementation details",
     "Testing JavaScript"),
    ("no video at all, just make me write markup until the browser passes it",
     "Responsive Web Design Certification"),
    ("learn the component framework from an empty folder with no scaffolding tool",
     "Complete Intro to React"),
    ("turn raw warehouse tables into tested, documented, versioned models",
     "Analytics Engineering with dbt"),
    ("changing careers into analytics and I need a portfolio piece",
     "Google Data Analytics Certificate"),
    ("build ingestion, orchestration and streaming week by week with a cohort",
     "Data Engineering Zoomcamp"),
    ("when a statistical method is inappropriate, not just how to call it",
     "Statistics with Python Specialization"),
    ("containers through to a real deployment pipeline I actually ship to",
     "Docker and Kubernetes: The Complete Guide"),
    ("refactor live infrastructure without downtime — state, drift, modules",
     "Terraform Deep Dive"),
    ("exploit every class of web vulnerability in a lab and be told why",
     "Web Security Academy"),
    ("a 24-hour hands-on hacking exam employers actually recognise",
     "Offensive Security Certified Professional"),
    ("I write code and cannot draw — give me spacing and hierarchy rules",
     "Refactoring UI"),
    ("interview users every week instead of once a quarter",
     "Continuous Discovery Habits"),
    ("how a news feed or chat service is actually built at scale",
     "System Design Interview Course"),
]


def measure(top_n: int = 5) -> dict:
    hits = {1: 0, 3: 0, 5: 0}
    reciprocal_rank = 0.0
    pool = 0
    misses = []

    reset_lexical_index()
    with session_scope() as db:
        for query, expected in PROBES:
            result = retrieve(db, query_text=query, top_n=top_n)
            titles = [c.title for c in result.candidates]
            pool += result.stats.get("vector", 0)
            if expected in titles:
                rank = titles.index(expected) + 1
                reciprocal_rank += 1 / rank
                for k in hits:
                    if rank <= k:
                        hits[k] += 1
            else:
                misses.append((query, expected, titles[:3]))

    n = len(PROBES)
    return {
        "n": n,
        "recall@1": hits[1] / n,
        "recall@3": hits[3] / n,
        "recall@5": hits[5] / n,
        "mrr": reciprocal_rank / n,
        "avg_pool": pool / n,
        "misses": misses,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality")
    parser.add_argument("--sweep", action="store_true", help="tune the relevance floor")
    parser.add_argument("--quiet", action="store_true", help="summary only")
    parser.add_argument(
        "--min-recall", type=float, default=0.0,
        help="exit non-zero if recall@5 falls below this (for CI gating)",
    )
    args = parser.parse_args()

    configure_logging()
    init_db()
    print(f"embedder: {__import__('app.services.embeddings', fromlist=['embedder']).embedder.id}")
    print(f"probes  : {len(PROBES)} (written to share minimal vocabulary with their target)\n")

    if args.sweep:
        print(f"{'ratio':>6} {'min':>6} {'r@1':>6} {'r@3':>6} {'r@5':>6} {'mrr':>6} {'pool':>6}")
        for ratio in (0.0, 0.25, 0.35, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8):
            for floor in (0.10,):
                settings.retrieval_score_ratio = ratio
                settings.retrieval_min_score = floor
                m = measure()
                print(f"{ratio:>6.2f} {floor:>6.2f} {m['recall@1']:>6.2f} {m['recall@3']:>6.2f} "
                      f"{m['recall@5']:>6.2f} {m['mrr']:>6.3f} {m['avg_pool']:>6.1f}")
        return 0

    m = measure()
    print(f"recall@1 {m['recall@1']:.2f}   recall@3 {m['recall@3']:.2f}   "
          f"recall@5 {m['recall@5']:.2f}   MRR {m['mrr']:.3f}")
    print(f"average candidate pool: {m['avg_pool']:.1f}")

    if m["misses"] and not args.quiet:
        print(f"\n{len(m['misses'])} miss(es):")
        for query, expected, got in m["misses"]:
            print(f"  q: {query[:66]}")
            print(f"     want {expected[:56]}")
            print(f"     got  {[t[:34] for t in got]}")

    if m["recall@5"] < args.min_recall:
        print(f"\nFAIL: recall@5 {m['recall@5']:.2f} < required {args.min_recall:.2f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
