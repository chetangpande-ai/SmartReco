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

# (query, expected product title). Each query avoids the product's own words.
PROBES = [
    ("something to block out engine noise on a long flight",
     "Sony WH-1000XM5 Wireless Headphones"),
    ("tiny buds that stop the world when I put them in",
     "Apple AirPods Pro 2 (USB-C)"),
    ("headphones for listening properly at my desk, sound leaking out is fine",
     "Sennheiser HD 660S2 Open-Back Headphones"),
    ("quiet my commute without spending much",
     "Anker Soundcore Q30 Headphones"),
    ("one box that fills the room with sound from above",
     "Sonos Era 300 Smart Speaker"),
    ("android handset with a pen tucked inside and a huge zoom",
     "Samsung Galaxy S24 Ultra 512GB"),
    ("cheap phone that keeps getting updates for years",
     "Google Pixel 8a 128GB"),
    ("big screen machine that runs silently all day off the charger",
     "Apple MacBook Air 15in M3 16GB/512GB"),
    ("portable computer I can repair myself instead of replacing",
     "Framework Laptop 13 DIY Edition"),
    ("thin work machine with the nicest typing experience",
     "Lenovo ThinkPad X1 Carbon Gen 12"),
    ("camera body with knobs instead of menus, film-like colour",
     "Fujifilm X-T5 Body"),
    ("lightest way onto a full 35mm sensor",
     "Canon EOS R8 Body"),
    ("one piece of glass that covers most paid work",
     "Sony FE 24-70mm f/2.8 GM II Lens"),
    ("steady handheld footage of me talking while I walk",
     "DJI Osmo Pocket 3 Creator Combo"),
    ("screen with true blacks for watching films in a dark room",
     "LG C4 65in OLED evo TV"),
    ("television bright enough to beat sunlight through the window",
     "Samsung QN90D 55in Neo QLED TV"),
    ("make the mumbled dialogue in films audible",
     "Sonos Arc Ultra Soundbar"),
    ("heat only the room I am actually sitting in",
     "Ecobee Smart Thermostat Premium"),
    ("play my existing PC library on the train",
     "Valve Steam Deck OLED 1TB"),
    ("track how hard I trained and whether I have recovered, weeks between charges",
     "Garmin Forerunner 265 Music"),
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
