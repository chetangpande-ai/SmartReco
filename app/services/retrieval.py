"""Hybrid retrieval: dense vectors and BM25, fused, filtered, then diversified.

Why not just vector search? Because embeddings and keywords fail in opposite ways.
Dense retrieval understands that "agent orchestration" is about LangGraph, but it
happily returns something 80% similar when the user typed an exact product name.
BM25 nails exact terms — "kubernetes", "dbt", "LoRA" — and is blind to paraphrase.
Fusing them covers both, and neither needs a model call.

The stages, in order:

    vector kNN  ─┐
                 ├─ reciprocal rank fusion ─ metadata filter ─ MMR ─ top N
    BM25        ─┘

Everything here is deterministic. LLM-based re-ranking lives in the agent, where the
grader can decide it is worth paying for — keeping this module testable without a key.
"""

import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Product
from app.services.embeddings import embedder
from app.services.vectorstore import Filter, get_vector_store

log = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9][a-z0-9+#.-]*")
# Words that carry no discriminating power in a catalogue query. The shape of this list
# is domain-specific: a learner types "best free course to learn deep learning", and every
# word except "deep" and "learning" matches nothing useful while inflating document length
# in the BM25 denominator. "course", "tutorial" and "learn" appear in almost every record
# here, which is exactly what makes them worthless as discriminators.
# Deliberately NOT here: "learning" and "beginner". "learning" is in a third of these
# titles (Machine Learning, Deep Learning, LinkedIn Learning) and "beginner" is a level
# the text carries — stopwording either would throw away the strongest lexical signal in
# the catalogue to remove a word that only *looks* generic.
STOPWORDS = frozenset(
    "the a an and or for with to of in on at is are be as by from it this that how "
    "what best top good great course courses class classes tutorial tutorials lesson "
    "lessons guide intro introduction using your you my me i".split()
)

# BM25 constants. k1 controls term-frequency saturation, b how much document length
# is penalised. These are the standard defaults and there is no reason to tune them
# for a catalogue this size.
BM25_K1 = 1.5
BM25_B = 0.75

# Reciprocal rank fusion constant. 60 is the value from the original RRF paper; it is
# large enough that the difference between rank 1 and rank 2 does not swamp the
# contribution of the other ranker.
RRF_K = 60


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall((text or "").lower()) if t not in STOPWORDS and len(t) > 1]


@dataclass
class Candidate:
    product_id: int
    title: str
    category: str
    tier: str
    price_cents: int
    rating: float
    brand: str = ""
    spec: str = ""
    tags: list[str] = field(default_factory=list)
    description: str = ""

    vector_score: float = 0.0
    lexical_score: float = 0.0
    vector_rank: int = 0  # 0 == not retrieved by this ranker
    lexical_rank: int = 0
    fused_score: float = 0.0
    final_score: float = 0.0

    @property
    def retrieved_by(self) -> str:
        if self.vector_rank and self.lexical_rank:
            return "both"
        return "vector" if self.vector_rank else "lexical"


@dataclass
class RetrievalResult:
    candidates: list[Candidate]
    stats: dict = field(default_factory=dict)

    def ids(self) -> list[int]:
        return [c.product_id for c in self.candidates]


class LexicalIndex:
    """A small in-memory BM25 index over the published catalogue.

    Written out rather than pulled in as a dependency: it is forty lines, and owning
    the tokenizer means the lexical and dense sides agree on what a term is.
    """

    def __init__(self, docs: list[tuple[int, str]]):
        self.doc_tokens: dict[int, Counter] = {}
        self.doc_len: dict[int, int] = {}
        self.df: Counter = Counter()

        for product_id, text in docs:
            tokens = tokenize(text)
            counts = Counter(tokens)
            self.doc_tokens[product_id] = counts
            self.doc_len[product_id] = len(tokens)
            self.df.update(counts.keys())

        self.n_docs = len(docs)
        self.avg_len = (sum(self.doc_len.values()) / self.n_docs) if self.n_docs else 0.0

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        terms = tokenize(query)
        if not terms or not self.n_docs:
            return []

        scores: dict[int, float] = defaultdict(float)
        for term in set(terms):
            n_q = self.df.get(term, 0)
            if n_q == 0:
                continue
            idf = math.log(1 + (self.n_docs - n_q + 0.5) / (n_q + 0.5))
            for product_id, counts in self.doc_tokens.items():
                f = counts.get(term, 0)
                if not f:
                    continue
                norm = 1 - BM25_B + BM25_B * (self.doc_len[product_id] / (self.avg_len or 1))
                scores[product_id] += idf * (f * (BM25_K1 + 1)) / (f + BM25_K1 * norm)

        return sorted(scores.items(), key=lambda kv: -kv[1])[:k]


_index: LexicalIndex | None = None
_index_key: tuple | None = None


def get_lexical_index(db: Session) -> LexicalIndex:
    """Rebuilt only when the published catalogue actually changes."""
    global _index, _index_key
    key = tuple(
        db.execute(
            select(func.count(Product.id), func.max(Product.updated_at)).where(
                Product.is_published.is_(True)
            )
        ).one()
    )
    if _index is None or key != _index_key:
        rows = db.execute(
            select(
                Product.id, Product.title, Product.description, Product.category,
                Product.tags, Product.brand, Product.spec,
            ).where(Product.is_published.is_(True))
        ).all()
        docs = [
            # Title twice and tags twice: in a catalogue of short documents the title is
            # the strongest term signal and would otherwise be drowned by description.
            # Brand is indexed separately so "sony headphones" is a lexical hit.
            (r.id, f"{r.title} {r.title} {r.brand} {r.category} {' '.join(r.tags or [])} "
                   f"{' '.join(r.tags or [])} {r.spec} {r.description}")
            for r in rows
        ]
        _index = LexicalIndex(docs)
        _index_key = key
        log.info("lexical index rebuilt", extra={"docs": _index.n_docs})
    return _index


def reset_lexical_index() -> None:
    global _index, _index_key
    _index, _index_key = None, None


def rrf_fuse(
    vector_hits: list[tuple[int, float]], lexical_hits: list[tuple[int, float]]
) -> dict[int, dict]:
    """Reciprocal rank fusion: combine by *rank*, never by raw score.

    Cosine similarity and BM25 live on incomparable scales — one is bounded in [-1,1],
    the other is unbounded and depends on corpus statistics. Normalising them into a
    weighted sum means inventing an exchange rate that changes with every catalogue
    edit. Ranks have no such problem.
    """
    fused: dict[int, dict] = {}
    for rank, (product_id, score) in enumerate(vector_hits, start=1):
        entry = fused.setdefault(product_id, {"fused": 0.0, "v_rank": 0, "l_rank": 0, "v": 0.0, "l": 0.0})
        entry["fused"] += 1.0 / (RRF_K + rank)
        entry["v_rank"], entry["v"] = rank, score
    for rank, (product_id, score) in enumerate(lexical_hits, start=1):
        entry = fused.setdefault(product_id, {"fused": 0.0, "v_rank": 0, "l_rank": 0, "v": 0.0, "l": 0.0})
        entry["fused"] += 1.0 / (RRF_K + rank)
        entry["l_rank"], entry["l"] = rank, score
    return fused


def _cut_weak_hits(hits: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Drop kNN results that are only there because k had to be filled.

    A vector index returns the k nearest neighbours whether or not any of them are
    actually near. On a 35-course catalogue with k=40 that is the entire catalogue,
    and rank fusion then dutifully ranks pure noise. Keeping only hits within
    `retrieval_score_ratio` of the best one is model-agnostic: it asks "is this close
    to the best match we found" rather than "does this clear some absolute number I
    tuned for one embedding model".
    """
    if not hits:
        return []
    best = max(score for _, score in hits)
    floor = max(settings.retrieval_min_score, best * settings.retrieval_score_ratio)
    return [(pid, score) for pid, score in hits if score >= floor]


def _passes(p: Product, flt: Filter | None, exclude: set[int]) -> bool:
    """The vector store applied these already; BM25 results have not been filtered."""
    if p.id in exclude:
        return False
    if flt is None:
        return True
    if flt.categories and p.category not in flt.categories:
        return False
    if flt.tiers and p.tier not in flt.tiers:
        return False
    if flt.max_price_cents is not None and p.price_cents > flt.max_price_cents:
        return False
    return True


def mmr_select(
    candidates: list[Candidate],
    vectors: dict[int, np.ndarray],
    query_vector: np.ndarray,
    top_n: int,
    lambda_: float,
) -> list[Candidate]:
    """Maximal Marginal Relevance — relevance minus redundancy.

    Without it the top 4 for "agentic AI" are four near-identical advanced AI courses.
    A recommendation set should span a journey, not repeat one point on it.
    lambda_ = 1.0 is pure relevance; 0.0 is pure diversity.
    """
    if not candidates:
        return []

    pool = list(candidates)
    selected: list[Candidate] = []
    q = query_vector / (np.linalg.norm(query_vector) or 1.0)

    while pool and len(selected) < top_n:
        best, best_score = None, -math.inf
        for cand in pool:
            vec = vectors.get(cand.product_id)
            relevance = float(q @ vec) if vec is not None else cand.fused_score
            redundancy = 0.0
            if selected and vec is not None:
                redundancy = max(
                    (
                        float(vec @ vectors[s.product_id])
                        for s in selected
                        if s.product_id in vectors
                    ),
                    default=0.0,
                )
            score = lambda_ * relevance - (1 - lambda_) * redundancy
            if score > best_score:
                best, best_score = cand, score
        best.final_score = round(best_score, 4)
        selected.append(best)
        pool.remove(best)
    return selected


def retrieve(
    db: Session,
    *,
    query_text: str,
    query_vector: np.ndarray | None = None,
    flt: Filter | None = None,
    exclude_ids: set[int] | None = None,
    top_n: int | None = None,
    mmr_lambda: float | None = None,
    pool_size: int | None = None,
) -> RetrievalResult:
    top_n = top_n or settings.rec_top_k
    mmr_lambda = settings.retrieval_mmr_lambda if mmr_lambda is None else mmr_lambda
    pool_size = pool_size or settings.retrieval_candidates
    exclude = exclude_ids or set()

    if query_vector is None:
        query_vector = embedder.embed_query(query_text, db=db)

    store = get_vector_store()
    raw_vector = [
        (h.product_id, h.score) for h in store.query(query_vector.tolist(), pool_size, flt)
    ]
    vector_hits = _cut_weak_hits(raw_vector)
    lexical_hits = get_lexical_index(db).search(query_text, pool_size)

    fused = rrf_fuse(vector_hits, lexical_hits)
    if not fused:
        return RetrievalResult([], {"vector": 0, "lexical": 0, "fused": 0, "returned": 0})

    products = {
        p.id: p for p in db.scalars(select(Product).where(Product.id.in_(list(fused))))
    }

    candidates: list[Candidate] = []
    for product_id, entry in fused.items():
        p = products.get(product_id)
        # A vector hit with no SQL row means the index is ahead of the database. Drop
        # it: recommending a product we cannot render is worse than recommending fewer.
        if p is None or not p.is_published or not _passes(p, flt, exclude):
            continue
        candidates.append(
            Candidate(
                product_id=p.id,
                title=p.title,
                category=p.category,
                tier=p.tier,
                price_cents=p.price_cents,
                rating=p.rating,
                brand=p.brand,
                spec=p.spec,
                tags=list(p.tags or []),
                description=p.description,
                vector_score=round(entry["v"], 4),
                lexical_score=round(entry["l"], 4),
                vector_rank=entry["v_rank"],
                lexical_rank=entry["l_rank"],
                fused_score=round(entry["fused"], 6),
            )
        )

    candidates.sort(key=lambda c: -c.fused_score)
    shortlist = candidates[: max(top_n * 3, 12)]

    vectors = {}
    if shortlist:
        texts = [products[c.product_id].embedding_text() for c in shortlist]
        matrix = embedder.embed_documents(texts, db=db)
        vectors = {c.product_id: matrix[i] for i, c in enumerate(shortlist)}

    selected = mmr_select(shortlist, vectors, query_vector, top_n, mmr_lambda)

    stats = {
        "query": query_text[:120],
        "vector": len(vector_hits),
        "lexical": len(lexical_hits),
        "fused": len(fused),
        "after_filter": len(candidates),
        "returned": len(selected),
        "mmr_lambda": mmr_lambda,
        "overlap": sum(1 for c in candidates if c.retrieved_by == "both"),
        "top": [
            {"id": c.product_id, "title": c.title[:40], "by": c.retrieved_by,
             "fused": c.fused_score, "final": c.final_score}
            for c in selected
        ],
    }
    log.debug("retrieved", extra={k: stats[k] for k in ("vector", "lexical", "fused", "returned")})
    return RetrievalResult(selected, stats)


def popular(db: Session, top_n: int, exclude_ids: set[int] | None = None) -> list[Candidate]:
    """Cold-start path: no behaviour to reason about, so fall back to the catalogue's
    own signal rather than inventing an interest the user has not expressed."""
    exclude = exclude_ids or set()
    rows = db.scalars(
        select(Product)
        .where(Product.is_published.is_(True))
        .order_by(Product.rating.desc(), Product.reviews.desc())
        .limit(top_n + len(exclude))
    )
    return [
        Candidate(
            product_id=p.id, title=p.title, category=p.category, tier=p.tier,
            price_cents=p.price_cents, rating=p.rating, brand=p.brand, spec=p.spec,
            tags=list(p.tags or []), description=p.description, final_score=p.rating,
        )
        for p in rows
        if p.id not in exclude
    ][:top_n]
