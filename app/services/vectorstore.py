"""Vector storage behind a two-implementation interface.

Chroma runs in-process with no extra service and is the default. Pinecone is selected
with VECTOR_BACKEND=pinecone for a managed deployment. Both are real, which is the only
reason the abstraction exists — the filter shape below covers exactly the four
predicates the recommender actually uses, not a general query language.

Scores are always cosine similarity in [-1, 1], higher is better. Chroma returns cosine
*distance*, so it is converted at the boundary; nothing above this file should have to
remember which backend is in use.
"""

import logging
from dataclasses import dataclass, field
from typing import Protocol

from app.config import settings

log = logging.getLogger(__name__)

COLLECTION = "smartreco_products"


@dataclass
class Filter:
    categories: list[str] | None = None
    tiers: list[str] | None = None
    max_price_cents: int | None = None
    exclude_ids: list[int] | None = None

    def is_empty(self) -> bool:
        return not (self.categories or self.tiers or self.max_price_cents or self.exclude_ids)


@dataclass
class VectorRecord:
    id: str
    embedding: list[float]
    document: str
    metadata: dict


@dataclass
class VectorHit:
    id: str
    score: float
    document: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def product_id(self) -> int:
        return int(self.id)


class VectorStore(Protocol):
    backend: str

    def upsert(self, records: list[VectorRecord]) -> None: ...
    def delete(self, ids: list[str]) -> None: ...
    def query(self, embedding: list[float], k: int, flt: Filter | None = None) -> list[VectorHit]: ...
    def all_hashes(self) -> dict[str, str]: ...
    def count(self) -> int: ...
    def reset(self) -> None: ...
    def health(self) -> dict: ...


class ChromaStore:
    backend = "chroma"

    def __init__(self, embedder_id: str, dim: int):
        import chromadb

        self.embedder_id = embedder_id
        self.dim = dim
        self._client = chromadb.PersistentClient(path=settings.chroma_dir)
        self._col = self._client.get_or_create_collection(
            name=COLLECTION,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"embedder": embedder_id, "dim": dim},
        )
        self._warn_on_embedder_mismatch()

    def _warn_on_embedder_mismatch(self) -> None:
        stored = (self._col.metadata or {}).get("embedder")
        if stored and stored != self.embedder_id:
            log.error(
                "vector index was built with a different embedder — reindex required",
                extra={"indexed_with": stored, "now_using": self.embedder_id},
            )

    @property
    def stale_embedder(self) -> bool:
        stored = (self._col.metadata or {}).get("embedder")
        return bool(stored) and stored != self.embedder_id

    def upsert(self, records: list[VectorRecord]) -> None:
        if not records:
            return
        self._col.upsert(
            ids=[r.id for r in records],
            embeddings=[r.embedding for r in records],
            documents=[r.document for r in records],
            metadatas=[r.metadata for r in records],
        )

    def delete(self, ids: list[str]) -> None:
        if ids:
            self._col.delete(ids=ids)

    def query(self, embedding: list[float], k: int, flt: Filter | None = None) -> list[VectorHit]:
        if k <= 0:
            return []
        res = self._col.query(
            query_embeddings=[embedding],
            n_results=k,
            where=self._where(flt),
            include=["metadatas", "documents", "distances"],
        )
        if not res["ids"] or not res["ids"][0]:
            return []
        return [
            VectorHit(
                id=i,
                score=1.0 - float(d),  # cosine distance -> similarity
                document=doc or "",
                metadata=meta or {},
            )
            for i, d, doc, meta in zip(
                res["ids"][0],
                res["distances"][0],
                res["documents"][0],
                res["metadatas"][0],
                strict=False,
            )
        ]

    @staticmethod
    def _where(flt: Filter | None) -> dict | None:
        if flt is None or flt.is_empty():
            return None
        clauses: list[dict] = []
        if flt.categories:
            clauses.append({"category": {"$in": flt.categories}})
        if flt.tiers:
            clauses.append({"tier": {"$in": flt.tiers}})
        if flt.max_price_cents is not None:
            clauses.append({"price_cents": {"$lte": flt.max_price_cents}})
        if flt.exclude_ids:
            clauses.append({"product_id": {"$nin": flt.exclude_ids}})
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def all_hashes(self) -> dict[str, str]:
        out: dict[str, str] = {}
        offset, page = 0, 500
        while True:
            rows = self._col.get(include=["metadatas"], limit=page, offset=offset)
            ids = rows["ids"]
            if not ids:
                return out
            for i, meta in zip(ids, rows["metadatas"], strict=False):
                out[i] = (meta or {}).get("content_hash", "")
            if len(ids) < page:
                return out
            offset += page

    def count(self) -> int:
        return self._col.count()

    def reset(self) -> None:
        self._client.delete_collection(COLLECTION)
        self._col = self._client.get_or_create_collection(
            name=COLLECTION,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"embedder": self.embedder_id, "dim": self.dim},
        )

    def health(self) -> dict:
        return {
            "backend": self.backend,
            "vectors": self.count(),
            "embedder": self.embedder_id,
            "stale_embedder": self.stale_embedder,
            "path": settings.chroma_dir,
        }


class PineconeStore:
    backend = "pinecone"

    def __init__(self, embedder_id: str, dim: int):
        from pinecone import Pinecone, ServerlessSpec

        if not settings.pinecone_api_key:
            raise RuntimeError("VECTOR_BACKEND=pinecone but PINECONE_API_KEY is unset")

        self.embedder_id = embedder_id
        self.dim = dim
        self._pc = Pinecone(api_key=settings.pinecone_api_key)
        name = settings.pinecone_index

        if not self._pc.has_index(name):
            log.info("creating pinecone index", extra={"index": name, "dim": dim})
            self._pc.create_index(
                name=name,
                dimension=dim,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=settings.pinecone_cloud, region=settings.pinecone_region
                ),
            )
        else:
            # An index's dimension is fixed at creation. Reusing one that does not match
            # the embedder fails on every upsert instead — inside the outbox worker,
            # minutes later, as a retry loop that dead-letters the whole catalogue. Chroma
            # catches the equivalent as `stale_embedder`; this is the same check.
            existing = self._pc.describe_index(name).dimension
            if existing != dim:
                raise RuntimeError(
                    f"pinecone index {name!r} is {existing}-dimensional but "
                    f"{embedder_id} produces {dim}. Point PINECONE_INDEX at a "
                    f"{dim}-dimensional index, or delete this one and let it be recreated."
                )
        self._index = self._pc.Index(name)
        self._namespace = settings.pinecone_namespace

    def upsert(self, records: list[VectorRecord]) -> None:
        if not records:
            return
        # Pinecone has no document field; the text rides along in metadata so the
        # reranker can read it back without a second round trip to SQL.
        vectors = [
            {
                "id": r.id,
                "values": r.embedding,
                "metadata": {**r.metadata, "document": r.document[:4000]},
            }
            for r in records
        ]
        for start in range(0, len(vectors), 100):
            self._index.upsert(vectors=vectors[start : start + 100], namespace=self._namespace)

    def delete(self, ids: list[str]) -> None:
        if ids:
            self._index.delete(ids=ids, namespace=self._namespace)

    def query(self, embedding: list[float], k: int, flt: Filter | None = None) -> list[VectorHit]:
        if k <= 0:
            return []
        res = self._index.query(
            vector=embedding,
            top_k=k,
            filter=self._filter(flt),
            include_metadata=True,
            namespace=self._namespace,
        )
        hits = []
        for m in res.get("matches", []):
            meta = dict(m.get("metadata") or {})
            hits.append(
                VectorHit(
                    id=m["id"],
                    score=float(m.get("score", 0.0)),
                    document=meta.pop("document", ""),
                    metadata=meta,
                )
            )
        return hits

    @staticmethod
    def _filter(flt: Filter | None) -> dict | None:
        """Pinecone treats multiple top-level keys as AND, so no $and wrapper."""
        if flt is None or flt.is_empty():
            return None
        f: dict = {}
        if flt.categories:
            f["category"] = {"$in": flt.categories}
        if flt.tiers:
            f["tier"] = {"$in": flt.tiers}
        if flt.max_price_cents is not None:
            f["price_cents"] = {"$lte": flt.max_price_cents}
        if flt.exclude_ids:
            f["product_id"] = {"$nin": flt.exclude_ids}
        return f

    def all_hashes(self) -> dict[str, str]:
        """Content hash per vector id — the whole basis of drift detection.

        `list()` paginates `ListResponse` objects, not id strings; iterating one directly
        yields nothing useful, so `fetch` was called with junk and this returned {}.
        Reconcile then read every product as `missing` and re-upserted the entire
        catalogue hourly, while `stale` and `orphaned` could never be detected at all —
        a silent no-op that looked like a working repair.
        """
        out: dict[str, str] = {}
        for page in self._index.list(namespace=self._namespace):
            ids = [item.id for item in page.vectors]
            if not ids:
                continue
            fetched = self._index.fetch(ids=ids, namespace=self._namespace)
            for vid, vec in fetched.vectors.items():
                out[vid] = (vec.metadata or {}).get("content_hash", "")
        return out

    def count(self) -> int:
        stats = self._index.describe_index_stats()
        if self._namespace:
            return int(stats.namespaces.get(self._namespace, {}).get("vector_count", 0))
        return int(stats.get("total_vector_count", 0))

    def reset(self) -> None:
        try:
            self._index.delete(delete_all=True, namespace=self._namespace)
        except Exception as exc:
            # A namespace springs into existence on first write, so delete_all against a
            # brand-new index raises 404 "Namespace not found". The requested end state —
            # nothing in there — already holds, and this is the *first* thing a fresh
            # Pinecone setup does, so failing here fails onboarding.
            if "not found" not in str(exc).lower():
                raise
            log.debug("pinecone reset: nothing to clear", extra={"ns": self._namespace or "-"})

    def health(self) -> dict:
        return {
            "backend": self.backend,
            "vectors": self.count(),
            "embedder": self.embedder_id,
            "stale_embedder": False,  # dimension mismatch surfaces as an upsert error
            "index": settings.pinecone_index,
        }


_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        from app.services.embeddings import embedder

        if settings.vector_backend == "pinecone":
            _store = PineconeStore(embedder.id, embedder.dim)
        else:
            _store = ChromaStore(embedder.id, embedder.dim)
        log.info("vector store ready", extra=_store.health())
    return _store


def reset_vector_store() -> None:
    """Drop the cached handle so the next call re-reads config. Tests use this."""
    global _store
    _store = None
