"""Embeddings with a persistent content-hash cache.

Two providers:
  * Mesh (`openai/text-embedding-3-small`, 1536-d) — the real one.
  * HashingEmbedder — deterministic, offline, free. Used when LLM_ENABLED=false or no
    key is set, so tests and demos run without spending credit.

The two must never be mixed inside one index: a query vector from one provider scored
against document vectors from the other is noise that *looks* like a working search.
So `Embedder.id` is stamped into the vector collection at creation, and a mismatch
triggers a reindex rather than a silent wrong answer. For the same reason a Mesh
failure during indexing raises instead of quietly degrading to hashing — the outbox
retries it later, which is the correct recovery.
"""

import hashlib
import logging
import math
import re
from collections import Counter
from contextlib import nullcontext

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import session_scope
from app.models import EmbeddingCache
from app.services.mesh import mesh

log = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")
_MESH_BATCH = 96  # texts per embeddings request


def _session(db: Session | None):
    """Borrow the caller's transaction, or own one. Borrowed sessions are not
    committed here — whoever opened the transaction decides when it lands."""
    return nullcontext(db) if db is not None else session_scope()


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class HashingEmbedder:
    """Feature hashing over unigrams and bigrams, sublinear tf, L2 normalised.

    Not semantic — "car" and "automobile" stay far apart. It is good enough for
    lexical overlap, which is all the offline path needs, and it costs nothing.
    """

    def __init__(self, dim: int):
        self.dim = dim

    @property
    def id(self) -> str:
        return f"hash:v1:{self.dim}"

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            toks = _tokens(text)
            grams = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)]
            for gram, tf in Counter(grams).items():
                h = hashlib.blake2b(gram.encode(), digest_size=8).digest()
                idx = int.from_bytes(h[:4], "big") % self.dim
                sign = 1.0 if h[4] & 1 else -1.0  # signed hashing cancels collisions
                out[row, idx] += sign * (1.0 + math.log(tf))
            norm = np.linalg.norm(out[row])
            if norm > 0:
                out[row] /= norm
        return out


class Embedder:
    def __init__(self) -> None:
        self._hashing = HashingEmbedder(settings.embedding_dim)
        self._mem: dict[str, np.ndarray] = {}

    @property
    def uses_mesh(self) -> bool:
        return settings.has_llm

    @property
    def id(self) -> str:
        if self.uses_mesh:
            return f"mesh:{settings.meshapi_embedding_model}:{settings.embedding_dim}"
        return self._hashing.id

    @property
    def dim(self) -> int:
        return settings.embedding_dim

    def embed_query(self, text: str, db: Session | None = None) -> np.ndarray:
        return self.embed_documents([text], db=db)[0]

    def embed_documents(self, texts: list[str], db: Session | None = None) -> np.ndarray:
        """Pass `db` when the caller already holds a write transaction.

        Opening a second session here would deadlock on SQLite: this write waits for
        the caller's lock, and the caller cannot commit until this returns. The busy
        timeout does not save you — it just delays the failure.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        keys = [self._key(t) for t in texts]
        cached = self._load_cached(keys, db)

        # Deduplicate within the batch. Without this a repeated text is embedded — and
        # billed — once per occurrence, and the cache insert collides on its own key.
        pending: dict[str, str] = {}
        for key, text in zip(keys, texts, strict=True):
            if key not in cached:
                pending.setdefault(key, text)

        if pending:
            pending_keys = list(pending)
            fresh = self._compute([pending[k] for k in pending_keys])
            for key, vec in zip(pending_keys, fresh, strict=True):
                cached[key] = vec
                self._mem[key] = vec
            self._persist(list(zip(pending_keys, fresh, strict=True)), db)

        log.debug(
            "embed", extra={"n": len(texts), "computed": len(pending), "id": self.id}
        )
        return np.vstack([cached[k] for k in keys]).astype(np.float32)

    def _compute(self, texts: list[str]) -> np.ndarray:
        if not self.uses_mesh:
            return self._hashing.embed(texts)

        vectors: list[list[float]] = []
        for start in range(0, len(texts), _MESH_BATCH):
            vectors.extend(mesh.embed(texts[start : start + _MESH_BATCH]))

        arr = np.asarray(vectors, dtype=np.float32)
        if arr.shape[1] != self.dim:
            raise ValueError(
                f"embedding dim {arr.shape[1]} != configured {self.dim}; "
                f"set EMBEDDING_DIM={arr.shape[1]} and reindex"
            )
        # Cosine similarity is a dot product once everything is unit length, which
        # lets the reranker and MMR skip renormalising on every comparison.
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.where(norms == 0, 1, norms)

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.id}\x00{text}".encode()).hexdigest()

    def _load_cached(self, keys: list[str], db: Session | None) -> dict[str, np.ndarray]:
        found = {k: self._mem[k] for k in keys if k in self._mem}
        want = [k for k in keys if k not in found]
        if not want:
            return found

        with _session(db) as session:
            rows = session.scalars(
                select(EmbeddingCache).where(EmbeddingCache.key.in_(want))
            ).all()
            for row in rows:
                vec = np.frombuffer(row.vector, dtype=np.float32)
                found[row.key] = vec
                self._mem[row.key] = vec
        return found

    def _persist(self, items: list[tuple[str, np.ndarray]], db: Session | None) -> None:
        if not items:
            return
        with _session(db) as session:
            existing = set(
                session.scalars(
                    select(EmbeddingCache.key).where(
                        EmbeddingCache.key.in_([k for k, _ in items])
                    )
                ).all()
            )
            session.add_all(
                [
                    EmbeddingCache(
                        key=k, embedder_id=self.id, dim=int(v.shape[0]), vector=v.tobytes()
                    )
                    for k, v in items
                    if k not in existing
                ]
            )


embedder = Embedder()
