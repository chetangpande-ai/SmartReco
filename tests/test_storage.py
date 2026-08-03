"""Embeddings cache, vector stores, and the transactional-outbox dual write."""

from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy import delete, func, select

from app.db import session_scope
from app.models import EmbeddingCache, Product, VectorOutbox
from app.schemas import ProductIn
from app.services import outbox
from app.services.catalog import (
    compute_content_hash,
    create_product,
    delete_product,
    slugify,
    unique_slug,
    update_product,
)
from app.services.embeddings import Embedder, HashingEmbedder, embedder
from app.services.vectorstore import (
    ChromaStore,
    Filter,
    PineconeStore,
    VectorRecord,
    get_vector_store,
)


@pytest.fixture
def product_factory():
    """Products created by a test are removed again so the shared catalogue is stable."""
    created: list[int] = []

    def make(title: str, **kwargs) -> int:
        defaults = {
            "description": f"A course about {title}", "category": "misc",
            "tier": "entry", "tags": ["test"], "price_cents": 1000,
        }
        with session_scope() as db:
            product = create_product(db, ProductIn(title=title, **{**defaults, **kwargs}))
            created.append(product.id)
            return product.id

    yield make

    with session_scope() as db:
        for pid in created:
            product = db.get(Product, pid)
            if product is not None:
                delete_product(db, product)
    outbox.drain_all()
    with session_scope() as db:
        db.execute(delete(VectorOutbox).where(VectorOutbox.product_id.in_(created)))


class TestHashingEmbedder:
    def test_deterministic(self):
        h = HashingEmbedder(256)
        assert np.allclose(h.embed(["agentic ai"])[0], h.embed(["agentic ai"])[0])

    def test_unit_length(self):
        vec = HashingEmbedder(256).embed(["agentic ai with langgraph"])[0]
        assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-5)

    def test_similar_text_scores_higher_than_unrelated(self):
        h = HashingEmbedder(256)
        a, partial, unrelated = h.embed(["agentic ai langgraph", "agentic ai", "excel budgets"])
        assert float(a @ partial) > float(a @ unrelated)

    def test_empty_string_is_finite(self):
        assert np.isfinite(HashingEmbedder(256).embed([""])[0]).all()


class TestEmbeddingCache:
    def test_offline_uses_hashing_backend(self):
        assert not embedder.uses_mesh
        assert embedder.id.startswith("hash:")

    def test_duplicate_texts_in_one_batch_are_computed_once(self):
        """Regression: duplicates used to collide on the cache key and be billed twice."""
        with session_scope() as db:
            db.execute(delete(EmbeddingCache))
        texts = ["cache alpha", "cache beta", "cache alpha"]
        vectors = embedder.embed_documents(texts)
        with session_scope() as db:
            assert db.scalar(select(func.count()).select_from(EmbeddingCache)) == 2
        assert vectors.shape == (3, 256)
        assert np.allclose(vectors[0], vectors[2])

    def test_cold_embedder_reads_from_the_database(self):
        texts = ["persisted one", "persisted two"]
        embedder.embed_documents(texts)
        with session_scope() as db:
            before = db.scalar(select(func.count()).select_from(EmbeddingCache))

        cold = Embedder()  # empty in-memory cache
        again = cold.embed_documents(texts)

        with session_scope() as db:
            assert db.scalar(select(func.count()).select_from(EmbeddingCache)) == before
        assert np.allclose(again, embedder.embed_documents(texts))

    def test_empty_input(self):
        assert embedder.embed_documents([]).shape == (0, 256)


class TestFilterTranslation:
    @pytest.fixture
    def full(self):
        return Filter(categories=["audio"], tiers=["flagship"], max_price_cents=90000,
                      exclude_ids=[7])

    def test_chroma_wraps_multiple_clauses_in_and(self, full):
        where = ChromaStore._where(full)
        assert "$and" in where and len(where["$and"]) == 4

    def test_chroma_leaves_a_single_clause_unwrapped(self):
        assert "category" in ChromaStore._where(Filter(categories=["audio"]))

    def test_pinecone_uses_implicit_and(self, full):
        f = PineconeStore._filter(full)
        assert "$and" not in f and len(f) == 4

    def test_both_return_none_for_an_empty_filter(self):
        assert ChromaStore._where(Filter()) is None
        assert PineconeStore._filter(Filter()) is None


class FakeIndex:
    """The subset of a Pinecone index this app touches, recording every call."""

    def __init__(self):
        self.upserts: list[dict] = []
        self.deletes: list[dict] = []
        self.queries: list[dict] = []
        self.fetched: list[list[str]] = []
        self.stored: dict[str, str] = {}  # id -> content_hash
        self.namespace_exists = True
        self.stats = {"total_vector_count": 12, "namespaces": {"tenant-a": {"vector_count": 5}}}

    def upsert(self, vectors, namespace=None):
        self.upserts.append({"vectors": vectors, "namespace": namespace})

    def delete(self, ids=None, namespace=None, delete_all=False):
        if delete_all and not self.namespace_exists:
            raise RuntimeError("[404] Namespace not found")
        self.deletes.append({"ids": ids, "namespace": namespace, "delete_all": delete_all})

    def list(self, namespace=None):
        """Yields ListResponse *objects*, exactly as the 9.x SDK does — not id strings.
        Getting this shape wrong is what silently broke drift detection."""
        ids = sorted(self.stored)
        for start in range(0, len(ids), 3):  # several pages, so pagination is exercised
            yield SimpleNamespace(
                vectors=[SimpleNamespace(id=i) for i in ids[start : start + 3]]
            )

    def fetch(self, ids, namespace=None):
        self.fetched.append(list(ids))
        return SimpleNamespace(
            vectors={
                i: SimpleNamespace(metadata={"content_hash": self.stored[i], "title": "t"})
                for i in ids
                if i in self.stored
            }
        )

    def query(self, vector, top_k, filter=None, include_metadata=False, namespace=None):
        self.queries.append({"vector": vector, "top_k": top_k, "filter": filter,
                             "namespace": namespace})
        return {
            "matches": [
                {"id": "7", "score": 0.83,
                 "metadata": {"title": "Sony WH-1000XM5", "document": "text that rides along",
                              "content_hash": "abc"}},
            ]
        }

    def describe_index_stats(self):
        return _Stats(self.stats)


class _Stats(dict):
    """describe_index_stats returns an object with attribute access for namespaces and
    mapping access for the totals. Model both, because the code uses both."""

    @property
    def namespaces(self):
        return self["namespaces"]


class FakePinecone:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.created: list[dict] = []
        self.index = FakeIndex()
        self.existing: dict[str, int] = {}  # name -> dimension

    def has_index(self, name):
        return name in self.existing

    def describe_index(self, name):
        return SimpleNamespace(name=name, dimension=self.existing[name], metric="cosine")

    def create_index(self, name, dimension, metric, spec):
        self.created.append({"name": name, "dimension": dimension, "metric": metric,
                             "spec": spec})
        self.existing[name] = dimension

    def Index(self, name):  # noqa: N802 — mirrors the SDK's own casing
        return self.index

    # The calls under test land on the index; surface them here so assertions read
    # against one object.
    upserts = property(lambda self: self.index.upserts)
    deletes = property(lambda self: self.index.deletes)
    queries = property(lambda self: self.index.queries)


class TestPineconeBackend:
    """Pinecone is a hosted service with no local mode, so these run against a stub of
    the 9.x surface. That still catches what actually breaks on a backend swap — a wrong
    call shape, a dropped namespace, an unchunked batch — none of which the filter
    translation tests above would notice.
    """

    @pytest.fixture
    def store(self, monkeypatch):
        import pinecone

        from app.config import settings

        fake = FakePinecone()
        monkeypatch.setattr(pinecone, "Pinecone", lambda api_key=None: fake)
        monkeypatch.setattr(settings, "pinecone_api_key", "pc-test-key")
        monkeypatch.setattr(settings, "pinecone_index", "smartreco")
        monkeypatch.setattr(settings, "pinecone_namespace", "")

        def build(namespace: str = "", exists: bool = False, existing_dim: int = 8):
            monkeypatch.setattr(settings, "pinecone_namespace", namespace)
            if exists:
                fake.existing["smartreco"] = existing_dim
            return PineconeStore("mesh:test:8", 8), fake

        return build

    def test_creates_a_serverless_index_when_missing(self, store):
        _, fake = store()
        assert fake.created and fake.created[0]["dimension"] == 8
        assert fake.created[0]["metric"] == "cosine"

    def test_reuses_an_existing_index(self, store):
        _, fake = store(exists=True)
        assert fake.created == []

    def test_an_existing_index_of_the_wrong_dimension_is_refused_at_startup(self, store):
        """Found on a real account: an index named `smartreco` already existed at 1024
        dims against a 1536-dim embedder. Without this check `has_index` returns True,
        creation is skipped, and every upsert fails later inside the outbox worker —
        a retry loop that dead-letters the whole catalogue with a cryptic error.
        """
        with pytest.raises(RuntimeError, match="1024-dimensional"):
            store(exists=True, existing_dim=1024)

    def test_missing_api_key_fails_loudly(self, monkeypatch):
        from app.config import settings

        monkeypatch.setattr(settings, "pinecone_api_key", "")
        with pytest.raises(RuntimeError, match="PINECONE_API_KEY"):
            PineconeStore("mesh:test:8", 8)

    def test_upserts_are_chunked_to_the_api_limit(self, store):
        """Pinecone rejects batches over 100 vectors. A 250-product reindex arrives as
        one call without this, and fails wholesale."""
        s, fake = store(exists=True)
        s.upsert([
            VectorRecord(id=str(i), embedding=[0.1] * 8, document=f"doc {i}",
                         metadata={"product_id": i})
            for i in range(250)
        ])
        assert [len(c["vectors"]) for c in fake.upserts] == [100, 100, 50]

    def test_document_rides_along_in_metadata(self, store):
        """Pinecone has no document field, so the reranker would otherwise need a second
        round trip to SQL for every hit."""
        s, fake = store(exists=True)
        s.upsert([VectorRecord(id="1", embedding=[0.1] * 8, document="x" * 5000,
                               metadata={"product_id": 1})])
        meta = fake.upserts[0]["vectors"][0]["metadata"]
        assert meta["product_id"] == 1
        assert len(meta["document"]) == 4000  # truncated, not rejected

    def test_empty_upsert_makes_no_call(self, store):
        s, fake = store(exists=True)
        s.upsert([])
        assert fake.upserts == []

    def test_query_maps_matches_and_lifts_the_document_out(self, store):
        s, fake = store(exists=True)
        hits = s.query([0.2] * 8, 3, Filter(tiers=["flagship"]))
        assert fake.queries[0]["top_k"] == 3
        assert fake.queries[0]["filter"] == {"tier": {"$in": ["flagship"]}}
        assert len(hits) == 1
        assert hits[0].id == "7" and hits[0].score == pytest.approx(0.83)
        assert hits[0].document == "text that rides along"
        assert "document" not in hits[0].metadata and hits[0].metadata["title"]

    def test_a_zero_k_query_never_reaches_the_network(self, store):
        s, fake = store(exists=True)
        assert s.query([0.2] * 8, 0) == []
        assert fake.queries == []

    def test_namespace_is_threaded_through_every_call(self, store):
        s, fake = store(namespace="tenant-a", exists=True)
        s.upsert([VectorRecord(id="1", embedding=[0.1] * 8, document="d", metadata={})])
        s.delete(["1"])
        s.query([0.1] * 8, 2)
        s.reset()
        assert fake.upserts[0]["namespace"] == "tenant-a"
        assert fake.deletes[0]["namespace"] == "tenant-a"
        assert fake.queries[0]["namespace"] == "tenant-a"
        assert fake.deletes[-1]["delete_all"] is True

    def test_empty_delete_makes_no_call(self, store):
        s, fake = store(exists=True)
        s.delete([])
        assert fake.deletes == []

    def test_count_reads_the_namespace_when_one_is_set(self, store):
        s, _ = store(namespace="tenant-a", exists=True)
        assert s.count() == 5

    def test_count_reads_the_total_without_a_namespace(self, store):
        s, _ = store(exists=True)
        assert s.count() == 12

    def test_health_matches_the_chroma_shape(self, store):
        s, _ = store(exists=True)
        health = s.health()
        assert health["backend"] == "pinecone"
        assert {"backend", "vectors", "embedder"} <= set(health)

    def test_all_hashes_reads_ids_out_of_the_paginated_response(self, store):
        """Caught live: `list()` paginates ListResponse objects, not id strings.
        Iterating one directly fetched junk and returned {} — so reconcile saw every
        product as `missing` and re-upserted the whole catalogue hourly, while `stale`
        and `orphaned` could never be detected. A no-op that looked like a repair.
        """
        s, fake = store(exists=True)
        fake.index.stored = {str(i): f"hash-{i}" for i in range(1, 8)}

        hashes = s.all_hashes()
        assert hashes == {str(i): f"hash-{i}" for i in range(1, 8)}
        assert len(fake.index.fetched) == 3, "should follow every page"
        assert all(isinstance(i, str) for page in fake.index.fetched for i in page)

    def test_all_hashes_is_empty_on_an_empty_index(self, store):
        s, _ = store(exists=True)
        assert s.all_hashes() == {}

    def test_reset_tolerates_a_namespace_that_does_not_exist_yet(self, store):
        """A namespace springs into existence on first write, so delete_all against a
        brand-new index raises 404. reindex_all() calls reset() first — so this failed
        the very first thing a fresh Pinecone setup does."""
        s, fake = store(exists=True)
        fake.index.namespace_exists = False
        s.reset()  # must not raise: the requested end state already holds

    def test_reset_still_propagates_a_real_failure(self, store):
        s, fake = store(exists=True)

        def boom(**kwargs):
            raise RuntimeError("[403] Forbidden")

        fake.index.delete = boom
        with pytest.raises(RuntimeError, match="Forbidden"):
            s.reset()


class TestVectorStore:
    def test_catalogue_is_indexed(self, catalog):
        assert get_vector_store().count() == len(catalog)

    def test_query_returns_similarities_in_order(self, catalog):
        store = get_vector_store()
        query = embedder.embed_query("langgraph agents orchestration").tolist()
        hits = store.query(query, 3)
        assert hits
        assert all(-1.001 <= h.score <= 1.001 for h in hits)
        assert all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1))

    def test_metadata_round_trips(self, catalog):
        store = get_vector_store()
        hits = store.query(embedder.embed_query("kubernetes").tolist(), 1)
        assert hits[0].metadata["category"]
        assert hits[0].product_id == int(hits[0].id)

    def test_all_hashes_covers_the_catalogue(self, catalog):
        assert len(get_vector_store().all_hashes()) == len(catalog)

    def test_delete_of_nothing_is_a_noop(self, catalog):
        store = get_vector_store()
        before = store.count()
        store.delete([])
        assert store.count() == before


class TestSlugs:
    @pytest.mark.parametrize(
        "title,expected",
        [
            ("Agentic AI with LangGraph!", "agentic-ai-with-langgraph"),
            ("Café  Naïve —  Data 101", "cafe-naive-data-101"),
            ("!!!", "product"),
        ],
    )
    def test_slugify(self, title, expected):
        assert slugify(title) == expected

    def test_collisions_get_a_suffix(self, catalog, product_factory):
        product_factory("Duplicate Title Course")
        with session_scope() as db:
            assert unique_slug(db, "duplicate-title-course") == "duplicate-title-course-2"


class TestContentHash:
    def test_covers_publication_state(self, catalog, product_factory):
        """Regression: unpublishing used to hash identically, so the product silently
        stayed in the index and remained recommendable."""
        pid = product_factory("Hash Publication Course")
        with session_scope() as db:
            product = db.get(Product, pid)
            published = compute_content_hash(product)
            product.is_published = False
            assert compute_content_hash(product) != published

    def test_changes_with_price(self, catalog, product_factory):
        pid = product_factory("Hash Price Course", price_cents=1000)
        with session_scope() as db:
            product = db.get(Product, pid)
            before = compute_content_hash(product)
            product.price_cents = 2000
            assert compute_content_hash(product) != before


class TestDualWrite:
    def test_create_writes_sql_and_outbox_but_not_the_index(self, catalog):
        store = get_vector_store()
        before = store.count()
        with session_scope() as db:
            product = create_product(
                db, ProductIn(title="Inline Write Check", category="misc", level="beginner")
            )
            pid = product.id
            pending = db.scalars(
                select(VectorOutbox).where(
                    VectorOutbox.product_id == pid, VectorOutbox.status == "pending"
                )
            ).all()
            assert len(pending) == 1 and pending[0].op == "upsert"
            assert product.vector_synced_at is None
        assert store.count() == before, "vector must not be written inside the request"

        assert outbox.drain()["upserts"] == 1
        assert store.count() == before + 1
        with session_scope() as db:
            assert db.get(Product, pid).vector_synced_at is not None
            delete_product(db, db.get(Product, pid))
        outbox.drain_all()

    def test_noop_update_does_not_requeue(self, catalog, product_factory):
        pid = product_factory("Noop Update Course")
        outbox.drain_all()
        same = ProductIn(title="Noop Update Course", description="A course about Noop Update Course",
                         category="misc", level="beginner", tags=["test"], price_cents=1000)
        with session_scope() as db:
            update_product(db, db.get(Product, pid), same)
            pending = db.scalars(
                select(VectorOutbox).where(
                    VectorOutbox.product_id == pid, VectorOutbox.status == "pending"
                )
            ).all()
        assert pending == [], "a no-op save must not pay to re-embed"

    def test_unpublish_removes_and_republish_restores(self, catalog, product_factory):
        pid = product_factory("Visibility Course")
        outbox.drain_all()
        store = get_vector_store()
        assert str(pid) in store.all_hashes()

        base = {"title": "Visibility Course",
                "description": "A course about Visibility Course",
                "category": "misc", "tier": "entry", "tags": ["test"],
                "price_cents": 1000}
        with session_scope() as db:
            update_product(db, db.get(Product, pid), ProductIn(**base, is_published=False))
        outbox.drain_all()
        assert str(pid) not in store.all_hashes()

        with session_scope() as db:
            update_product(db, db.get(Product, pid), ProductIn(**base, is_published=True))
        outbox.drain_all()
        assert str(pid) in store.all_hashes()

    def test_rapid_edits_coalesce_to_one_embedding(self, catalog, product_factory):
        pid = product_factory("Coalesce Course")
        outbox.drain_all()
        base = {"title": "Coalesce Course",
                "description": "A course about Coalesce Course",
                "category": "misc", "tier": "entry", "tags": ["test"]}
        with session_scope() as db:
            for price in (1100, 1200, 1300):
                update_product(db, db.get(Product, pid), ProductIn(**base, price_cents=price))
        result = outbox.drain()
        assert result["upserts"] == 1 and result["superseded"] == 2

    def test_delete_outbox_row_outlives_the_product(self, catalog, product_factory):
        pid = product_factory("Doomed Course")
        outbox.drain_all()
        with session_scope() as db:
            delete_product(db, db.get(Product, pid))
            assert db.get(Product, pid) is None
            pending = db.scalars(
                select(VectorOutbox).where(
                    VectorOutbox.product_id == pid, VectorOutbox.status == "pending"
                )
            ).all()
            assert len(pending) == 1, "delete rows must survive the product they refer to"
        outbox.drain_all()
        assert str(pid) not in get_vector_store().all_hashes()


class TestOutboxFailureHandling:
    def test_retries_with_backoff_then_dead_letters(self, catalog, product_factory, monkeypatch):
        from app.models import utcnow

        pid = product_factory("Flaky Sync Course")
        store = get_vector_store()

        def boom(*args, **kwargs):
            raise RuntimeError("vector store down")

        monkeypatch.setattr(store, "upsert", boom)

        assert outbox.drain()["failed"] == 1
        with session_scope() as db:
            row = db.scalar(
                select(VectorOutbox).where(VectorOutbox.product_id == pid)
            )
            assert row.status == "pending" and row.attempts == 1
            assert "vector store down" in row.last_error
            assert row.next_attempt_at > utcnow().replace(tzinfo=None)

        for _ in range(outbox.MAX_ATTEMPTS):
            with session_scope() as db:
                for r in db.scalars(
                    select(VectorOutbox).where(VectorOutbox.status == "pending")
                ):
                    r.next_attempt_at = utcnow()
            outbox.drain()

        with session_scope() as db:
            row = db.scalar(select(VectorOutbox).where(VectorOutbox.product_id == pid))
            assert row.status == "dead" and row.attempts == outbox.MAX_ATTEMPTS
        assert outbox.health()["dead"] >= 1


class TestReconcile:
    def test_repairs_a_vector_the_index_lost(self, catalog, fresh_vector_store):
        store = fresh_vector_store
        victim = str(next(iter(catalog.values())))
        store.delete([victim])
        assert victim not in store.all_hashes()

        result = outbox.reconcile()
        assert result["missing"] == 1
        assert victim in store.all_hashes()

    def test_purges_an_orphan(self, catalog, fresh_vector_store):
        store = fresh_vector_store
        store.upsert(
            [VectorRecord(id="99999", embedding=[0.0] * 256, document="ghost",
                          metadata={"product_id": 99999, "content_hash": "x"})]
        )
        result = outbox.reconcile()
        assert result["orphaned"] == 1
        assert "99999" not in store.all_hashes()

    def test_detects_and_fixes_stale_content(self, catalog, fresh_vector_store):
        store = fresh_vector_store
        pid = str(next(iter(catalog.values())))
        with session_scope() as db:
            product = db.get(Product, int(pid))
            store.upsert(
                [VectorRecord(id=pid, embedding=[0.0] * 256, document=product.embedding_text(),
                              metadata={"product_id": int(pid), "content_hash": "TAMPERED"})]
            )
        result = outbox.reconcile()
        assert result["stale"] == 1
        assert store.all_hashes()[pid] != "TAMPERED"

    def test_healthy_catalogue_reports_in_sync(self, catalog):
        outbox.drain_all()
        health = outbox.health()
        assert health["in_sync"]
        assert health["sql_published"] == health["vector_count"]
