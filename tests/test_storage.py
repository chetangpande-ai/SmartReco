"""Embeddings cache, vector stores, and the transactional-outbox dual write."""

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
