"""The behaviour-driven browse row on the learner's home page.

There were four rows here once. Three were cut when the dashboard was simplified — see
the module docstring in `app/services/shelves.py` for why each one went — so what is
left is the "because you viewed" row and the similarity lookup the course page shares
with it.
"""

from app.db import session_scope
from app.models import Product
from app.services import shelves as S


class TestBecauseYouViewed:
    def test_anchors_on_the_last_course_opened(self, catalog, user_factory, event_factory):
        uid = user_factory()
        event_factory(uid, "product_view", product_id=catalog["SQL for Data Analysis"], hours_ago=4)
        event_factory(uid, "product_view", product_id=catalog["Deep Learning Specialization"])

        with session_scope() as db:
            shelf = S.build(db, uid)[0]

        assert "Deep Learning Specialization" in shelf.title

    def test_never_recommends_the_anchor_back_to_the_learner(
        self, catalog, user_factory, event_factory
    ):
        uid = user_factory()
        event_factory(uid, "product_view", product_id=catalog["Deep Learning Specialization"])

        with session_scope() as db:
            shelf = S.build(db, uid)[0]

        assert catalog["Deep Learning Specialization"] not in {p.id for p in shelf.products}

    def test_an_impression_is_not_a_view(self, catalog, user_factory, event_factory):
        """`rec_impression` is exposure we manufactured. Anchoring on it would make the
        row a reply to our own last suggestion instead of to the learner."""
        uid = user_factory()
        event_factory(
            uid, "product_view", product_id=catalog["Deep Learning Specialization"], hours_ago=3
        )
        event_factory(uid, "rec_impression", product_id=catalog["Total TypeScript"])

        with session_scope() as db:
            shelf = S.build(db, uid)[0]

        assert "Deep Learning Specialization" in shelf.title

    def test_one_course_opened_nine_times_still_anchors_once(
        self, catalog, user_factory, event_factory
    ):
        uid = user_factory()
        event_factory(uid, "product_view", product_id=catalog["Total TypeScript"], count=9)

        with session_scope() as db:
            rows = S.build(db, uid)

        assert len(rows) <= 1

    def test_no_history_means_no_row_rather_than_an_empty_one(self, catalog, user_factory):
        uid = user_factory()
        with session_scope() as db:
            assert S.build(db, uid) == []


class TestNoRepeats:
    def test_what_the_page_already_shows_is_routed_around(
        self, catalog, user_factory, event_factory
    ):
        """The agent's picks and the learner's in-progress courses are above this row.
        Repeating them under a second heading is what made the old dashboard feel long."""
        uid = user_factory()
        event_factory(uid, "product_view", product_id=catalog["Deep Learning Specialization"])
        already_shown = {catalog["Machine Learning Specialization"], catalog["Practical Deep Learning for Coders"]}

        with session_scope() as db:
            rows = S.build(db, uid, already_shown)

        assert not already_shown & {p.id for s in rows for p in s.products}


class TestDismissals:
    def test_a_dismissed_course_never_comes_back(self, catalog, user_factory, event_factory):
        """The one control the learner has over what they are shown. A row that ignores
        it silently undoes the click."""
        uid = user_factory()
        event_factory(uid, "product_view", product_id=catalog["Deep Learning Specialization"])
        event_factory(uid, "dismiss", product_id=catalog["Machine Learning Specialization"])

        with session_scope() as db:
            rows = S.build(db, uid)

        assert catalog["Machine Learning Specialization"] not in {
            p.id for s in rows for p in s.products
        }

    def test_dismissing_still_leaves_the_row_standing(
        self, catalog, user_factory, event_factory
    ):
        uid = user_factory()
        event_factory(uid, "product_view", product_id=catalog["Deep Learning Specialization"])
        event_factory(uid, "dismiss", product_id=catalog["Machine Learning Specialization"])

        with session_scope() as db:
            assert S.build(db, uid)


class TestSimilarTo:
    def test_the_anchor_is_never_its_own_related_course(self, catalog):
        with session_scope() as db:
            anchor = db.get(Product, catalog["Deep Learning Specialization"])
            similar = S.similar_to(db, anchor)

        assert anchor.id not in {p.id for p in similar}

    def test_only_published_courses_come_back(self, catalog, fresh_vector_store):
        """The course page and the browse row share this. An unpublished course reaching
        either is a dead link on a page someone is already reading."""
        from app.schemas import ProductIn
        from app.services.catalog import update_product

        with session_scope() as db:
            hidden = db.get(Product, catalog["Machine Learning Specialization"])
            fields = ProductIn(
                title=hidden.title, description=hidden.description, category=hidden.category,
                tier=hidden.tier, tags=hidden.tags, price_cents=hidden.price_cents,
                rating=hidden.rating, brand=hidden.brand, spec=hidden.spec,
                teaches=hidden.teaches, requires=hidden.requires, is_published=False,
            )
            update_product(db, hidden, fields)

        try:
            with session_scope() as db:
                anchor = db.get(Product, catalog["Deep Learning Specialization"])
                similar = S.similar_to(db, anchor, limit=8)
            assert catalog["Machine Learning Specialization"] not in {p.id for p in similar}
        finally:
            with session_scope() as db:
                restored = db.get(Product, catalog["Machine Learning Specialization"])
                update_product(db, restored, fields.model_copy(update={"is_published": True}))
