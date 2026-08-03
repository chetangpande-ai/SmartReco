"""Graph shape, the groundedness verifier, and an end-to-end offline run."""

import pytest
from sqlalchemy import select

from app.agent import nodes
from app.agent.graph import describe
from app.agent.state import new_state
from app.db import session_scope
from app.models import AgentRun, Product, Recommendation, RecommendationItem, UserProfile
from app.services import guardrails, recommender
from app.services import profile as P

EXPECTED_NODES = {
    "analyze", "plan", "retrieve", "coldstart", "grade", "refine", "generate",
    "verify", "finalize",
}


class TestGraphShape:
    def test_all_nodes_registered(self):
        assert set(describe()["nodes"]) == EXPECTED_NODES

    @pytest.mark.parametrize(
        "source,target",
        [
            ("analyze", "plan"),
            ("retrieve", "grade"),
            ("refine", "retrieve"),
            ("coldstart", "generate"),
            ("generate", "verify"),
        ],
    )
    def test_edge_exists(self, source, target):
        assert (source, target) in {(e["from"], e["to"]) for e in describe()["edges"]}

    def test_refine_loop(self):
        edges = {(e["from"], e["to"]) for e in describe()["edges"]}
        assert ("grade", "refine") in edges and ("refine", "retrieve") in edges

    def test_repair_loop(self):
        assert ("verify", "generate") in {(e["from"], e["to"]) for e in describe()["edges"]}

    def test_cold_start_branch_is_conditional(self):
        edge = next(
            e for e in describe()["edges"] if e["from"] == "plan" and e["to"] == "coldstart"
        )
        assert edge["conditional"]


@pytest.fixture
def verify_state(catalog):
    ids = list(catalog.values())[:3]
    state = new_state(1, "test", "req", 3)
    state.update(
        candidates=[
            {"product_id": pid, "title": f"t{pid}", "brand": "Test", "category": "ai-ml",
             "tier": "advanced", "price_cents": 39900, "rating": 4.5, "spec": "",
             "tags": [], "description": ""}
            for pid in ids
        ],
        headline="Your next step",
        narrative="Grounded and honest copy about your interests.",
        cta="Have a look",
        grade_score=0.8,
    )
    return state, ids


class TestVerifier:
    def test_drops_hallucinated_ids(self, verify_state):
        state, ids = verify_state
        state["picks"] = [
            {"product_id": ids[0], "reason": "ok"},
            {"product_id": 999999, "reason": "invented"},
        ]
        out = nodes.verify(state)
        assert 999999 not in [p["product_id"] for p in out["picks"]]
        assert "999999" in out["repair_notes"]
        assert out["repaired"] is True

    def test_clean_run_produces_confidence(self, verify_state):
        state, ids = verify_state
        state["picks"] = [{"product_id": pid, "reason": "grounded"} for pid in ids]
        out = nodes.verify(state)
        assert not out.get("repair_notes")
        assert len(out["picks"]) == 3
        assert 0.0 <= out["confidence"] <= 1.0

    def test_catches_dishonest_copy_with_valid_ids(self, verify_state):
        """Grounded is not the same as honest."""
        state, ids = verify_state
        state["picks"] = [{"product_id": ids[0], "reason": "Guaranteed to land you a job"}]
        out = nodes.verify(state)
        assert "forbidden_claim" in out["repair_notes"]

    def test_rejects_an_unpublished_product(self, verify_state):
        state, ids = verify_state
        with session_scope() as db:
            db.get(Product, ids[0]).is_published = False
        try:
            state["picks"] = [{"product_id": ids[0], "reason": "ok"}]
            out = nodes.verify(state)
            assert ids[0] not in [p["product_id"] for p in out["picks"]]
        finally:
            with session_scope() as db:
                db.get(Product, ids[0]).is_published = True

    def test_no_picks_is_a_problem(self, verify_state):
        state, _ = verify_state
        state["picks"] = []
        assert "no valid products" in nodes.verify(state)["repair_notes"]


class TestRepairRouting:
    def test_first_failure_retries_generation(self):
        state = new_state(1, "t", "r", 3)
        state.update(repair_notes="something", node_path=["generate"])
        assert nodes.route_after_verify(state) == "generate"

    def test_second_failure_stops_looping(self):
        """An unbounded repair loop burns budget without converging."""
        state = new_state(1, "t", "r", 3)
        state.update(repair_notes="something", node_path=["generate", "generate"])
        assert nodes.route_after_verify(state) == "finalize"

    def test_clean_verification_finalises(self):
        state = new_state(1, "t", "r", 3)
        state.update(repair_notes="", node_path=["generate"])
        assert nodes.route_after_verify(state) == "finalize"


class TestColdStart:
    def test_falls_back_to_catalogue_ratings(self, catalog, user_factory):
        uid = user_factory()
        out = nodes.coldstart(new_state(uid, "t", "r", 4))
        assert len(out["candidates"]) == 4
        assert out["strategy"] == "coldstart"

    def test_plan_routes_a_signalless_user_to_coldstart(self):
        state = new_state(1, "t", "r", 4)
        state["query"] = ""
        assert nodes.route_after_plan({**state, **nodes.plan(state)}) == "coldstart"


class TestEndToEndOffline:
    @pytest.fixture
    def busy_user(self, catalog, user_factory, event_factory):
        uid = user_factory()
        ai_ids = [
            catalog["Deep Learning Specialization"],
            catalog["Natural Language Processing with Transformers"],
            catalog["Practical Deep Learning for Coders"],
        ]
        for pid in ai_ids:
            event_factory(uid, "product_view", product_id=pid, count=3)
        event_factory(uid, "search", query="deep learning neural networks", count=2)
        event_factory(uid, "dwell", product_id=ai_ids[0], dwell_ms=240_000)
        with session_scope() as db:
            P.refresh(db, uid)
        return uid

    def test_produces_a_grounded_recommendation(self, busy_user):
        rec = recommender.generate_for_user(busy_user, force=True)
        assert rec is not None

        with session_scope() as db:
            rec = db.get(Recommendation, rec.id)
            items = list(
                db.scalars(
                    select(RecommendationItem)
                    .where(RecommendationItem.recommendation_id == rec.id)
                    .order_by(RecommendationItem.rank)
                )
            )
            assert items, "a recommendation with no products is not a recommendation"
            for item in items:
                product = db.get(Product, item.product_id)
                assert product is not None and product.is_published
            assert [i.rank for i in items] == list(range(len(items)))

    def test_copy_passes_guardrails(self, busy_user):
        rec = recommender.generate_for_user(busy_user, force=True)
        with session_scope() as db:
            rec = db.get(Recommendation, rec.id)
            blob = f"{rec.headline} {rec.narrative} {rec.cta}"
        assert guardrails.check_copy(blob).ok

    def test_offline_run_spends_nothing(self, busy_user):
        rec = recommender.generate_for_user(busy_user, force=True)
        with session_scope() as db:
            run = db.scalar(select(AgentRun).where(AgentRun.recommendation_id == rec.id))
            assert run.status == "ok"
            assert run.llm_calls == 0 and run.cost_usd == 0.0
            assert run.node_path[0] == "analyze" and "verify" in run.node_path

    def test_falls_back_to_deterministic_copy_without_a_model(self, busy_user):
        rec = recommender.generate_for_user(busy_user, force=True)
        with session_scope() as db:
            assert db.get(Recommendation, rec.id).strategy == "fallback"

    def test_resets_the_drift_baseline(self, busy_user):
        recommender.generate_for_user(busy_user, force=True)
        with session_scope() as db:
            prof = db.get(UserProfile, busy_user)
            assert prof.last_rec_at is not None
            assert prof.events_since_rec == 0
            assert prof.last_rec_signature

    def test_repeat_call_is_served_from_cache(self, busy_user):
        first = recommender.generate_for_user(busy_user, force=True)
        with session_scope() as db:
            runs_before = len(recommender.recent_runs(db, 100))
        second = recommender.generate_for_user(busy_user)
        with session_scope() as db:
            assert len(recommender.recent_runs(db, 100)) == runs_before
        assert second.id == first.id

    def test_fallback_copy_reads_as_english(self, busy_user):
        """Regression: evidence facts used to be spliced in as 'You've been searched for'."""
        rec = recommender.generate_for_user(busy_user, force=True)
        with session_scope() as db:
            rec = db.get(Recommendation, rec.id)
        assert "You've been searched" not in rec.narrative
        assert "You searched" in rec.narrative or "You looked" in rec.narrative
        assert rec.narrative.endswith(".")
        assert len(rec.headline) <= 70
