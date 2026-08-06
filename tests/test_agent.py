"""Graph shape, the groundedness verifier, and an end-to-end offline run."""

import pytest
from sqlalchemy import select

from app.agent import nodes
from app.agent.graph import describe
from app.agent.state import new_state
from app.config import settings
from app.db import session_scope
from app.models import AgentRun, Product, Recommendation, RecommendationItem, UserProfile
from app.services import guardrails, recommender
from app.services import profile as P
from app.services.mesh import LLMResult

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


class TestAdversarialGeneration:
    """Stubs the model response itself (not just hand-built state) so this exercises the
    real generate() -> verify() integration against a hallucinating/attacking model,
    the gap flagged in the 2026-08 audit: nothing mocked mesh to test that path."""

    @pytest.fixture
    def mesh_available(self, monkeypatch):
        # Same pattern test_mesh.py uses: makes settings.has_llm true without a real key.
        monkeypatch.setattr(settings, "llm_enabled", True)
        monkeypatch.setattr(settings, "meshapi_api_key", "rsk_test")

    def _generate_state(self, catalog):
        ids = list(catalog.values())[:3]
        state = new_state(1, "test", "req", 3)
        state.update(
            profile_summary="studying: ai-ml",
            evidence=["enrolled in something"],
            candidates=[
                {"product_id": pid, "title": f"t{pid}", "brand": "Test", "category": "ai-ml",
                 "tier": "advanced", "price_cents": 39900, "rating": 4.5, "spec": "",
                 "tags": [], "description": ""}
                for pid in ids
            ],
        )
        return state, ids

    def test_a_hallucinated_pick_from_generate_is_dropped_by_verify(
        self, catalog, mesh_available, monkeypatch
    ):
        state, ids = self._generate_state(catalog)

        def fake_chat_json(messages, **kwargs):
            data = {
                "headline": "Your next step",
                "narrative": "Grounded copy about your interests.",
                "cta": "Have a look",
                "picks": [
                    {"product_id": ids[0], "reason": "real pick"},
                    {"product_id": 999999, "reason": "a course the model invented"},
                ],
            }
            result = LLMResult(text="", model="test/model", prompt_tokens=10, completion_tokens=10)
            return data, result

        monkeypatch.setattr(nodes.mesh, "chat_json", fake_chat_json)
        state.update(nodes.generate(state))
        assert 999999 in [p["product_id"] for p in state["picks"]], "sanity: the stub id reached state"

        state.update(nodes.verify(state))
        assert 999999 not in [p["product_id"] for p in state["picks"]]
        assert ids[0] in [p["product_id"] for p in state["picks"]]
        assert state["repaired"] is True

    def test_injected_hype_in_the_narrative_is_caught_by_verify(
        self, catalog, mesh_available, monkeypatch
    ):
        state, ids = self._generate_state(catalog)

        def fake_chat_json(messages, **kwargs):
            data = {
                "headline": "Guaranteed results",
                "narrative": "This course guarantees you a six-figure job, only 2 seats left.",
                "cta": "Act now",
                "picks": [{"product_id": ids[0], "reason": "real pick"}],
            }
            result = LLMResult(text="", model="test/model", prompt_tokens=10, completion_tokens=10)
            return data, result

        monkeypatch.setattr(nodes.mesh, "chat_json", fake_chat_json)
        state.update(nodes.generate(state))
        state.update(nodes.verify(state))
        assert state["repaired"] is True
        assert "forbidden_claim" in state["repair_notes"] or "fabricated_urgency" in state["repair_notes"]


class TestHeuristicGradeSkip:
    @pytest.fixture
    def mesh_available(self, monkeypatch):
        monkeypatch.setattr(settings, "llm_enabled", True)
        monkeypatch.setattr(settings, "meshapi_api_key", "rsk_test")

    def _state(self, catalog, *, clear_margin: bool):
        ids = list(catalog.values())[:2]
        state = new_state(1, "test", "req", 2)
        state["profile_summary"] = "studying: ai-ml"
        state["evidence"] = ["enrolled in something"]
        state["profile_features"] = {
            "interests": {"ai-ml": 5.0},
            "tag_scores": {"pytorch": 5.0},
            "brand_scores": {"DeepLearning.AI": 5.0},
            "price_affinity_cents": 5000,
            "tier_affinity": "advanced",
        }
        if clear_margin:
            candidates = [
                {"product_id": ids[0], "title": "match", "brand": "DeepLearning.AI",
                 "category": "ai-ml", "tier": "advanced", "price_cents": 5000, "rating": 5.0,
                 "spec": "", "tags": ["pytorch"], "description": ""},
                {"product_id": ids[1], "title": "mismatch", "brand": "Meta",
                 "category": "web-dev", "tier": "beginner", "price_cents": 50_000, "rating": 0.0,
                 "spec": "", "tags": ["react"], "description": ""},
            ]
        else:
            # Two candidates with identical heuristic inputs — margin is exactly 0.
            candidates = [
                {"product_id": pid, "title": f"t{pid}", "brand": "", "category": "",
                 "tier": "", "price_cents": 0, "rating": 0.0, "spec": "", "tags": [],
                 "description": ""}
                for pid in ids
            ]
        state["candidates"] = candidates
        return state

    def test_skips_the_llm_call_when_the_heuristic_is_confident(
        self, catalog, mesh_available, monkeypatch
    ):
        state = self._state(catalog, clear_margin=True)

        def fail_if_called(*a, **k):
            raise AssertionError("LLM should not have been called")

        monkeypatch.setattr(nodes.mesh, "chat_json", fail_if_called)
        out = nodes.grade(state)
        assert "grade_skipped_heuristic_confident" in out["warnings"]
        assert out["candidates"][0]["title"] == "match"

    def test_calls_the_llm_when_the_heuristic_is_ambiguous(
        self, catalog, mesh_available, monkeypatch
    ):
        state = self._state(catalog, clear_margin=False)

        def fake_chat_json(messages, **kwargs):
            data = {"score": 0.8, "notes": "fine", "ranked_ids": [], "better_query": ""}
            result = LLMResult(text="", model="test/model", prompt_tokens=5, completion_tokens=5)
            return data, result

        monkeypatch.setattr(nodes.mesh, "chat_json", fake_chat_json)
        out = nodes.grade(state)
        assert "grade_skipped_heuristic_confident" not in out.get("warnings", [])
        assert out["grade_score"] == 0.8


class TestExplorationIntegration:
    def test_generate_surfaces_a_low_exposure_candidate(
        self, catalog, user_factory, event_factory, monkeypatch
    ):
        """End-to-end wiring check: generate() actually queries impression_counts and
        applies the exploration slot, not just the pure function in isolation."""
        monkeypatch.setattr(settings, "explore_epsilon", 1.0)
        uid = user_factory()
        ids = list(catalog.values())[:5]
        for pid in ids[:-1]:
            event_factory(uid, "rec_impression", product_id=pid, count=10)
        # ids[-1] has never been shown — the exploration slot should force it in.

        state = new_state(uid, "test", "req", 2)
        state["profile_summary"] = "studying: ai-ml"
        state["evidence"] = []
        state["candidates"] = [
            {"product_id": pid, "title": f"t{pid}", "brand": "Test", "category": "ai-ml",
             "tier": "advanced", "price_cents": 39900, "rating": 4.5, "spec": "",
             "tags": [], "description": ""}
            for pid in ids
        ]
        out = nodes.generate(state)
        pick_ids = [p["product_id"] for p in out["picks"]]
        assert ids[-1] in pick_ids

    def test_epsilon_zero_never_forces_a_low_exposure_candidate_in(
        self, catalog, user_factory, event_factory, monkeypatch
    ):
        monkeypatch.setattr(settings, "explore_epsilon", 0.0)
        uid = user_factory()
        ids = list(catalog.values())[:5]
        for pid in ids[:-1]:
            event_factory(uid, "rec_impression", product_id=pid, count=10)

        state = new_state(uid, "test", "req", 2)
        state["profile_summary"] = "studying: ai-ml"
        state["evidence"] = []
        state["candidates"] = [
            {"product_id": pid, "title": f"t{pid}", "brand": "Test", "category": "ai-ml",
             "tier": "advanced", "price_cents": 39900, "rating": 4.5, "spec": "",
             "tags": [], "description": ""}
            for pid in ids
        ]
        out = nodes.generate(state)
        pick_ids = [p["product_id"] for p in out["picks"]]
        assert ids[-1] not in pick_ids
        assert pick_ids == ids[:2]


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

    def test_respects_exclude_ids_from_filters(self, catalog, user_factory):
        """Regression: coldstart() used to call popular() without exclude_ids at all,
        even though popular() already supported it — dismissed/committed courses would
        silently reappear on the no-history path."""
        uid = user_factory()
        excluded = list(catalog.values())[:2]
        state = new_state(uid, "t", "r", 4)
        state["filters"] = {"exclude_ids": excluded}
        out = nodes.coldstart(state)
        got_ids = {c["product_id"] for c in out["candidates"]}
        assert not got_ids & set(excluded)


class TestDismiss:
    def test_analyze_excludes_a_dismissed_product(self, catalog, user_factory, event_factory):
        uid = user_factory()
        pid = catalog["Deep Learning Specialization"]
        event_factory(uid, "dismiss", product_id=pid)
        out = nodes.analyze(new_state(uid, "t", "r", 4))
        assert pid in out["filters"]["exclude_ids"]

    def test_dismiss_does_not_register_as_interest(self, catalog, user_factory, event_factory):
        """A "not interested" click must never look like mild positive interest."""
        uid = user_factory()
        pid = catalog["Deep Learning Specialization"]
        event_factory(uid, "dismiss", product_id=pid)
        with session_scope() as db:
            prof = P.refresh(db, uid)
        assert prof.interests == {}

    def test_dismiss_is_a_valid_event_type(self):
        from app.schemas import EventIn

        assert EventIn(type="dismiss", product_id=1).type == "dismiss"

    def test_dismiss_has_zero_weight(self):
        assert P.EVENT_WEIGHTS["dismiss"] == 0.0


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
