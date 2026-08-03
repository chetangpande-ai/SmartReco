"""Event ingest, behaviour profile, and the trigger policy."""

from datetime import timedelta

import numpy as np
import pytest
from sqlalchemy import func, select

from app.config import settings
from app.db import session_scope
from app.models import Event, Recommendation, UserProfile, utcnow
from app.schemas import EventIn
from app.services import profile as P
from app.services import triggers as T
from app.services.events import ingestor, to_row


class TestEventRow:
    def test_maps_a_validated_event(self):
        row = to_row(
            EventIn(type="product_view", product_slug="x", idem="k1", session_id="s1"),
            user_id=7,
            anon_id="anon",
        )
        assert row["user_id"] == 7 and row["dedupe_key"] == "k1"
        assert row["_slug"] == "x", "slug is transport-only and stripped before insert"

    def test_generates_a_dedupe_key_when_missing(self):
        row = to_row(EventIn(type="page_view"), user_id=None, anon_id="a")
        assert len(row["dedupe_key"]) == 32

    def test_parses_client_timestamps(self):
        ms = int(utcnow().timestamp() * 1000)
        assert to_row(EventIn(type="page_view", ts=ms), user_id=1, anon_id="")["client_ts"]

    @pytest.mark.parametrize("ts", [1, -5, 99999999999999])
    def test_rejects_implausible_clocks(self, ts):
        """A wildly wrong client clock would otherwise corrupt the recency decay."""
        assert to_row(EventIn(type="page_view", ts=ts), user_id=1, anon_id="")["client_ts"] is None


class TestEventValidation:
    def test_rejects_unknown_types(self):
        with pytest.raises(ValueError):
            EventIn(type="not_a_real_event")

    def test_truncates_oversized_meta_rather_than_rejecting(self):
        """Losing one event's extra metadata beats losing the event."""
        event = EventIn(type="page_view", meta={f"k{i}": i for i in range(50)})
        assert len(event.meta) == 20

    def test_accepts_every_documented_type(self):
        from app.schemas import EVENT_TYPES

        for event_type in EVENT_TYPES:
            assert EventIn(type=event_type).type == event_type


class TestBulkWrite:
    def test_deduplicates_by_idempotency_key(self, catalog, user_factory):
        uid = user_factory()
        rows = [
            to_row(EventIn(type="page_view", idem="dup-1"), user_id=uid, anon_id="a"),
            to_row(EventIn(type="page_view", idem="dup-1"), user_id=uid, anon_id="a"),
            to_row(EventIn(type="page_view", idem="fresh-1"), user_id=uid, anon_id="a"),
        ]
        touched = ingestor._write(rows)
        with session_scope() as db:
            assert db.scalar(select(func.count()).select_from(Event)) == 2
        assert touched == {uid}

    def test_resolves_slugs_in_one_pass(self, catalog, user_factory):
        uid = user_factory()
        slug = "deep-learning-specialization"
        ingestor._write(
            [to_row(EventIn(type="product_view", product_slug=slug, idem="s-1"),
                    user_id=uid, anon_id="a")]
        )
        with session_scope() as db:
            event = db.scalar(select(Event).where(Event.dedupe_key == "s-1"))
            assert event.product_id == catalog["Deep Learning Specialization"]

    def test_unknown_slug_leaves_product_null(self, catalog, user_factory):
        uid = user_factory()
        ingestor._write(
            [to_row(EventIn(type="product_view", product_slug="nope", idem="s-2"),
                    user_id=uid, anon_id="a")]
        )
        with session_scope() as db:
            assert db.scalar(select(Event).where(Event.dedupe_key == "s-2")).product_id is None


class TestProfile:
    def test_recency_decay_halves_every_half_life(self, catalog, user_factory, event_factory):
        uid = user_factory()
        recent = catalog["Deep Learning Specialization"]   # ai-ml
        older = catalog["SQL for Data Analysis"]                  # data
        event_factory(uid, "product_view", product_id=recent, hours_ago=0)
        event_factory(uid, "product_view", product_id=older,
                      hours_ago=settings.profile_halflife_hours * 2)
        with session_scope() as db:
            prof = P.refresh(db, uid)
        assert prof.interests["data"] / prof.interests["ai-ml"] == pytest.approx(0.25, abs=0.02)

    def test_intent_weighting_order(self):
        w = P.EVENT_WEIGHTS
        assert w["purchase"] > w["search"] > w["product_view"] > w["page_view"] > w["rec_impression"]

    @pytest.mark.parametrize("ms,expected", [(60_000, 1.0), (180_000, 3.0), (3_600_000, 3.0)])
    def test_dwell_is_capped(self, ms, expected):
        """An abandoned tab must not be allowed to dominate the profile."""
        assert P._dwell_weight(ms) == pytest.approx(expected)

    def test_search_only_user_still_gets_a_category(self, catalog, user_factory, event_factory):
        """Regression: vocabulary used to come from the user's own history, so someone
        who only searched had none."""
        uid = user_factory()
        event_factory(uid, "search", query="ai-ml deep-learning neural-networks", count=3)
        with session_scope() as db:
            prof = P.refresh(db, uid)
        assert "ai-ml" in prof.interests
        assert "deep-learning" in prof.tag_scores
        assert prof.centroid is not None

    def test_tier_brand_and_price_affinity(self, catalog, user_factory, event_factory):
        uid = user_factory()
        for title in ["Deep Learning Specialization",
                      "Natural Language Processing with Transformers"]:
            event_factory(uid, "product_view", product_id=catalog[title], count=2)
        event_factory(uid, "product_view", product_id=catalog["Machine Learning Specialization"])
        with session_scope() as db:
            prof = P.refresh(db, uid)
        assert prof.tier_affinity == "advanced", "four advanced views outweigh one beginner view"
        assert 4900 < prof.price_affinity_cents < 8900
        assert set(prof.brand_scores) >= {"DeepLearning.AI", "O'Reilly"}

    def test_centroid_is_unit_length(self, catalog, user_factory, event_factory):
        uid = user_factory()
        event_factory(uid, "product_view", product_id=catalog["SQL for Data Analysis"], count=3)
        with session_scope() as db:
            prof = P.refresh(db, uid)
        assert np.linalg.norm(P.as_vector(prof.centroid)) == pytest.approx(1.0, abs=1e-4)

    def test_sees_events_staged_but_not_committed(self, catalog, user_factory):
        """autoflush is off, so refresh must flush or it silently under-counts."""
        uid = user_factory()
        with session_scope() as db:
            db.add(Event(user_id=uid, type="page_view", dedupe_key="staged-1"))
            prof = P.refresh(db, uid)
            assert prof.events_total == 1

    def test_no_events_is_safe(self, user_factory):
        uid = user_factory()
        with session_scope() as db:
            prof = P.refresh(db, uid)
        assert prof.events_total == 0 and prof.interests == {}


class TestEvidence:
    def test_facts_compose_after_a_you_prefix(self, catalog, user_factory, event_factory):
        uid = user_factory()
        pid = catalog["Deep Learning Specialization"]
        event_factory(uid, "search", query="deep learning", count=2)
        event_factory(uid, "dwell", product_id=pid, dwell_ms=252_000)
        event_factory(uid, "product_view", product_id=pid, count=2)
        with session_scope() as db:
            facts = P.evidence(db, uid)
        assert facts
        for fact in facts:
            assert not fact[0].isupper(), f"{fact!r} will not read as 'You {fact}'"
        assert any("searched for 'deep learning' 2 times" in f for f in facts)
        assert any("4m 12s" in f for f in facts)
        assert any("keep coming back to DeepLearning.AI" in f for f in facts)

    def test_short_dwell_is_not_reported(self, catalog, user_factory, event_factory):
        uid = user_factory()
        event_factory(uid, "dwell", product_id=catalog["SQL for Data Analysis"], dwell_ms=5_000)
        with session_scope() as db:
            assert not any("spent" in f for f in P.evidence(db, uid))


class TestDrift:
    def test_identical_centroids(self):
        v = np.array([1, 0, 0], dtype=np.float32).tobytes()
        assert P.drift(v, v) < 1e-5

    def test_orthogonal_centroids(self):
        a = np.array([1, 0, 0], dtype=np.float32).tobytes()
        b = np.array([0, 1, 0], dtype=np.float32).tobytes()
        assert P.drift(a, b) == pytest.approx(1.0, abs=1e-5)

    def test_missing_baseline_forces_a_run(self):
        v = np.array([1, 0, 0], dtype=np.float32).tobytes()
        assert P.drift(v, None) == 1.0

    def test_small_change_stays_under_the_threshold(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32).tobytes()
        b = np.array([0.99, 0.14, 0.0], dtype=np.float32).tobytes()
        assert P.drift(a, b) < settings.rec_drift_threshold


class TestSignature:
    def test_stable_when_the_ranking_does_not_change(self, catalog, user_factory, event_factory):
        """Scores move on every event; a score-based key would never hit the cache."""
        uid = user_factory()
        pid = catalog["Deep Learning Specialization"]
        event_factory(uid, "product_view", product_id=pid, count=4, hours_ago=0.2)
        with session_scope() as db:
            first = P.signature(P.refresh(db, uid))
        event_factory(uid, "product_view", product_id=pid)
        with session_scope() as db:
            assert P.signature(P.refresh(db, uid)) == first

    def test_changes_on_a_real_pivot(self, catalog, user_factory, event_factory):
        uid = user_factory()
        event_factory(uid, "product_view",
                      product_id=catalog["Deep Learning Specialization"],
                      count=4, hours_ago=0.2)
        with session_scope() as db:
            before = P.signature(P.refresh(db, uid))
        event_factory(uid, "product_view", product_id=catalog["Practical Ethical Hacking"],
                      count=12)
        event_factory(uid, "search", query="tv mini-led hdr", count=6)
        with session_scope() as db:
            assert P.signature(P.refresh(db, uid)) != before


def _decide(uid, force=False):
    with session_scope() as db:
        return T.evaluate(db, uid, db.get(UserProfile, uid), force=force)


def _give_recommendation(uid, *, signature, minutes_ago=0, ttl_minutes=120, centroid=None):
    with session_scope() as db:
        prof = db.get(UserProfile, uid)
        created = utcnow() - timedelta(minutes=minutes_ago)
        rec = Recommendation(
            user_id=uid, headline="h", narrative="n", strategy="agentic",
            behavior_signature=signature, is_current=True, created_at=created,
            expires_at=created + timedelta(minutes=ttl_minutes),
        )
        db.add(rec)
        prof.last_rec_at = created
        prof.last_rec_centroid = centroid if centroid is not None else prof.centroid
        db.flush()
        return rec.id


class TestTriggerPolicy:
    def test_no_activity(self, user_factory):
        uid = user_factory()
        with session_scope() as db:
            P.refresh(db, uid)
        assert _decide(uid).reason == "no_activity"

    def test_warming_up(self, catalog, user_factory, event_factory):
        uid = user_factory()
        event_factory(uid, "product_view", product_id=catalog["SQL for Data Analysis"], count=3)
        with session_scope() as db:
            P.refresh(db, uid)
        decision = _decide(uid)
        assert decision.reason == "warming_up" and not decision.run

    def test_first_recommendation_fires(self, catalog, user_factory, event_factory):
        uid = user_factory()
        event_factory(uid, "product_view", product_id=catalog["SQL for Data Analysis"], count=6)
        with session_scope() as db:
            P.refresh(db, uid)
        decision = _decide(uid)
        assert decision.run and decision.reason == "first_recommendation"

    def test_cache_hit_serves_without_a_call(self, catalog, user_factory, event_factory):
        uid = user_factory()
        event_factory(uid, "product_view", product_id=catalog["SQL for Data Analysis"], count=6)
        with session_scope() as db:
            signature = P.signature(P.refresh(db, uid))
        _give_recommendation(uid, signature=signature)
        with session_scope() as db:
            P.refresh(db, uid)
        decision = _decide(uid)
        assert not decision.run and decision.reason == "cache_hit"
        assert decision.cached is not None

    def test_too_few_new_events(self, catalog, user_factory, event_factory):
        uid = user_factory()
        # Setup events are backdated so they sit unambiguously *before* the
        # recommendation; at hours_ago=0 they can share its timestamp and be
        # miscounted as new, which makes the assertion race the clock.
        event_factory(uid, "product_view", product_id=catalog["SQL for Data Analysis"],
                      count=6, hours_ago=0.05)
        with session_scope() as db:
            P.refresh(db, uid)
        _give_recommendation(uid, signature="OTHER")
        event_factory(uid, "product_view", product_id=catalog["Practical Ethical Hacking"],
                      count=2)
        with session_scope() as db:
            P.refresh(db, uid)
        assert _decide(uid).reason == "too_few_new_events"

    def test_cooldown(self, catalog, user_factory, event_factory):
        uid = user_factory()
        event_factory(uid, "product_view", product_id=catalog["SQL for Data Analysis"],
                      count=6, hours_ago=0.05)
        with session_scope() as db:
            P.refresh(db, uid)
        _give_recommendation(uid, signature="OTHER")
        event_factory(uid, "product_view", product_id=catalog["Practical Ethical Hacking"],
                      count=8)
        with session_scope() as db:
            P.refresh(db, uid)
        assert _decide(uid).reason == "cooldown"

    def test_unchanged_interests_are_not_worth_a_call(self, catalog, user_factory, event_factory):
        uid = user_factory()
        pid = catalog["SQL for Data Analysis"]
        event_factory(uid, "product_view", product_id=pid, count=8, hours_ago=0.1)
        with session_scope() as db:
            prof = P.refresh(db, uid)
            centroid = prof.centroid
        _give_recommendation(uid, signature="OTHER", minutes_ago=10, centroid=centroid)
        event_factory(uid, "product_view", product_id=pid, count=6)
        with session_scope() as db:
            P.refresh(db, uid)
        decision = _decide(uid)
        assert not decision.run and decision.reason == "interests_unchanged"
        assert decision.drift < settings.rec_drift_threshold

    def test_drift_fires_a_run(self, catalog, user_factory, event_factory):
        uid = user_factory()
        event_factory(uid, "product_view", product_id=catalog["SQL for Data Analysis"],
                      count=8, hours_ago=0.1)
        with session_scope() as db:
            centroid = P.refresh(db, uid).centroid
        _give_recommendation(uid, signature="OTHER", minutes_ago=10, centroid=centroid)
        event_factory(uid, "product_view",
                      product_id=catalog["Data Engineering Zoomcamp"], count=10)
        event_factory(uid, "search", query="security pentesting exploitation", count=6)
        with session_scope() as db:
            P.refresh(db, uid)
        decision = _decide(uid)
        assert decision.run and decision.reason == "interest_drift"

    def test_staleness_fires_even_without_drift(self, catalog, user_factory, event_factory):
        uid = user_factory()
        pid = catalog["SQL for Data Analysis"]
        event_factory(uid, "product_view", product_id=pid, count=8, hours_ago=0.1)
        with session_scope() as db:
            centroid = P.refresh(db, uid).centroid
        _give_recommendation(uid, signature="OTHER", minutes_ago=10, centroid=centroid)
        event_factory(uid, "product_view", product_id=pid,
                      count=settings.rec_staleness_events + 1)
        with session_scope() as db:
            P.refresh(db, uid)
        assert _decide(uid).reason == "staleness"

    def test_expiry_forces_a_refresh(self, catalog, user_factory, event_factory):
        uid = user_factory()
        pid = catalog["SQL for Data Analysis"]
        event_factory(uid, "product_view", product_id=pid, count=8, hours_ago=0.1)
        with session_scope() as db:
            centroid = P.refresh(db, uid).centroid
        _give_recommendation(uid, signature="OTHER", minutes_ago=200, ttl_minutes=120,
                             centroid=centroid)
        event_factory(uid, "product_view", product_id=pid, count=6)
        with session_scope() as db:
            P.refresh(db, uid)
        assert _decide(uid).reason == "expired"

    def test_manual_refresh_overrides_behavioural_gates(self, catalog, user_factory,
                                                        event_factory):
        uid = user_factory()
        event_factory(uid, "product_view", product_id=catalog["SQL for Data Analysis"], count=6)
        with session_scope() as db:
            P.refresh(db, uid)
        _give_recommendation(uid, signature="OTHER")
        decision = _decide(uid, force=True)
        assert decision.run and decision.reason == "manual_refresh"

    def test_every_skip_is_attributed(self):
        stats = T.efficiency_stats()
        assert set(stats) >= {"agent_runs", "skipped", "avoided_pct", "by_reason"}
        assert all(isinstance(v, int) for v in stats["by_reason"].values())
