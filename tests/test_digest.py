"""Digest rendering, once-only delivery, and the scheduler."""

import pytest
from markupsafe import escape
from sqlalchemy import func, select

from app import scheduler
from app.config import settings
from app.db import session_scope
from app.models import Notification, User
from app.services import digest, notify, recommender
from app.services import profile as P


@pytest.fixture
def digest_user(catalog, user_factory, event_factory):
    uid = user_factory("digest@test.local")
    pid = catalog["Sony WH-1000XM5 Wireless Headphones"]
    event_factory(uid, "product_view", product_id=pid, count=6, hours_ago=1)
    event_factory(uid, "search", query="noise cancelling headphones", hours_ago=1)
    with session_scope() as db:
        P.refresh(db, uid)
    recommender.generate_for_user(uid, force=True)
    return uid


class TestNotifierSelection:
    def test_file_sink_when_smtp_is_unconfigured(self):
        assert not settings.has_smtp
        assert notify.get_notifier().backend == "file"


class TestAudience:
    def test_active_user_included(self, catalog, user_factory, event_factory):
        uid = user_factory()
        event_factory(uid, "product_view",
                      product_id=catalog["Google Pixel 8a 128GB"], count=6, hours_ago=1)
        with session_scope() as db:
            assert uid in {u.id for u in digest.active_users_today(db)}

    def test_below_minimum_events_excluded(self, catalog, user_factory, event_factory):
        uid = user_factory()
        event_factory(uid, "product_view",
                      product_id=catalog["Google Pixel 8a 128GB"], count=1, hours_ago=1)
        with session_scope() as db:
            assert uid not in {u.id for u in digest.active_users_today(db)}

    def test_opted_out_excluded(self, catalog, user_factory, event_factory):
        uid = user_factory(digest_opt_in=False)
        event_factory(uid, "product_view",
                      product_id=catalog["Google Pixel 8a 128GB"], count=6, hours_ago=1)
        with session_scope() as db:
            assert uid not in {u.id for u in digest.active_users_today(db)}

    def test_outside_the_window_excluded(self, catalog, user_factory, event_factory):
        uid = user_factory()
        event_factory(uid, "product_view",
                      product_id=catalog["Google Pixel 8a 128GB"], count=6, hours_ago=40)
        with session_scope() as db:
            assert uid not in {u.id for u in digest.active_users_today(db)}


class TestRendering:
    def test_html_and_text_are_complete(self, digest_user):
        with session_scope() as db:
            user = db.get(User, digest_user)
            rec = recommender.get_current(db, digest_user)
            subject, html, text = digest.render_digest(db, user, rec)

        assert subject == rec.headline
        assert str(escape(rec.narrative[:40])) in html
        assert "/products/" in html and "/products/" in text
        assert settings.base_url in html
        assert "{{" not in html and "{%" not in html
        assert rec.headline in text

    def test_model_prose_is_escaped_not_injected(self, digest_user):
        with session_scope() as db:
            user = db.get(User, digest_user)
            rec = recommender.get_current(db, digest_user)
            rec.narrative = 'Nice <script>alert("xss")</script> course'
            _, html, _ = digest.render_digest(db, user, rec)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_includes_the_evidence_section(self, digest_user):
        with session_scope() as db:
            user = db.get(User, digest_user)
            rec = recommender.get_current(db, digest_user)
            _, html, _ = digest.render_digest(db, user, rec)
        assert "Why you" in html
        assert "You searched for" in html or "You looked at" in html


class TestDelivery:
    def test_sends_once_per_day(self, digest_user):
        first = digest.send_daily_digests()
        assert first["sent"] == 1

        with session_scope() as db:
            note = db.scalar(select(Notification))
            assert note.status == "sent"
            assert note.dedupe_key.startswith(f"digest:{digest_user}:")
            assert note.recommendation_id is not None

        second = digest.send_daily_digests()
        assert second["sent"] == 0
        with session_scope() as db:
            assert db.scalar(select(func.count()).select_from(Notification)) == 1

    def test_records_last_digest_at(self, digest_user):
        digest.send_daily_digests()
        with session_scope() as db:
            assert db.get(User, digest_user).last_digest_at is not None

    def test_writes_to_the_file_sink(self, digest_user):
        digest.send_daily_digests()
        files = list(notify.OUTBOX_DIR.glob("*digest_at_test.local.html"))
        assert files
        body = files[-1].read_text(encoding="utf-8")
        assert "To: digest@test.local" in body
        for f in files:
            f.unlink(missing_ok=True)

    def test_failure_is_recorded_not_swallowed(self, catalog, user_factory, monkeypatch):
        uid = user_factory()

        class Broken:
            backend = "broken"

            def send(self, *args, **kwargs):
                raise RuntimeError("smtp refused connection")

        monkeypatch.setattr(notify, "get_notifier", lambda: Broken())
        with session_scope() as db:
            delivered = notify.send_once(
                db, db.get(User, uid), dedupe_key="digest:test:fail",
                subject="s", html="<p>x</p>", text="x",
            )
        assert not delivered
        with session_scope() as db:
            note = db.scalar(
                select(Notification).where(Notification.dedupe_key == "digest:test:fail")
            )
            assert note.status == "failed" and "smtp refused" in note.error


class TestScheduler:
    @pytest.fixture
    def running(self):
        settings.scheduler_enabled = True
        sched = scheduler.start()
        yield sched
        scheduler.shutdown()
        settings.scheduler_enabled = False

    @pytest.mark.parametrize(
        "job_id", ["outbox_drain", "reconcile", "daily_digest", "prune_limiters"]
    )
    def test_job_registered(self, running, job_id):
        assert job_id in {j["id"] for j in scheduler.job_status()}

    def test_digest_uses_a_cron_trigger(self, running):
        job = next(j for j in scheduler.job_status() if j["id"] == "daily_digest")
        assert f"hour='{settings.digest_hour}'" in job["trigger"]

    def test_every_job_has_a_next_run(self, running):
        assert all(j["next_run"] for j in scheduler.job_status())

    def test_overruns_are_skipped_not_queued(self, running):
        for job in running.get_jobs():
            assert job.max_instances == 1 and job.coalesce

    def test_disabled_scheduler_is_a_noop(self):
        settings.scheduler_enabled = False
        assert scheduler.start() is None
        assert scheduler.job_status() == []
