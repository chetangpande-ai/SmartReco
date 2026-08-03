"""Digest rendering, once-only delivery, and the scheduler."""

import sys

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
    pid = catalog["Deep Learning Specialization"]
    event_factory(uid, "product_view", product_id=pid, count=6, hours_ago=1)
    event_factory(uid, "search", query="deep learning neural networks", hours_ago=1)
    with session_scope() as db:
        P.refresh(db, uid)
    recommender.generate_for_user(uid, force=True)
    return uid


class TestNotifierSelection:
    def test_file_sink_when_smtp_is_unconfigured(self):
        assert not settings.has_smtp
        assert notify.get_notifier().backend == "file"

    def test_smtp_selected_once_a_host_is_configured(self, monkeypatch):
        monkeypatch.setattr(settings, "smtp_host", "smtp.example.com")
        assert notify.get_notifier().backend == "smtp"


class FakeSMTP:
    """Stands in for smtplib.SMTP so the send path is exercised without a server.

    The MIME structure is the part that actually breaks in production — a missing
    text/plain alternative lands the mail in spam, and a wrong header set gets it
    rejected outright. None of that is visible from the file sink.
    """

    instances: list["FakeSMTP"] = []

    def __init__(self, host, port, timeout=None):
        self.host, self.port, self.timeout = host, port, timeout
        self.started_tls = False
        self.login_args = None
        self.sent = None
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starttls(self, context=None):
        self.started_tls = True

    def login(self, user, password):
        self.login_args = (user, password)

    def send_message(self, message):
        self.sent = message


class TestSmtpSend:
    @pytest.fixture
    def smtp(self, monkeypatch):
        FakeSMTP.instances.clear()
        monkeypatch.setattr(settings, "smtp_host", "smtp.example.com")
        monkeypatch.setattr(settings, "smtp_port", 587)
        monkeypatch.setattr(settings, "smtp_starttls", True)
        monkeypatch.setattr(settings, "smtp_user", "mailer@example.com")
        monkeypatch.setattr(settings, "smtp_password", "secret")
        monkeypatch.setattr(settings, "mail_from", "SmartReco <no-reply@example.com>")
        monkeypatch.setattr(notify.smtplib, "SMTP", FakeSMTP)
        return FakeSMTP

    def test_connects_to_the_configured_server(self, smtp):
        notify.SmtpNotifier().send("shopper@example.com", "Subj", "<p>hi</p>", "hi")
        conn = smtp.instances[-1]
        assert (conn.host, conn.port) == ("smtp.example.com", 587)
        assert conn.timeout == 30, "a hung SMTP server must not hang the scheduler"

    def test_upgrades_to_tls_and_authenticates(self, smtp):
        notify.SmtpNotifier().send("shopper@example.com", "Subj", "<p>hi</p>", "hi")
        conn = smtp.instances[-1]
        assert conn.started_tls
        assert conn.login_args == ("mailer@example.com", "secret")

    def test_skips_login_when_no_user_is_set(self, smtp, monkeypatch):
        monkeypatch.setattr(settings, "smtp_user", "")
        notify.SmtpNotifier().send("shopper@example.com", "Subj", "<p>hi</p>", "hi")
        assert smtp.instances[-1].login_args is None

    def test_skips_starttls_when_disabled(self, smtp, monkeypatch):
        monkeypatch.setattr(settings, "smtp_starttls", False)
        notify.SmtpNotifier().send("shopper@example.com", "Subj", "<p>hi</p>", "hi")
        assert not smtp.instances[-1].started_tls

    def test_message_is_multipart_alternative_with_both_bodies(self, smtp):
        notify.SmtpNotifier().send(
            "shopper@example.com", "Your picks", "<p>html body</p>", "text body"
        )
        message = smtp.instances[-1].sent
        assert message["To"] == "shopper@example.com"
        assert message["From"] == "SmartReco <no-reply@example.com>"
        assert message["Subject"] == "Your picks"
        assert message.get_content_type() == "multipart/alternative"

        parts = {p.get_content_type(): p.get_content() for p in message.iter_parts()}
        assert set(parts) == {"text/plain", "text/html"}
        assert "text body" in parts["text/plain"]
        assert "<p>html body</p>" in parts["text/html"]

    def test_reports_where_it_delivered(self, smtp):
        result = notify.SmtpNotifier().send("s@example.com", "S", "<p>h</p>", "t")
        assert result == "smtp:smtp.example.com"

    def test_a_dead_server_is_recorded_as_a_failed_notification(self, smtp, monkeypatch,
                                                                catalog, user_factory):
        class Refusing(FakeSMTP):
            def send_message(self, message):
                raise OSError("connection refused")

        monkeypatch.setattr(notify.smtplib, "SMTP", Refusing)
        uid = user_factory()
        with session_scope() as db:
            delivered = notify.send_once(
                db, db.get(User, uid), dedupe_key="digest:smtp:down",
                subject="s", html="<p>x</p>", text="x",
            )
        assert not delivered
        with session_scope() as db:
            note = db.scalar(
                select(Notification).where(Notification.dedupe_key == "digest:smtp:down")
            )
            assert note.status == "failed" and "connection refused" in note.error


class FakeSes:
    """The one boto3 call this app makes. Records the request so its shape is checked
    without an AWS account — SES rejects a malformed Message body outright."""

    def __init__(self):
        self.region = None
        self.sent: dict | None = None

    def client(self, service, region_name=None):
        assert service == "ses"
        self.region = region_name
        return self

    def send_email(self, **kwargs):
        self.sent = kwargs
        return {"MessageId": "0100018f-test"}


class TestSesSend:
    """SES needs an AWS account, a verified sender identity, and — in the sandbox — a
    verified recipient too, so this has never run against the real service. What a stub
    still proves is the request shape, which is what SES rejects.
    """

    @pytest.fixture
    def ses(self, monkeypatch):
        fake = FakeSes()
        monkeypatch.setitem(sys.modules, "boto3", fake)
        monkeypatch.setattr(settings, "aws_region", "eu-west-1")
        monkeypatch.setattr(settings, "mail_from", "SmartReco <no-reply@example.com>")
        return fake

    def test_selected_by_configuration(self, monkeypatch):
        monkeypatch.setattr(settings, "mail_backend", "ses")
        assert notify.get_notifier().backend == "ses"

    def test_uses_the_configured_region(self, ses):
        notify.SesNotifier().send("shopper@example.com", "Subj", "<p>hi</p>", "hi")
        assert ses.region == "eu-west-1"

    def test_sends_both_bodies_in_the_shape_ses_expects(self, ses):
        notify.SesNotifier().send("shopper@example.com", "Your picks", "<p>html</p>", "text")
        assert ses.sent["Source"] == "SmartReco <no-reply@example.com>"
        assert ses.sent["Destination"] == {"ToAddresses": ["shopper@example.com"]}
        assert ses.sent["Message"]["Subject"]["Data"] == "Your picks"
        assert ses.sent["Message"]["Body"]["Text"]["Data"] == "text"
        assert ses.sent["Message"]["Body"]["Html"]["Data"] == "<p>html</p>"

    def test_reports_where_it_delivered(self, ses):
        assert notify.SesNotifier().send("s@example.com", "S", "<p>h</p>", "t") == "ses"

    def test_a_missing_boto3_says_what_to_install(self, monkeypatch):
        """boto3 is in the optional `aws` extra. Without this, MAIL_BACKEND=ses on a
        default install dies inside a 16:00 cron job with a bare ModuleNotFoundError."""
        monkeypatch.setitem(sys.modules, "boto3", None)
        with pytest.raises(RuntimeError, match="uv sync --extra aws"):
            notify.SesNotifier().send("s@example.com", "S", "<p>h</p>", "t")


class TestAudience:
    def test_active_user_included(self, catalog, user_factory, event_factory):
        uid = user_factory()
        event_factory(uid, "product_view",
                      product_id=catalog["SQL for Data Analysis"], count=6, hours_ago=1)
        with session_scope() as db:
            assert uid in {u.id for u in digest.active_users_today(db)}

    def test_below_minimum_events_excluded(self, catalog, user_factory, event_factory):
        uid = user_factory()
        event_factory(uid, "product_view",
                      product_id=catalog["SQL for Data Analysis"], count=1, hours_ago=1)
        with session_scope() as db:
            assert uid not in {u.id for u in digest.active_users_today(db)}

    def test_opted_out_excluded(self, catalog, user_factory, event_factory):
        uid = user_factory(digest_opt_in=False)
        event_factory(uid, "product_view",
                      product_id=catalog["SQL for Data Analysis"], count=6, hours_ago=1)
        with session_scope() as db:
            assert uid not in {u.id for u in digest.active_users_today(db)}

    def test_outside_the_window_excluded(self, catalog, user_factory, event_factory):
        uid = user_factory()
        event_factory(uid, "product_view",
                      product_id=catalog["SQL for Data Analysis"], count=6, hours_ago=40)
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
