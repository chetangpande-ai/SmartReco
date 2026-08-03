"""HTTP-level integration: auth, tracking ingest, page rendering, admin access control."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.db import session_scope
from app.main import app
from app.models import Event, Product, User
from app.security import CSRF_COOKIE, CSRF_FIELD


@pytest.fixture
def client(catalog):
    with TestClient(app) as test_client:
        yield test_client


def csrf(client) -> dict:
    """Prime the CSRF cookie and return the matching form field."""
    client.get("/")
    return {CSRF_FIELD: client.cookies.get(CSRF_COOKIE)}


def register(client, email="apiuser@example.com", password="password12345"):
    return client.post(
        "/register",
        data={"email": email, "password": password, "name": "API User", **csrf(client)},
        follow_redirects=False,
    )


class TestPublicPages:
    def test_catalogue_renders(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "Deep Learning Specialization" in response.text

    def test_product_detail_exposes_tracking_attributes(self, client, catalog):
        response = client.get("/products/deep-learning-specialization")
        assert response.status_code == 200
        assert 'data-product-slug="deep-learning-specialization"' in response.text

    def test_unknown_product_is_404(self, client):
        assert client.get("/products/does-not-exist").status_code == 404

    def test_search(self, client):
        response = client.get("/search", params={"q": "sql"})
        assert "SQL for Data Analysis" in response.text

    def test_security_headers(self, client):
        headers = client.get("/").headers
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["X-Frame-Options"] == "DENY"
        assert "default-src 'self'" in headers["Content-Security-Policy"]
        assert headers["X-Request-ID"]

    def test_healthz(self, client):
        assert client.get("/healthz").json() == {"status": "ok"}

    def test_metrics_is_prometheus_text(self, client):
        response = client.get("/metrics")
        assert "text/plain" in response.headers["content-type"]
        assert "smartreco_uptime_seconds" in response.text


class TestAuth:
    def test_register_then_access_protected_page(self, client):
        assert register(client).status_code == 303
        assert client.get("/me").status_code == 200

    def test_duplicate_email_rejected(self, client):
        register(client, "dupe@example.com")
        client.cookies.clear()
        response = register(client, "dupe@example.com")
        assert response.status_code == 409

    def test_short_password_rejected(self, client):
        response = register(client, "short@example.com", "abc")
        assert response.status_code == 400

    def test_login_and_logout(self, client):
        register(client, "loginflow@example.com")
        client.post("/logout", data=csrf(client), follow_redirects=False)
        assert client.get("/me", follow_redirects=False).status_code == 303

        response = client.post(
            "/login",
            data={"email": "loginflow@example.com", "password": "password12345", "next": "/me",
                  **csrf(client)},
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert client.get("/me").status_code == 200

    def test_wrong_password_is_not_distinguishable(self, client):
        register(client, "wrongpw@example.com")
        client.post("/logout", data=csrf(client))
        bad_password = client.post(
            "/login",
            data={"email": "wrongpw@example.com", "password": "nope12345", **csrf(client)},
        )
        unknown_user = client.post(
            "/login",
            data={"email": "ghost@example.com", "password": "nope12345", **csrf(client)},
        )
        assert bad_password.status_code == unknown_user.status_code == 401
        assert "Incorrect email or password" in bad_password.text
        assert "Incorrect email or password" in unknown_user.text

    def test_anonymous_is_redirected_from_protected_pages(self, client):
        response = client.get("/me", follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == "/login?next=/me"

    def test_open_redirect_is_blocked(self, client):
        register(client, "redirect@example.com")
        client.post("/logout", data=csrf(client))
        response = client.post(
            "/login",
            data={"email": "redirect@example.com", "password": "password12345",
                  "next": "https://evil.example.com", **csrf(client)},
            follow_redirects=False,
        )
        assert response.headers["location"] == "/"


class TestCsrf:
    def test_post_without_a_token_is_rejected(self, client):
        response = client.post(
            "/register", data={"email": "nocsrf@example.com", "password": "password12345"}
        )
        assert response.status_code == 403

    def test_mismatched_token_is_rejected(self, client):
        client.get("/")
        response = client.post(
            "/register",
            data={"email": "badcsrf@example.com", "password": "password12345",
                  CSRF_FIELD: "not-the-cookie-value"},
        )
        assert response.status_code == 403


class TestEventIngest:
    def test_accepts_a_batch_and_persists_it(self, client):
        response = client.post(
            "/api/events/batch",
            json={"events": [
                {"type": "product_view", "product_slug": "practical-ethical-hacking",
                 "idem": "api-1"},
                {"type": "search", "query": "kubernetes", "idem": "api-2"},
            ]},
        )
        assert response.status_code == 202
        assert response.json()["accepted"] == 2

        _flush(client)
        with session_scope() as db:
            assert db.scalar(select(func.count()).select_from(Event)) >= 2

    def test_replayed_events_do_not_duplicate(self, client):
        payload = {"events": [{"type": "page_view", "idem": "replay-key"}]}
        client.post("/api/events/batch", json=payload)
        client.post("/api/events/batch", json=payload)
        _flush(client)
        with session_scope() as db:
            count = db.scalar(
                select(func.count()).select_from(Event).where(Event.dedupe_key == "replay-key")
            )
        assert count == 1

    def test_one_bad_event_does_not_discard_the_batch(self, client):
        """A stale cached tracker.js must not be able to kill a user's telemetry."""
        response = client.post(
            "/api/events/batch",
            json={"events": [
                {"type": "page_view", "idem": "mixed-good-1"},
                {"type": "retired_event_type", "idem": "mixed-bad"},
                {"type": "search", "query": "sql", "idem": "mixed-good-2"},
            ]},
        )
        assert response.status_code == 202
        body = response.json()
        assert body["accepted"] == 2 and body["rejected"] == 1

    def test_batch_size_is_capped(self, client):
        response = client.post(
            "/api/events/batch",
            json={"events": [{"type": "page_view", "idem": f"big-{i}"} for i in range(101)]},
        )
        assert response.status_code == 422

    def test_the_client_never_sends_a_batch_the_server_will_reject(self):
        """tracker.js chunks to MAX_BATCH before sending. If that ever exceeds the
        server's cap, a stash drain becomes a 422 — and fetch() does not reject on a 422,
        so the client discards the whole batch instead of retrying it. Read from the
        source so the two numbers cannot drift apart unnoticed.
        """
        import re

        from annotated_types import MaxLen

        from app.config import ROOT
        from app.schemas import EventBatchIn

        source = (ROOT / "app" / "static" / "js" / "tracker.js").read_text(encoding="utf-8")
        client_batch = int(re.search(r"MAX_BATCH\s*=\s*(\d+)", source).group(1))
        server_cap = next(
            m.max_length
            for m in EventBatchIn.model_fields["events"].metadata
            if isinstance(m, MaxLen)
        )
        assert client_batch <= server_cap

    def test_every_event_the_ui_emits_is_accepted_and_weighted(self):
        """Three lists have to agree: what the UI emits, what the schema accepts, and
        what the profile scores. They drifted during the domain migration — `enroll` and
        `wishlist` were wired into the buttons but never added to EVENT_TYPES, so the two
        highest-intent signals in the product were 422'd and dropped in silence.
        """
        import re

        from app.config import ROOT
        from app.schemas import EVENT_TYPES
        from app.services.profile import EVENT_WEIGHTS

        js = (ROOT / "app" / "static" / "js" / "tracker.js").read_text(encoding="utf-8")
        html = "\n".join(
            p.read_text(encoding="utf-8")
            for p in (ROOT / "app" / "templates").rglob("*.html")
        )
        emitted = set(re.findall(r'track\("([a-z_]+)"', js)) | set(
            re.findall(r'data-track-click="([a-z_]+)"', html)
        )
        # The card picks its type with a Jinja conditional, so read those out too.
        emitted |= set(re.findall(r"'([a-z_]+click)' if ", html))
        emitted |= set(re.findall(r" else '([a-z_]+click)'", html))

        assert emitted, "found no emitted event types — the regexes have rotted"
        assert emitted <= EVENT_TYPES, f"emitted but rejected by the API: {emitted - EVENT_TYPES}"
        # dwell is scored by duration in _dwell_weight(), not by a flat weight.
        assert emitted - {"dwell"} <= set(EVENT_WEIGHTS), (
            f"accepted but scored 0.1 by default: {emitted - {'dwell'} - set(EVENT_WEIGHTS)}"
        )

    def test_the_client_retries_on_a_rejected_response_not_only_a_dead_network(self):
        """fetch() resolves on 429 and 500 — only a network failure rejects it. A
        `.catch()`-only handler therefore drops exactly the batches worth retrying."""
        from app.config import ROOT

        source = (ROOT / "app" / "static" / "js" / "tracker.js").read_text(encoding="utf-8")
        assert "res.ok" in source, "tracker.js must inspect the response status"
        assert source.count("stash(batch)") >= 3  # beacon, bad status, network failure

    def test_empty_batch_is_rejected(self, client):
        assert client.post("/api/events/batch", json={"events": []}).status_code == 422

    def test_anonymous_tracking_sets_a_stable_id(self, client):
        client.post("/api/events/batch", json={"events": [{"type": "page_view", "idem": "anon-1"}]})
        assert client.cookies.get("sr_anon")

    def test_events_attach_to_the_logged_in_user(self, client):
        register(client, "tracked@example.com")
        client.post(
            "/api/events/batch",
            json={"events": [{"type": "product_view",
                              "product_slug": "sql-for-data-analysis", "idem": "attach-1"}]},
        )
        _flush(client)
        with session_scope() as db:
            user = db.scalar(select(User).where(User.email == "tracked@example.com"))
            event = db.scalar(select(Event).where(Event.dedupe_key == "attach-1"))
            assert event.user_id == user.id
            assert event.product_id is not None


def _flush(client):
    """The ingest worker batches; give it a moment to land the rows."""
    import time

    for _ in range(40):
        time.sleep(0.05)
        with session_scope() as db:
            if db.scalar(select(func.count()).select_from(Event)):
                return
    return


class TestAdminAccessControl:
    def test_anonymous_is_redirected(self, client):
        assert client.get("/admin", follow_redirects=False).status_code == 303

    def test_regular_user_is_forbidden(self, client):
        register(client, "plainuser@example.com")
        assert client.get("/admin").status_code == 403

    def test_admin_can_reach_the_dashboards(self, client):
        register(client, "adminuser@example.com")
        with session_scope() as db:
            db.scalar(select(User).where(User.email == "adminuser@example.com")).role = "admin"

        assert "Operations" in client.get("/admin").text
        assert "Catalogue" in client.get("/admin/products").text
        assert "Agent runs" in client.get("/admin/agent-runs").text

    def test_admin_can_create_a_product_and_it_reaches_the_vector_store(self, client):
        from app.services.vectorstore import get_vector_store

        register(client, "creator@example.com")
        with session_scope() as db:
            db.scalar(select(User).where(User.email == "creator@example.com")).role = "admin"

        before = get_vector_store().count()
        response = client.post(
            "/admin/products",
            data={"title": "API Created Product", "description": "Created over HTTP",
                  "category": "misc", "tier": "beginner", "tags": "api, test",
                  "price": "10", "rating": "4.0",
                  "brand": "Tester", "is_published": "on", **csrf(client)},
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert get_vector_store().count() == before + 1

        with session_scope() as db:
            product = db.scalar(select(Product).where(Product.slug == "api-created-product"))
            assert product.vector_synced_at is not None
            pid = product.id

        response = client.post(
            f"/admin/products/{pid}/delete", data=csrf(client), follow_redirects=False
        )
        assert response.status_code == 303
        assert get_vector_store().count() == before
