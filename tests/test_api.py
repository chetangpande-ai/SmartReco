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
        assert "Sony WH-1000XM5 Wireless Headphones" in response.text

    def test_product_detail_exposes_tracking_attributes(self, client, catalog):
        response = client.get("/products/sony-wh-1000xm5-wireless-headphones")
        assert response.status_code == 200
        assert 'data-product-slug="sony-wh-1000xm5-wireless-headphones"' in response.text

    def test_unknown_product_is_404(self, client):
        assert client.get("/products/does-not-exist").status_code == 404

    def test_search(self, client):
        response = client.get("/search", params={"q": "oled"})
        assert "LG C4 65in OLED evo TV" in response.text

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
                {"type": "product_view", "product_slug": "hisense-u6n-55in-mini-led-tv",
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
                              "product_slug": "google-pixel-8a-128gb", "idem": "attach-1"}]},
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
                  "category": "misc", "tier": "entry", "tags": "api, test",
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
