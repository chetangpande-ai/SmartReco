"""Tracing configuration.

Every failure mode here is silent — traces go to the wrong place, or nowhere, and the
app carries on looking healthy. That is exactly why it is worth pinning down.
"""

import os
import sys
import types

import pytest
from sqlalchemy import select

from app import observability
from app.config import settings
from app.db import session_scope


class FakeLogfire(types.ModuleType):
    """Stands in for the real logfire module. Records what configure() was asked to do
    without installing a global tracer provider into the test process."""

    def __init__(self):
        super().__init__("logfire")
        self.configured: dict | None = None
        self.instrumented: list[str] = []
        self.ConsoleOptions = lambda **kw: ("console", kw)

    def configure(self, **kwargs):
        self.configured = kwargs

    def instrument_openai(self, *a, **kw):
        self.instrumented.append("openai")

    def instrument_sqlalchemy(self, *a, **kw):
        self.instrumented.append("sqlalchemy")

    def instrument_fastapi(self, *a, **kw):
        self.instrumented.append("fastapi")


LANGSMITH_ENV = (
    "LANGSMITH_TRACING", "LANGSMITH_TRACING_MODE", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT",
)


@pytest.fixture
def fresh(monkeypatch):
    """Reset the module's one-shot state, and never let a test touch the real logfire.

    configure() writes to os.environ because that is the only channel the langsmith SDK
    reads. Left behind, a stray LANGSMITH_TRACING_MODE=hybrid makes the next test that
    builds a real Client open an OTLP exporter and POST spans at api.smith.langchain.com.
    """
    fake = FakeLogfire()
    monkeypatch.setitem(sys.modules, "logfire", fake)
    env = {k: os.environ.get(k) for k in LANGSMITH_ENV}
    state = dict(observability._state)
    observability._state.update(configured=False, langsmith=False, logfire=False, bridge=False)
    yield fake
    observability._state.update(state)
    for key, value in env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def enable(monkeypatch, **overrides):
    defaults = {
        "logfire_enabled": True,
        "logfire_token": "pylf_v1_test",
        "logfire_console": False,
        "logfire_instrument_langchain": True,
        "langsmith_tracing": True,
        "langsmith_api_key": "lsv2_pt_test",
    }
    for key, value in {**defaults, **overrides}.items():
        monkeypatch.setattr(settings, key, value)


class TestOff:
    def test_no_credentials_means_no_tracing(self, fresh, monkeypatch):
        enable(monkeypatch, logfire_enabled=False, langsmith_tracing=False)
        state = observability.configure()
        assert not state["logfire"] and not state["langsmith"]
        assert fresh.configured is None

    def test_traced_is_a_transparent_passthrough(self, fresh, monkeypatch):
        enable(monkeypatch, logfire_enabled=False, langsmith_tracing=False)

        @observability.traced("noop")
        def double(n, *, offset=0):
            return n * 2 + offset

        assert double(21, offset=1) == 43

    def test_no_run_url_without_langsmith(self, fresh, monkeypatch):
        enable(monkeypatch, logfire_enabled=False, langsmith_tracing=False)
        assert observability.current_run_url() == ""

    def test_langsmith_key_without_the_flag_stays_off(self, fresh, monkeypatch):
        enable(monkeypatch, logfire_enabled=False, langsmith_tracing=False)
        monkeypatch.setattr(settings, "langsmith_api_key", "lsv2_pt_test")
        assert not observability.configure()["langsmith"]


class TestLogfire:
    def test_a_token_with_no_console_still_has_a_sink(self, fresh, monkeypatch):
        enable(monkeypatch)
        assert observability.configure()["logfire"]
        assert fresh.configured["console"] is False
        assert fresh.configured["service_name"] == settings.logfire_project_name

    def test_console_with_no_token_is_a_valid_sink(self, fresh, monkeypatch):
        enable(monkeypatch, logfire_token="", logfire_console=True)
        assert observability.configure()["logfire"]
        assert fresh.configured["token"] is None
        assert fresh.configured["console"] is not False  # a ConsoleOptions object

    def test_no_token_and_no_console_is_refused_with_a_warning(self, fresh, monkeypatch, caplog):
        """Configuring here would build a span per request and drop every one of them."""
        enable(monkeypatch, logfire_token="", logfire_console=False)
        with caplog.at_level("WARNING"):
            assert not observability.configure()["logfire"]
        assert "no sink" in caplog.text
        assert fresh.configured is None

    def test_never_blocks_boot_on_interactive_auth(self, fresh, monkeypatch):
        """send_to_logfire=True with no token makes logfire prompt for a project, which
        in a container is an indefinite hang at startup."""
        enable(monkeypatch)
        observability.configure()
        assert fresh.configured["send_to_logfire"] == "if-token-present"

    def test_instruments_the_mesh_client_and_the_database(self, fresh, monkeypatch):
        enable(monkeypatch)
        observability.configure()
        assert set(fresh.instrumented) == {"openai", "sqlalchemy"}

    def test_fastapi_is_instrumented_only_when_logfire_ran(self, fresh, monkeypatch):
        enable(monkeypatch, logfire_enabled=False)
        observability.instrument_app(object())
        assert "fastapi" not in fresh.instrumented

        observability._state.update(configured=False)
        enable(monkeypatch)
        observability.instrument_app(object())
        assert "fastapi" in fresh.instrumented


class TestLangSmithBridge:
    def test_hybrid_mode_when_logfire_owns_the_provider(self, fresh, monkeypatch):
        enable(monkeypatch)
        state = observability.configure()
        assert state["bridge"]
        assert os.environ["LANGSMITH_TRACING_MODE"] == "hybrid"

    def test_no_hybrid_without_logfire(self, fresh, monkeypatch):
        """Hybrid with no provider installed makes langsmith build its own and export to
        LangSmith twice — every run duplicated in the UI."""
        enable(monkeypatch, logfire_enabled=False)
        state = observability.configure()
        assert state["langsmith"] and not state["bridge"]
        assert "LANGSMITH_TRACING_MODE" not in os.environ

    def test_bridge_can_be_turned_off_independently(self, fresh, monkeypatch):
        enable(monkeypatch, logfire_instrument_langchain=False)
        state = observability.configure()
        assert state["logfire"] and state["langsmith"] and not state["bridge"]

    def test_langsmith_env_is_exported_for_the_sdk(self, fresh, monkeypatch):
        enable(monkeypatch, langsmith_api_key="lsv2_pt_exported")
        observability.configure()
        assert os.environ["LANGSMITH_TRACING"] == "true"
        assert os.environ["LANGSMITH_API_KEY"] == "lsv2_pt_exported"
        assert os.environ["LANGSMITH_PROJECT"] == settings.langsmith_project


class TestStatus:
    def test_reports_both_backends(self, fresh, monkeypatch):
        enable(monkeypatch)
        assert observability.status() == {
            "langsmith": settings.langsmith_project,
            "logfire": "cloud",
            "langchain_to_logfire": True,
        }

    def test_console_is_distinguishable_from_cloud(self, fresh, monkeypatch):
        enable(monkeypatch, logfire_token="", logfire_console=True)
        assert observability.status()["logfire"] == "console"

    def test_off_reads_as_off(self, fresh, monkeypatch):
        enable(monkeypatch, logfire_enabled=False, langsmith_tracing=False)
        assert observability.status() == {
            "langsmith": "off",
            "logfire": "off",
            "langchain_to_logfire": False,
        }

    def test_readyz_exposes_it(self, catalog):
        from fastapi.testclient import TestClient

        from app.main import app

        with TestClient(app) as client:
            body = client.get("/readyz").json()
        assert set(body["checks"]["tracing"]) == {
            "langsmith", "logfire", "langchain_to_logfire",
        }


class FakeLangSmith(types.ModuleType):
    """Enough of the langsmith SDK to check we call into it correctly, with no network."""

    def __init__(self, url: str = ""):
        super().__init__("langsmith")
        self.traced_as: list[tuple[str, str]] = []
        helpers = types.ModuleType("langsmith.run_helpers")
        helpers.get_current_run_tree = lambda: types.SimpleNamespace(get_url=lambda: url)
        self.run_helpers = helpers

    def traceable(self, *, name, run_type):
        self.traced_as.append((name, run_type))
        return lambda fn: fn


@pytest.fixture
def fake_langsmith(monkeypatch):
    def install(url: str = "") -> FakeLangSmith:
        fake = FakeLangSmith(url)
        monkeypatch.setitem(sys.modules, "langsmith", fake)
        monkeypatch.setitem(sys.modules, "langsmith.run_helpers", fake.run_helpers)
        return fake

    return install


class TestSpans:
    def test_traced_delegates_to_langsmith_when_on(self, fresh, monkeypatch, fake_langsmith):
        fake = fake_langsmith()
        enable(monkeypatch)

        @observability.traced("smartreco.recommend", run_type="chain")
        def run(n):
            return n + 1

        assert run(41) == 42
        assert fake.traced_as == [("smartreco.recommend", "chain")]

    def test_run_url_comes_from_the_current_run_tree(self, fresh, monkeypatch, fake_langsmith):
        fake_langsmith("https://smith.langchain.com/o/x/projects/p/y/run/z")
        enable(monkeypatch)
        assert observability.current_run_url().endswith("/run/z")

    def test_a_broken_run_tree_never_fails_the_recommendation(
        self, fresh, monkeypatch, fake_langsmith
    ):
        fake = fake_langsmith()
        fake.run_helpers.get_current_run_tree = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        enable(monkeypatch)
        assert observability.current_run_url() == ""


class TestMeshCallsAppearInTheTrace:
    """Mesh goes out through the raw OpenAI SDK. Unwrapped, a LangSmith trace is chain
    nodes with no prompts and no token counts under them."""

    def test_client_is_wrapped_when_langsmith_is_on(self, fresh, monkeypatch,
                                                    fake_langsmith):
        from app.services import mesh

        fake = fake_langsmith()
        wrappers = types.ModuleType("langsmith.wrappers")
        wrappers.wrap_openai = lambda c: ("wrapped", c)
        monkeypatch.setitem(sys.modules, "langsmith.wrappers", wrappers)
        fake.wrappers = wrappers
        enable(monkeypatch)

        assert mesh._traced_client("raw") == ("wrapped", "raw")

    def test_client_is_untouched_when_tracing_is_off(self, fresh, monkeypatch):
        from app.services import mesh

        enable(monkeypatch, logfire_enabled=False, langsmith_tracing=False)
        assert mesh._traced_client("raw") == "raw"


class TestRunUrlReachesTheDatabase:
    def test_agent_run_stores_the_trace_link(self, catalog, user_factory, event_factory,
                                             monkeypatch):
        """The link has to be read inside the traced call — read afterwards, as it once
        was, get_current_run_tree() is already None and every row stored ''."""
        from app.models import AgentRun
        from app.services import profile as P
        from app.services import recommender

        uid = user_factory()
        pid = catalog["Deep Learning Specialization"]
        event_factory(uid, "product_view", product_id=pid, count=6, hours_ago=0.05)
        event_factory(uid, "search", query="deep learning neural networks", hours_ago=0.05)
        with session_scope() as db:
            P.refresh(db, uid)

        # Model the real constraint: a run tree exists only while the traced call is on
        # the stack. Read it anywhere else and langsmith hands back None.
        marker = "https://smith.langchain.com/o/test/run/inside-the-graph"
        inside = {"now": False}
        monkeypatch.setattr(
            recommender, "current_run_url", lambda: marker if inside["now"] else ""
        )
        traced_invoke = recommender._invoke

        def invoke(state):
            inside["now"] = True
            try:
                return traced_invoke(state)
            finally:
                inside["now"] = False

        monkeypatch.setattr(recommender, "_invoke", invoke)

        recommender.generate_for_user(uid, force=True)
        with session_scope() as db:
            run = db.scalars(select(AgentRun).order_by(AgentRun.id.desc()).limit(1)).first()
        assert run.langsmith_url == marker
        # And the stub really does discriminate: read from out here, as _record_run used
        # to, it is empty — which is what every row held before.
        assert recommender.current_run_url() == ""


class TestConfigureIsIdempotent:
    def test_repeat_calls_do_not_reconfigure(self, fresh, monkeypatch):
        """configure() runs on every recommendation. Re-entering logfire.configure() would
        stack a second batch processor onto the provider each time."""
        enable(monkeypatch)
        observability.configure()
        fresh.configured = None
        observability.configure()
        observability.configure()
        assert fresh.configured is None
