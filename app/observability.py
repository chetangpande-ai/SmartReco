"""Tracing: LangSmith for the agent graph, Logfire for everything around it.

Two backends because they answer different questions. LangSmith shows the graph — which
node ran, what the model was actually asked, why `verify` rejected a draft. Logfire shows
the request the graph was serving: the HTTP span, the SQL underneath it, the Mesh call and
its token count, all on one timeline. Neither view is complete alone.

Ordering is the whole trick. `langsmith.Client` decides once, at construction, where its
OpenTelemetry spans go: if a real global tracer provider already exists it adopts it,
otherwise it builds its own pointed at LangSmith's OTLP endpoint. So Logfire has to be up
first or LangGraph spans never reach it — and nothing errors, they just quietly go
somewhere else. `configure()` enforces that order, which is why it is one function and not
two modules.

Everything degrades to a no-op when the credentials are absent, which is the default.
"""

import logging
import os
from collections.abc import Callable
from functools import wraps

from app.config import settings

log = logging.getLogger(__name__)

_state = {"configured": False, "langsmith": False, "logfire": False, "bridge": False}


def configure() -> dict:
    """Idempotent. Safe to call from a request path, a worker, or a script."""
    if _state["configured"]:
        return _state
    _state["configured"] = True
    _state["logfire"] = _start_logfire()
    _state["langsmith"] = _start_langsmith()
    return _state


def _start_logfire() -> bool:
    if not settings.logfire_enabled:
        return False

    # A token sends to the cloud, LOGFIRE_CONSOLE prints locally. With neither, spans are
    # built and dropped — instrumentation overhead on every request buying nothing — so
    # refuse rather than pretend. Silence here is the failure mode people waste an hour on.
    if not (settings.logfire_token or settings.logfire_console):
        log.warning(
            "logfire enabled but has no sink; set LOGFIRE_TOKEN, or LOGFIRE_CONSOLE=true "
            "to print spans to stdout. Continuing without it."
        )
        return False

    import logfire

    logfire.configure(
        token=settings.logfire_token or None,
        service_name=settings.logfire_project_name,
        environment=settings.app_env,
        # Never "if-token-absent-then-prompt": logfire's interactive project setup would
        # block a container start forever. This makes a missing token a no-op export.
        send_to_logfire="if-token-present",
        console=logfire.ConsoleOptions(min_log_level="info") if settings.logfire_console else False,
        # Off by default. On, it reads local source to name spans after the calling
        # expression, which costs a stack walk per span for prettier titles.
        inspect_arguments=False,
    )

    # The OpenAI SDK is how every Mesh call leaves this process, so instrumenting it once
    # here covers completions, embeddings, retries and streaming — model, latency, prompt
    # and token counts — without a single line inside services/mesh.py.
    logfire.instrument_openai()

    from app.db import engine

    logfire.instrument_sqlalchemy(engine=engine)

    log.info(
        "logfire tracing on",
        extra={"project": settings.logfire_project_name, "cloud": bool(settings.logfire_token)},
    )
    return True


def _start_langsmith() -> bool:
    if not (settings.langsmith_tracing and settings.langsmith_api_key):
        os.environ["LANGSMITH_TRACING"] = "false"
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project

    # "hybrid" = write each run to LangSmith *and* emit it as an OTel span, which lands in
    # Logfire because Logfire owns the global provider by now. Guarded on _state["logfire"]
    # deliberately: set with no provider installed, langsmith builds its own and every run
    # arrives in LangSmith twice.
    if _state["logfire"] and settings.logfire_instrument_langchain:
        os.environ["LANGSMITH_TRACING_MODE"] = "hybrid"
        _state["bridge"] = True
    else:
        os.environ.pop("LANGSMITH_TRACING_MODE", None)

    log.info(
        "langsmith tracing on",
        extra={"project": settings.langsmith_project, "otel_bridge": _state["bridge"]},
    )
    return True


def instrument_app(app) -> None:
    """FastAPI request spans. Separate from configure() because only the web process has
    an app to hand, and configure() also runs in the scheduler and in scripts."""
    if not configure()["logfire"]:
        return
    import logfire

    logfire.instrument_fastapi(app, excluded_urls=r"/static/.*,/healthz,/metrics")


def traced(name: str, run_type: str = "chain") -> Callable:
    """Decorator that becomes a LangSmith span when tracing is enabled."""

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not configure()["langsmith"]:
                return fn(*args, **kwargs)
            from langsmith import traceable

            return traceable(name=name, run_type=run_type)(fn)(*args, **kwargs)

        return wrapper

    return decorator


def current_run_url() -> str:
    """Deep link to this run in LangSmith, stored on the AgentRun row."""
    if not configure()["langsmith"]:
        return ""
    try:
        from langsmith.run_helpers import get_current_run_tree

        tree = get_current_run_tree()
        return tree.get_url() if tree else ""
    except Exception:
        return ""  # a missing trace link is never worth failing a recommendation over


def status() -> dict:
    """Surfaced on /readyz so "are traces actually being recorded" is answerable without
    reading logs — the question you ask once the demo is already running."""
    state = configure()
    return {
        "langsmith": settings.langsmith_project if state["langsmith"] else "off",
        "logfire": (
            ("cloud" if settings.logfire_token else "console") if state["logfire"] else "off"
        ),
        "langchain_to_logfire": state["bridge"],
    }
