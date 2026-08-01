"""LangSmith wiring.

LangGraph emits a span per node automatically once the LANGSMITH_* environment is set,
so the graph shape shows up for free. What it cannot see is our Mesh calls, because
those go through the OpenAI SDK directly rather than a LangChain model — so those are
wrapped explicitly, giving a trace that reads analyse -> retrieve -> grade(llm) ->
generate(llm) -> verify end to end.

Everything degrades to a no-op when tracing is off, which is the default.
"""

import logging
import os
from collections.abc import Callable
from functools import wraps

from app.config import settings

log = logging.getLogger(__name__)
_configured = False


def configure_tracing() -> bool:
    global _configured
    if _configured:
        return settings.langsmith_tracing
    _configured = True

    if not (settings.langsmith_tracing and settings.langsmith_api_key):
        os.environ["LANGSMITH_TRACING"] = "false"
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    log.info("langsmith tracing on", extra={"project": settings.langsmith_project})
    return True


def traced(name: str, run_type: str = "chain") -> Callable:
    """Decorator that becomes a LangSmith span when tracing is enabled."""

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not configure_tracing():
                return fn(*args, **kwargs)
            from langsmith import traceable

            return traceable(name=name, run_type=run_type)(fn)(*args, **kwargs)

        return wrapper

    return decorator


def current_run_url() -> str:
    """Deep link to this run in LangSmith, stored on the AgentRun row."""
    if not configure_tracing():
        return ""
    try:
        from langsmith.run_helpers import get_current_run_tree

        tree = get_current_run_tree()
        return tree.get_url() if tree else ""
    except Exception:
        return ""  # a missing trace link is never worth failing a recommendation over
