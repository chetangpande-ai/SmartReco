"""In-process counters exposed at /metrics in Prometheus text format.

A real deployment would use prometheus_client. This is ~60 lines, has no dependency,
and covers exactly what we need: how many events landed, how many LLM calls we made,
and — the number that matters most here — how many we avoided.
"""

import threading
import time
from collections import defaultdict

_lock = threading.Lock()
_counters: dict[tuple[str, tuple], float] = defaultdict(float)
_gauges: dict[tuple[str, tuple], float] = {}
_started = time.time()

HELP = {
    "smartreco_events_ingested_total": "Events accepted by the ingest endpoint",
    "smartreco_events_persisted_total": "Events written to the database",
    "smartreco_events_duplicate_total": "Events dropped as duplicates by idempotency key",
    "smartreco_llm_calls_total": "LLM calls actually sent to Mesh",
    "smartreco_llm_calls_skipped_total": "Recommendation requests served without an LLM call",
    "smartreco_llm_tokens_total": "Tokens consumed via Mesh",
    "smartreco_llm_cost_usd_total": "Estimated spend in USD",
    "smartreco_agent_runs_total": "LangGraph executions by status",
    "smartreco_outbox_pending": "Vector outbox rows awaiting sync",
    "smartreco_http_requests_total": "HTTP requests by method and status",
}


def _key(name: str, labels: dict | None) -> tuple[str, tuple]:
    return name, tuple(sorted((labels or {}).items()))


def inc(name: str, value: float = 1.0, **labels) -> None:
    with _lock:
        _counters[_key(name, labels)] += value


def gauge(name: str, value: float, **labels) -> None:
    with _lock:
        _gauges[_key(name, labels)] = value


def snapshot() -> dict[str, float]:
    """Flat name{label=..} -> value view, used by the admin dashboard."""
    with _lock:
        out = {}
        for (name, labels), v in list(_counters.items()) + list(_gauges.items()):
            suffix = "{" + ",".join(f'{k}="{x}"' for k, x in labels) + "}" if labels else ""
            out[name + suffix] = v
        return out


def get(name: str, **labels) -> float:
    with _lock:
        k = _key(name, labels)
        return _counters.get(k, _gauges.get(k, 0.0))


def render_prometheus() -> str:
    lines = [
        "# HELP smartreco_uptime_seconds Seconds since process start",
        "# TYPE smartreco_uptime_seconds gauge",
        f"smartreco_uptime_seconds {time.time() - _started:.0f}",
    ]
    with _lock:
        seen: set[str] = set()
        for store, kind in ((_counters, "counter"), (_gauges, "gauge")):
            for (name, labels), value in sorted(store.items()):
                if name not in seen:
                    seen.add(name)
                    if name in HELP:
                        lines.append(f"# HELP {name} {HELP[name]}")
                    lines.append(f"# TYPE {name} {kind}")
                label_str = "{" + ",".join(f'{k}="{v}"' for k, v in labels) + "}" if labels else ""
                lines.append(f"{name}{label_str} {value}")
    return "\n".join(lines) + "\n"
