"""Liveness, readiness and metrics.

/healthz answers "is the process up" — cheap, no dependencies, safe for a load balancer
to hammer. /readyz actually touches the database and vector store, because a process
that is up but cannot reach its stores should not receive traffic.
"""

import logging

from fastapi import APIRouter, Response
from sqlalchemy import text

from app import metrics, observability
from app.db import engine
from app.services.events import ingestor
from app.services.mesh import mesh

log = logging.getLogger(__name__)
router = APIRouter(tags=["ops"])


@router.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@router.get("/readyz")
def readyz(response: Response) -> dict:
    checks: dict[str, object] = {}

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:
        checks["database"] = f"error: {exc}"

    try:
        from app.services.vectorstore import get_vector_store

        checks["vector_store"] = get_vector_store().health()
    except Exception as exc:
        checks["vector_store"] = f"error: {exc}"

    # Not a dependency: the agent degrades to deterministic copy without Mesh, so a
    # dead gateway must not take the whole app out of the load balancer.
    checks["llm"] = mesh.status()
    checks["tracing"] = observability.status()
    checks["ingest_queue_depth"] = ingestor.depth
    checks["events_dropped"] = ingestor.dropped

    failed = [k for k, v in checks.items() if isinstance(v, str) and v.startswith("error")]
    if failed:
        response.status_code = 503
        return {"status": "degraded", "failing": failed, "checks": checks}
    return {"status": "ready", "checks": checks}


@router.get("/metrics")
def prometheus_metrics() -> Response:
    return Response(content=metrics.render_prometheus(), media_type="text/plain; version=0.0.4")
