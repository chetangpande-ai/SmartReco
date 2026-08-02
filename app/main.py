"""Application wiring: lifespan, middleware, routers."""

import logging
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import metrics, scheduler
from app.config import ROOT, settings
from app.db import SessionLocal, init_db
from app.deps import ANON_COOKIE, new_anon_id
from app.logging_conf import configure_logging, new_request_id, request_id_var
from app.models import User
from app.routers import admin, auth, events, health, pages, recommendations
from app.security import (
    CSRF_COOKIE,
    SESSION_COOKIE,
    cookie_kwargs,
    decode_session_token,
    new_csrf_token,
)
from app.services.events import ingestor

log = logging.getLogger(__name__)

ANON_MAX_AGE = 60 * 60 * 24 * 365
CSP = (
    "default-src 'self'; "
    "img-src 'self' data:; "
    "style-src 'self' 'unsafe-inline'; "
    "script-src 'self'; "
    "connect-src 'self'; "
    "frame-ancestors 'none'; "
    "base-uri 'self'"
)


async def lifespan(app: FastAPI):
    configure_logging()
    init_db()
    await ingestor.start()
    scheduler.start()
    log.info(
        "smartreco up",
        extra={"env": settings.app_env, "llm": settings.has_llm, "model": settings.meshapi_model},
    )
    yield
    scheduler.shutdown()
    await ingestor.stop()


app = FastAPI(
    title="SmartReco",
    description="Behavioral AI recommendation agent",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(ROOT / "app" / "static")), name="static")
app.include_router(health.router)
app.include_router(events.router)
app.include_router(auth.router)
app.include_router(recommendations.router)
app.include_router(admin.router)
app.include_router(pages.router)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """One pass for: request id, identity, CSRF token, timing, security headers."""
    request_id_var.set(new_request_id())
    started = time.perf_counter()

    is_static = request.url.path.startswith("/static")
    request.state.user = None
    if not is_static:
        request.state.user = _load_user(request)

    csrf = request.cookies.get(CSRF_COOKIE) or new_csrf_token()
    request.state.csrf = csrf
    anon_id = request.cookies.get(ANON_COOKIE) or new_anon_id()
    request.state.anon_id = anon_id

    response = await call_next(request)

    if not is_static:
        metrics.inc(
            "smartreco_http_requests_total",
            method=request.method,
            status=response.status_code,
        )
        response.headers["X-Request-ID"] = request_id_var.get()
        response.headers["X-Response-Time-ms"] = f"{(time.perf_counter() - started) * 1000:.1f}"
        response.headers["Content-Security-Policy"] = CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

    if CSRF_COOKIE not in request.cookies:
        response.set_cookie(CSRF_COOKIE, csrf, **cookie_kwargs(ANON_MAX_AGE))
    if ANON_COOKIE not in request.cookies:
        response.set_cookie(ANON_COOKIE, anon_id, **cookie_kwargs(ANON_MAX_AGE))
    return response


def _load_user(request: Request) -> User | None:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    payload = decode_session_token(token)
    if not payload:
        return None
    db = SessionLocal()
    try:
        return db.get(User, int(payload["sub"]))
    finally:
        db.close()


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Browsers get a redirect to the login page; fetch/XHR gets JSON. One dependency
    (`require_user`) then serves both the pages and the API.

    Path is checked as well as Accept because Accept is not reliable — clients send
    `*/*`, and a navigation that 401s would then dump raw JSON into the viewport
    instead of showing a login page. Everything under /api/ is machine-facing by
    construction, so that split is the dependable signal.
    """
    is_api = request.url.path.startswith("/api/")
    wants_html = not is_api or "text/html" in request.headers.get("accept", "")
    if exc.status_code == 401 and wants_html:
        return RedirectResponse(f"/login?next={request.url.path}", status_code=303)
    if wants_html and exc.status_code in (403, 404):
        from app.templating import render

        # status_code must be forwarded: an error page served as 200 is invisible to
        # monitoring and tells crawlers the missing page exists.
        return render(
            request,
            "error.html",
            status_code=exc.status_code,
            code=exc.status_code,
            detail=exc.detail,
        )
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
