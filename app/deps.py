"""Shared FastAPI dependencies: who is calling, and are they allowed to."""

import uuid

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import User
from app.security import (
    CSRF_COOKIE,
    CSRF_FIELD,
    SESSION_COOKIE,
    csrf_ok,
    decode_session_token,
)

ANON_COOKIE = "sr_anon"


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User | None:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    payload = decode_session_token(token)
    if not payload:
        return None
    return db.get(User, int(payload["sub"]))


def require_user(user: User | None = Depends(get_current_user)) -> User:
    if user is None:
        # main.py turns this into a redirect for browser navigations and leaves it as
        # a 401 for fetch/XHR, so one dependency serves both the pages and the API.
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "login required")
    return user


def require_admin(user: User = Depends(require_user)) -> User:
    if not user.is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "admin only")
    return user


def get_anon_id(request: Request) -> str:
    """Stable id for logged-out visitors so their pre-signup browsing isn't lost."""
    return request.cookies.get(ANON_COOKIE) or ""


def new_anon_id() -> str:
    return uuid.uuid4().hex


async def verify_csrf(request: Request) -> None:
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return
    form = await request.form()
    if not csrf_ok(request.cookies.get(CSRF_COOKIE), form.get(CSRF_FIELD)):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "invalid CSRF token")
