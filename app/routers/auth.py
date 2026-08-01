"""Email + password auth. Kept deliberately small, as the brief asks."""

import logging

from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.deps import verify_csrf
from app.models import User, UserProfile
from app.ratelimit import auth_limiter
from app.schemas import RegisterIn
from app.security import (
    SESSION_COOKIE,
    cookie_kwargs,
    create_session_token,
    hash_password,
    verify_password,
)
from app.templating import render

log = logging.getLogger(__name__)
router = APIRouter(tags=["auth"])

SESSION_MAX_AGE = 60 * 60 * 24 * 7


def _safe_next(target: str | None) -> str:
    """Only same-site paths. An attacker-supplied absolute URL here is an open redirect."""
    if not target or not target.startswith("/") or target.startswith("//"):
        return "/"
    return target


def _login_response(user: User, next_url: str) -> RedirectResponse:
    response = RedirectResponse(_safe_next(next_url), status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(
        SESSION_COOKIE,
        create_session_token(user.id, user.role),
        **cookie_kwargs(SESSION_MAX_AGE),
    )
    return response


@router.get("/login")
def login_form(request: Request, next: str = "/"):
    if request.state.user:
        return RedirectResponse(_safe_next(next), status_code=303)
    return render(request, "login.html", next=next, error=None)


@router.post("/login", dependencies=[Depends(verify_csrf)])
def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    next: str = Form("/"),
    db: Session = Depends(get_db),
):
    if not auth_limiter.allow(f"login:{request.client.host}"):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "too many attempts")

    user = db.scalar(select(User).where(User.email == email.strip().lower()))
    if not user or not verify_password(password, user.password_hash):
        # One message for both cases: distinguishing them tells an attacker which
        # addresses are registered.
        log.warning("failed login", extra={"email": email[:64]})
        return render(
            request, "login.html", next=next, error="Incorrect email or password.", status_code=401
        )

    log.info("login", extra={"user_id": user.id})
    return _login_response(user, next)


@router.get("/register")
def register_form(request: Request):
    if request.state.user:
        return RedirectResponse("/", status_code=303)
    return render(request, "register.html", error=None)


@router.post("/register", dependencies=[Depends(verify_csrf)])
def register(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    name: str = Form(""),
    db: Session = Depends(get_db),
):
    if not auth_limiter.allow(f"register:{request.client.host}"):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "too many attempts")

    try:
        data = RegisterIn(email=email, password=password, name=name)
    except ValidationError as exc:
        first = exc.errors()[0]["msg"].removeprefix("Value error, ")
        return render(request, "register.html", error=first, status_code=400)

    exists = db.scalar(
        select(User.id).where(func.lower(User.email) == data.email.lower())
    )
    if exists:
        return render(
            request, "register.html", error="That email is already registered.", status_code=409
        )

    user = User(
        email=data.email.lower(),
        password_hash=hash_password(data.password),
        name=data.name or data.email.split("@")[0],
        role="user",
    )
    db.add(user)
    db.flush()
    db.add(UserProfile(user_id=user.id))
    db.commit()

    log.info("registered", extra={"user_id": user.id})
    return _login_response(user, "/")


@router.post("/logout", dependencies=[Depends(verify_csrf)])
def logout():
    response = RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response
