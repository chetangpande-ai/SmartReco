"""Password hashing, session JWTs, and CSRF. Deliberately small.

Sessions are a signed JWT in an HttpOnly cookie: no server-side session table to keep
consistent, and JS can never read the token, so an XSS bug cannot exfiltrate it.
"""

import hmac
import secrets
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from app.config import settings

SESSION_COOKIE = "sr_session"
CSRF_COOKIE = "sr_csrf"
CSRF_FIELD = "csrf_token"
SESSION_TTL = timedelta(days=7)

# bcrypt silently ignores everything past 72 bytes. Truncating explicitly means two
# passwords sharing a 72-byte prefix don't become the same credential by accident.
_BCRYPT_MAX_BYTES = 72


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode()[:_BCRYPT_MAX_BYTES], bcrypt.gensalt(rounds=12)).decode()


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode()[:_BCRYPT_MAX_BYTES], password_hash.encode())


def create_session_token(user_id: int, role: str) -> str:
    now = datetime.now(timezone.utc)
    return jwt.encode(
        {"sub": str(user_id), "role": role, "iat": now, "exp": now + SESSION_TTL},
        settings.secret_key,
        algorithm="HS256",
    )


def decode_session_token(token: str) -> dict | None:
    """None on anything invalid — expired, tampered, malformed. Callers treat it as
    'not logged in', which is the only sensible response."""
    try:
        return jwt.decode(token, settings.secret_key, algorithms=["HS256"])
    except jwt.PyJWTError:
        return None


def new_csrf_token() -> str:
    return secrets.token_urlsafe(32)


def csrf_ok(cookie_value: str | None, form_value: str | None) -> bool:
    """Double-submit: an attacker's site can trigger a request but cannot read our
    cookie to populate the matching form field."""
    if not cookie_value or not form_value:
        return False
    return hmac.compare_digest(cookie_value, form_value)


def cookie_kwargs(max_age: int | None = None) -> dict:
    return {
        "httponly": True,
        "samesite": "lax",
        "secure": settings.app_env == "production",
        "path": "/",
        "max_age": max_age,
    }
