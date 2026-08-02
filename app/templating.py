"""Jinja setup plus the context every page needs."""

import hashlib

from fastapi import Request
from fastapi.templating import Jinja2Templates

from app.config import ROOT
from app.security import CSRF_FIELD

templates = Jinja2Templates(directory=str(ROOT / "app" / "templates"))
templates.env.filters["money"] = lambda cents: f"${cents / 100:,.0f}"


def _hue(text: str) -> int:
    """Stable hue per product, so a given item always renders the same colour.

    Product photography is the one thing a demo catalogue cannot honestly have — real
    imagery belongs to the manufacturers. Generated tiles keyed off the slug look
    deliberate rather than broken, and cost no bytes and no third-party requests.
    """
    return int(hashlib.sha1(text.encode()).hexdigest()[:6], 16) % 360


templates.env.filters["hue"] = _hue


def render(request: Request, template: str, status_code: int = 200, **context):
    return templates.TemplateResponse(
        request=request,
        name=template,
        status_code=status_code,
        context={
            "user": getattr(request.state, "user", None),
            "csrf_token": getattr(request.state, "csrf", ""),
            "csrf_field": CSRF_FIELD,
            **context,
        },
    )
