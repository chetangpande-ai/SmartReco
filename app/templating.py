"""Jinja setup plus the context every page needs."""

from fastapi import Request
from fastapi.templating import Jinja2Templates

from app.config import ROOT
from app.security import CSRF_FIELD

templates = Jinja2Templates(directory=str(ROOT / "app" / "templates"))
templates.env.filters["money"] = lambda cents: f"${cents / 100:,.0f}"


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
