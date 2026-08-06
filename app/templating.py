"""Jinja setup plus the context every page needs."""

import hashlib

from fastapi import Request
from fastapi.templating import Jinja2Templates

from app.config import ROOT
from app.security import CSRF_FIELD

templates = Jinja2Templates(directory=str(ROOT / "app" / "templates"))

# Several courses in the catalogue genuinely cost nothing, and "$0" reads like a bug.
templates.env.filters["money"] = lambda cents: "Free" if not cents else f"${cents / 100:,.0f}"

# Track slugs are title-cased by CSS, which turns "ai-ml" into "Ai Ml".
_TRACK_LABELS = {"ai-ml": "AI & ML", "web-dev": "Web Dev"}
templates.env.filters["track"] = lambda slug: _TRACK_LABELS.get(slug, slug.replace("-", " "))


# One deliberate base hue per category, 42° apart around the wheel — enough separation
# to read as distinct at the pastel saturation/lightness the tiles render at. Previously
# the hue was hashed straight off the product slug: it produced a genuinely random
# colour per item, so a page of thirty-five tiles read as confetti instead of a system.
# Tying colour to *category* instead means colour becomes a second way to recognise a
# track at a glance, the same job the glyph already does.
_CATEGORY_HUE = {
    "ai-ml": 20, "web-dev": 62, "data": 104, "cloud": 146,
    "security": 188, "design": 230, "product": 272, "career": 314,
}
_DEFAULT_HUE = 250


def _hue(category: str, slug: str = "") -> int:
    """Category's base hue, nudged by a small slug-keyed jitter (±8°) so items in the
    same track are still individually distinguishable rather than perfectly identical."""
    base = _CATEGORY_HUE.get(category, _DEFAULT_HUE)
    if not slug:
        return base
    jitter = int(hashlib.sha1(slug.encode()).hexdigest()[:4], 16) % 17 - 8
    return (base + jitter) % 360


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
