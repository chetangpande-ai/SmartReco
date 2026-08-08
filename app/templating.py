"""Jinja setup plus the context every page needs."""

import hashlib
from functools import lru_cache

from fastapi import Request
from fastapi.templating import Jinja2Templates

from app import taxonomy
from app.config import ROOT
from app.security import CSRF_FIELD

templates = Jinja2Templates(directory=str(ROOT / "app" / "templates"))

# Several courses in the catalogue genuinely cost nothing, and "$0" reads like a bug.
templates.env.filters["money"] = lambda cents: "Free" if not cents else f"${cents / 100:,.0f}"

# Track slugs are title-cased by CSS, which turns "ai-ml" into "Ai Ml". The taxonomy
# already holds a written name for every category it knows; the fallbacks cover the two
# pre-taxonomy slugs that still appear in older fixtures and seeded data.
_TRACK_LABELS = {"ai-ml": "AI & ML", "web-dev": "Web Dev", "security": "Security"}


def _track(slug: str) -> str:
    node = taxonomy.category(slug)
    if node is not None:
        return node.name
    return _TRACK_LABELS.get(slug, slug.replace("-", " "))


templates.env.filters["track"] = _track
templates.env.filters["skill"] = taxonomy.skill_name


# Letters whose *name* begins with a vowel sound. Needed because the article depends on
# pronunciation, not spelling, and half these role names start with an acronym:
# "an AI Engineer" (ay-eye) but "a QA Engineer" (kyoo-ay) and "a UX Designer" (you-ex).
# A plain vowel check gets two of those three wrong.
_VOWEL_SOUNDING_LETTERS = set("AEFHILMNORSX")


def _article(phrase: str) -> str:
    word = (phrase or "").split(" ")[0].strip("/")
    if not word:
        return "a"
    if word.isupper():  # an acronym, read letter by letter
        return "an" if word[0] in _VOWEL_SOUNDING_LETTERS else "a"
    return "an" if word[0].upper() in "AEIOU" else "a"


templates.env.filters["article"] = _article


# One deliberate base hue per category, spread evenly around the wheel — enough
# separation to read as distinct at the pastel saturation/lightness the tiles render at.
# Previously the hue was hashed straight off the product slug: it produced a genuinely
# random colour per item, so a page of thirty-five tiles read as confetti instead of a
# system. Tying colour to *category* instead means colour becomes a second way to
# recognise a track at a glance, the same job the glyph already does.
#
# Derived from the taxonomy's own ordering rather than hand-assigned, because there are
# twenty-one categories now and a hand-written table would drift the moment one is added.
_LEGACY_HUE = {"web-dev": 62, "security": 188, "product": 272, "career": 314}
_DEFAULT_HUE = 250


@lru_cache(maxsize=1)
def _category_hues() -> dict[str, int]:
    slugs = [c.slug for c in taxonomy.categories()]
    step = 360 / max(len(slugs), 1)
    # The golden-ratio stride keeps neighbours in the list far apart on the wheel, so
    # adjacent categories in the nav are never adjacent in colour.
    return {slug: int((i * step * 0.618034 * len(slugs)) % 360) for i, slug in enumerate(slugs)}


def _hue(category: str, slug: str = "") -> int:
    """Category's base hue, nudged by a small slug-keyed jitter (±8°) so items in the
    same track are still individually distinguishable rather than perfectly identical."""
    base = _category_hues().get(category) or _LEGACY_HUE.get(category, _DEFAULT_HUE)
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
