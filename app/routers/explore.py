"""Explore Learning: the marketplace, and the taxonomy you browse it through.

`/courses` is the filtered grid. `/explore` and everything under it is the tree —
category, subcategory, topic, skill — which exists because a marketplace where the only
way in is a search box strands anyone who does not yet know the words. A learner who has
just decided to "get into AI" cannot search their way to `vector-databases`; they can
click their way there.

Every skill page is also a career page: the roles that need it are one join away, and
that link is what turns browsing into a plan.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app import taxonomy
from app.db import get_db
from app.services import catalog
from app.templating import render

log = logging.getLogger(__name__)
router = APIRouter(tags=["explore"])

PAGE_SIZE = 24

# The program types that get their own front door, in the order the nav lists them.
# Straight from the taxonomy's vocabulary, so a new program type needs no code here.
PROGRAM_SHELVES = (
    ("Professional Programs", ("Professional Certificate", "Career Program", "Executive Program")),
    ("Bootcamps", ("Bootcamp",)),
    ("Certifications", ("Certification Preparation", "Professional Certificate")),
    ("Projects", ("Project", "Capstone Project", "Hands-on Lab")),
    ("Workshops", ("Workshop", "Live Online Class", "Crash Course", "Micro-Course")),
    ("Free Courses", ("Free Course",)),
)


def _filters(request: Request) -> catalog.CourseFilters:
    """Query string -> filters. Unknown values are dropped rather than 400'd: a filter
    chip left in a bookmarked URL after a taxonomy edit should degrade to "not applied",
    not to an error page."""
    q = request.query_params
    return catalog.CourseFilters(
        q=q.get("q", "").strip()[:120],
        category=q.get("category", "") if taxonomy.category(q.get("category", "")) else "",
        subcategory=q.get("sub", "") if taxonomy.subcategory(q.get("sub", "")) else "",
        skill=q.get("skill", "") if taxonomy.skill(q.get("skill", "")) else "",
        role=q.get("role", "") if taxonomy.role(q.get("role", "")) else "",
        level=q.get("level", "") if q.get("level") in ("beginner", "intermediate", "advanced") else "",
        fmt=q.get("format", "") if q.get("format") in taxonomy.taxonomy().program_types else "",
        duration=q.get("duration", "") if q.get("duration") in catalog.DURATION_BUCKETS else "",
        price=q.get("price", "") if q.get("price") in ("free", "paid") else "",
        rating=q.get("rating", "") if q.get("rating") in ("4.0", "4.5") else "",
        certificate=q.get("certificate") == "1",
        sort=q.get("sort", catalog.DEFAULT_SORT),
    )


@router.get("/courses")
def marketplace(request: Request, page: int = 1, db: Session = Depends(get_db)):
    filters = _filters(request)
    page = max(1, page)
    courses, total = catalog.search(
        db, filters, limit=PAGE_SIZE, offset=(page - 1) * PAGE_SIZE
    )
    return render(
        request,
        "explore/courses.html",
        courses=courses,
        total=total,
        page=page,
        pages=max(1, -(-total // PAGE_SIZE)),
        filters=filters,
        tax=taxonomy.taxonomy(),
        duration_buckets=list(catalog.DURATION_BUCKETS),
        roles=sorted(taxonomy.roles().values(), key=lambda r: r.name),
        categories=catalog.categories(db),
        active_category=filters.category,
        query=filters.q,
    )


@router.get("/explore")
def explore(request: Request, db: Session = Depends(get_db)):
    counts = catalog.counts_by_category(db)
    by_format = catalog.counts_by_format(db)
    return render(
        request,
        "explore/index.html",
        tree=taxonomy.categories(),
        counts=counts,
        shelves=[
            (label, formats, sum(by_format.get(f, 0) for f in formats))
            for label, formats in PROGRAM_SHELVES
        ],
        roles=sorted(taxonomy.roles().values(), key=lambda r: r.name),
        categories=catalog.categories(db),
    )


@router.get("/explore/categories/{slug}")
def category_page(request: Request, slug: str, db: Session = Depends(get_db)):
    node = taxonomy.category(slug)
    if node is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such category")

    courses, total = catalog.search(db, catalog.CourseFilters(category=slug), limit=12)
    return render(
        request,
        "explore/category.html",
        node=node,
        courses=courses,
        total=total,
        categories=catalog.categories(db),
        active_category=slug,
    )


@router.get("/explore/skills/{slug}")
def skill_page(request: Request, slug: str, db: Session = Depends(get_db)):
    skill = taxonomy.skill(slug)
    if skill is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such skill")

    teaching, total = catalog.search(db, catalog.CourseFilters(skill=slug), limit=12)
    needs = [r for r in taxonomy.roles().values() if slug in r.skills]
    paths = [p for p in taxonomy.paths().values() if slug in p.skills]

    return render(
        request,
        "explore/skill.html",
        skill=skill,
        subcategory=taxonomy.subcategory(skill.subcategory),
        courses=teaching,
        total=total,
        # What this skill unlocks. A skill page that only lists courses is a dead end;
        # the roles are the reason anyone would learn it.
        roles=needs,
        paths=paths,
        # Skills that imply this one are the deeper specialisms beneath it.
        specialisms=[
            taxonomy.skill(s)
            for s in taxonomy.taxonomy().skills
            if slug in taxonomy.implies(s)
        ][:12],
        categories=catalog.categories(db),
        active_category=skill.category,
    )
