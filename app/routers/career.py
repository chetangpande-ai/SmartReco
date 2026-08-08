"""The career layer: the AI advisor, and the roadmaps it produces.

Generation is synchronous here, unlike the recommendation agent. The two look similar
and are not: a recommendation is something we decided to show someone who came to
browse, so it belongs on a background task behind a poller. A career plan is the thing
they filled in a form to get. Waiting a second for an answer you asked for is fine;
being redirected to a page that says "working on it" is not.

It is affordable because the plan is already computed before the model is involved —
one short completion for the narrative, with a template fallback underneath it.
"""

import logging

from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from app import taxonomy
from app.db import get_db
from app.deps import get_current_user, require_user, verify_csrf
from app.models import CareerProfile, User
from app.ratelimit import recommend_limiter
from app.services import advisor, catalog
from app.templating import render

log = logging.getLogger(__name__)
router = APIRouter(tags=["career"])


# The scenario the platform is built to answer, one click away. Server-side rather than
# a script because the CSP forbids inline JS, and a link is a better demo affordance
# anyway — it is shareable and it survives a page reload.
DEMO_PREFILL = {
    "current_role": "QA Engineer",
    "years": 10,
    "skills": "Java, Selenium, API testing",
    "technologies": "Jenkins, Postman, Git",
    "target": "ai-engineer",
    "interests": "agentic AI, evaluation",
    "hours": 8,
}

FEATURED_PATHS = ("qa-to-ai", "ai-engineer", "genai-engineer", "data-scientist", "cloud-architect")


@router.get("/career")
def advisor_page(
    request: Request,
    demo: int = 0,
    user: User | None = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """The intake form, and the plan if one has been generated."""
    profile = db.get(CareerProfile, user.id) if user else None
    plan = advisor.current_plan(db, user.id) if user else None
    target = taxonomy.role(plan.target_role) if plan else None
    return render(
        request,
        "career/advisor.html",
        profile=profile,
        # The form takes free text and the profile stores canonical slugs, so what goes
        # back into the box is the readable name, not `api-testing`.
        profile_skills=", ".join(
            taxonomy.skill_name(s) for s in (profile.skills if profile else [])
        ),
        prefill=DEMO_PREFILL if demo else None,
        plan=plan,
        target_name=target.name if target else "",
        roles=sorted(taxonomy.roles().values(), key=lambda r: r.name),
        levels=("beginner", "intermediate", "advanced"),
        featured_paths=[taxonomy.path(s) for s in FEATURED_PATHS if taxonomy.path(s)],
        categories=catalog.categories(db),
    )


@router.post("/career/advisor", dependencies=[Depends(verify_csrf)])
def build_plan(
    current_role: str = Form(""),
    years_experience: int = Form(0),
    skills: str = Form(""),
    technologies: str = Form(""),
    target_role: str = Form(""),
    interests: str = Form(""),
    weekly_hours: int = Form(0),
    level: str = Form(""),
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    if taxonomy.role(target_role) is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "choose a target role")
    if not recommend_limiter.allow(f"career:{user.id}"):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "please wait a moment")

    # Skills and technologies are two boxes on the form because that is how people think
    # about them, and one vocabulary underneath because the gap analysis does not care
    # which box "Docker" was typed into.
    known, unknown = taxonomy.resolve_skills(f"{skills},{technologies}")

    profile = db.get(CareerProfile, user.id) or CareerProfile(user_id=user.id)
    profile.current_role = current_role.strip()[:120]
    profile.years_experience = max(0, min(years_experience, 60))
    profile.skills = known
    profile.extra_skills = unknown
    profile.target_role = target_role
    profile.interests = interests.strip()[:300]
    profile.weekly_hours = max(0, min(weekly_hours, 80))
    profile.level = level if level in ("beginner", "intermediate", "advanced") else ""
    db.add(profile)
    db.flush()

    advisor.generate(db, user.id)
    db.commit()
    return RedirectResponse("/career#plan", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/careers")
def paths_index(
    request: Request,
    user: User | None = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    profile = db.get(CareerProfile, user.id) if user else None
    held = set(advisor.held_skills(db, profile)) if profile else set()

    # Readiness per path, so someone who has already told us about themselves sees which
    # routes they are closest to rather than an alphabetical list of aspirations.
    cards = []
    for path in taxonomy.paths().values():
        required = taxonomy.role(path.to_role).skills if path.to_role else path.skills
        have = [s for s in required if s in held]
        cards.append(
            {
                "path": path,
                "from_role": taxonomy.role(path.from_role) if path.from_role else None,
                "to_role": taxonomy.role(path.to_role) if path.to_role else None,
                "steps": len(path.skills),
                "have": len(have),
                "required": len(required),
                "readiness": len(have) / len(required) if required else 0.0,
            }
        )
    cards.sort(key=lambda c: (-c["readiness"], c["path"].name))

    return render(
        request,
        "career/paths.html",
        cards=cards,
        profile=profile,
        categories=catalog.categories(db),
    )


@router.get("/careers/{slug}")
def path_detail(
    request: Request,
    slug: str,
    user: User | None = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    path = taxonomy.path(slug)
    if path is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such career path")

    # Signed in with a stated background: the real, personalised plan. Otherwise the
    # same roadmap with every step still open.
    profile = db.get(CareerProfile, user.id) if user else None
    if profile and profile.skills:
        analysis = advisor.analyse(db, profile, path=path)
        personalised = True
    else:
        analysis = advisor.preview(db, path)
        personalised = False

    return render(
        request,
        "career/roadmap.html",
        path=path,
        analysis=analysis,
        personalised=personalised,
        profile=profile,
        from_role=taxonomy.role(path.from_role) if path.from_role else None,
        to_role=taxonomy.role(path.to_role) if path.to_role else None,
        categories=catalog.categories(db),
    )
