"""The AI Career Advisor: from where someone is to the role they named.

The interesting claim this makes is not "here are some popular courses". It is:

    these specific skills you already have transfer, these specific ones you are
    missing, here is the order to close them in, and here is the course for each

Everything above is computed, not generated. `analyse()` is deterministic: a set
difference against the target role's requirements, then a walk down the career path
resolving each gap to a real course whose prerequisites are already satisfied by the
step before it. That last part is the whole reason the output is worth reading — it is
what stops a tester with fifteen years behind them being handed an introduction to
programming, and it is why the model is not asked to pick the courses.

The model writes the *narrative* around those facts, through `mesh` like every other
call in this codebase, and there is a template fallback that says the same thing less
warmly when the LLM is off. A plan is never blocked on a model being available.
"""

import logging
from dataclasses import dataclass, field

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app import taxonomy
from app.agent import prompts
from app.models import CareerPlan, CareerProfile, Enrollment, Product, utcnow
from app.services import catalog
from app.services.mesh import MeshError, mesh

log = logging.getLogger(__name__)

TIERS = ("beginner", "intermediate", "advanced")

# Program types that satisfy each non-skill stage of the roadmap. Data, not `if` chains
# in the stage builder, so adding "Hackathon" to the project stage is a one-line edit.
STAGE_FORMATS = {
    "projects": ("Project", "Capstone Project", "Hands-on Lab", "Hackathon"),
    "certification": ("Certification Preparation", "Professional Certificate"),
    "interview": ("Interview Preparation", "Mock Interview"),
}

# How many years of experience puts someone past introductory material. Deliberately
# generous at the bottom: the cost of sending a senior person to a beginner course is a
# wasted week, and the cost of the reverse is that they bounce off and learn nothing.
_SENIORITY = ((2, "beginner"), (6, "intermediate"))

# Program types that are a stage of the plan, never the way you first learn a skill.
# An interview-prep course lists Python among its skills because its problems are
# written in it — routing someone there to *acquire* Python is how a plan ends up
# opening with "Grokking the Coding Interview" for a Java developer. These courses have
# their own stage further down the roadmap, which is where they belong.
NON_TEACHING_FORMATS = frozenset(("Interview Preparation", "Mock Interview", "Assessment"))


@dataclass
class Gap:
    skill: str
    name: str
    course: Product | None = None
    alternatives: list[Product] = field(default_factory=list)


@dataclass
class Analysis:
    role: taxonomy.Role | None
    path: taxonomy.Path | None
    required: list[str] = field(default_factory=list)  # what the target asks for
    have: list[str] = field(default_factory=list)  # of those, already held
    gaps: list[Gap] = field(default_factory=list)
    transferable: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    stages: list[dict] = field(default_factory=list)

    @property
    def title(self) -> str:
        if self.role:
            return self.role.name
        return self.path.name if self.path else ""

    @property
    def readiness(self) -> float:
        """Share of the target's skills already held. Shown as a percentage, so a target
        with no listed skills has to be 0 rather than a division by zero."""
        return len(self.have) / len(self.required) if self.required else 0.0

    @property
    def gap_slugs(self) -> list[str]:
        return [g.skill for g in self.gaps]

    @property
    def courses(self) -> list[Product]:
        seen, out = set(), []
        for gap in self.gaps:
            if gap.course is not None and gap.course.id not in seen:
                seen.add(gap.course.id)
                out.append(gap.course)
        return out


def preferred_tier(profile: CareerProfile) -> str:
    """Where to start someone, from what they told us about themselves."""
    if profile.level and profile.level.lower() in TIERS:
        return profile.level.lower()
    for years, tier in _SENIORITY:
        if profile.years_experience < years:
            return tier
    return "advanced"


def held_skills(db: Session, profile: CareerProfile) -> list[str]:
    """What the learner brings: what they said, what they finished here, and what both
    of those imply.

    Completed courses count because they are the one claim on this page we can actually
    verify, and it makes the roadmap move as someone works through it.

    The implication pass is what stops the advisor insulting people. Someone who says
    "Selenium, API testing" has been testing for a living; a literal set difference
    against a role asking for "Testing" would put it in their gap list.
    """
    have = list(profile.skills or [])
    finished = db.scalars(
        select(Product)
        .join(Enrollment, Enrollment.product_id == Product.id)
        .where(Enrollment.user_id == profile.user_id, Enrollment.status == "completed")
    )
    for product in finished:
        have.extend(s for s in product.teaches if s not in have)
    return taxonomy.expand_skills(have)


def choose_path(profile: CareerProfile) -> taxonomy.Path | None:
    """The route to the target role, preferring a bridge from where they already are.

    `paths_for_role` already puts transitions first. This narrows further: if one of
    them starts from the job they actually do, that is the path — "QA Engineer → AI
    Engineer" is a materially different plan from "Become an AI Engineer", and picking
    the generic one for someone who told us they are a tester wastes their experience.
    """
    candidates = taxonomy.paths_for_role(profile.target_role)
    if not candidates:
        return None

    current = _match_role(profile.current_role)
    for path in candidates:
        if current and path.from_role == current:
            return path
    return candidates[0]


def _match_role(text: str) -> str:
    """Free-text job title -> a taxonomy role slug, or "" when we cannot tell."""
    key = (text or "").strip().lower()
    if not key:
        return ""
    for slug, role in taxonomy.roles().items():
        if key == slug or key == role.name.lower():
            return slug
    # Substring match, longest name first so "AI Product Manager" beats "Product
    # Manager" on a title that contains both.
    for slug, role in sorted(
        taxonomy.roles().items(), key=lambda kv: -len(kv[1].name)
    ):
        if role.name.lower() in key:
            return slug
    return ""


def _rank(course: Product, skill: str, held: set[str], target_tier: int) -> tuple:
    """Sort key for the courses that teach one gap skill. Lower is better.

    Unmet prerequisites dominate everything else on purpose: routing someone into a
    course they cannot follow is the one failure that makes the whole plan useless, and
    it is exactly what a rating-first ranking would do — the best-rated RAG course in
    the catalogue is also the one that assumes you already know RAG.

    `focus` is where the skill sits in the course's own list, which is the difference
    between a course *about* Linux and a penetration-testing course that happens to
    cover some. Rating alone picks the second one, because it is a better course — just
    not a better answer to this question.
    """
    teaches = course.teaches
    unmet = sum(1 for r in course.requires if r not in held)
    tier_gap = abs(TIERS.index(course.tier) - target_tier)
    focus = teaches.index(skill) if skill in teaches else len(teaches)
    return (unmet, tier_gap, focus, -course.rating, -course.reviews)


def _target_tier(skill: str, held: set[str], seniority: int) -> int:
    """How advanced a course to aim at *for this particular skill*.

    Seniority is not global. Ten years of testing says nothing about how someone should
    meet machine learning for the first time, and applying it everywhere is how a plan
    ends up opening with a sixty-hour advanced course on a subject the learner has never
    seen. So seniority only applies where they are already working — judged by whether
    they hold any skill from the same part of the catalogue — and everywhere else the
    plan starts at the beginning, which is not an insult when the subject is genuinely new.
    """
    area = taxonomy.skill(skill)
    if area is None or not area.category:
        # Coarse role-level skills — "Cloud", "Analytics" — are named by roles and paths
        # but sit under no subcategory, so there is nothing to judge familiarity from.
        # Start accessible: the cost of an easy course is a short week, the cost of an
        # unfollowable one is the learner giving up on the plan.
        return 0
    familiar = {
        s.category for s in (taxonomy.skill(h) for h in held) if s and s.category
    }
    return seniority if area.category in familiar else 0


def preview(db: Session, path: taxonomy.Path) -> Analysis:
    """The same roadmap for someone we know nothing about.

    Backs the public career-path pages. A signed-out visitor sees the real plan — real
    courses, real order — with every step still open, which is a far better argument for
    signing in than a marketing page describing what the plan would contain.
    """
    # Every field set explicitly: this row is never flushed, so the column defaults that
    # would normally fill them in never run, and `preferred_tier` would be comparing
    # None to an int.
    anonymous = CareerProfile(
        user_id=0,
        target_role=path.to_role or path.slug,
        current_role=taxonomy.role(path.from_role).name if path.from_role else "",
        years_experience=0,
        skills=[],
        extra_skills=[],
        level="",
    )
    return analyse(db, anonymous, path=path)


def analyse(
    db: Session, profile: CareerProfile, path: taxonomy.Path | None = None
) -> Analysis:
    path = path or choose_path(profile)
    role = taxonomy.role(profile.target_role)
    if role is None and path is None:
        return Analysis(role=None, path=None, unknown=list(profile.extra_skills or []))

    # A couple of paths — Generative AI Engineer — describe a job the role list does not
    # name. Their own steps are then the requirement, which is the honest reading: the
    # path is what someone wrote down about becoming that thing.
    required = list(role.skills) if role else list(path.skills)

    have_all = held_skills(db, profile) if profile.user_id else []
    held = set(have_all)

    have = [s for s in required if s in held]
    missing = [s for s in required if s not in held]

    # The path is the ordering. Its steps are a teaching sequence someone thought about;
    # the role's skill list is an unordered job description. Path steps the learner
    # already has drop out, and role requirements the path never mentions are appended
    # so nothing silently vanishes from the plan.
    ordered = [s for s in (path.skills if path else []) if s not in held]
    ordered += [s for s in missing if s not in ordered]

    courses_for = catalog.by_skill(db, ordered)
    seniority = TIERS.index(preferred_tier(profile))

    gaps: list[Gap] = []
    covered_by: dict[str, Product] = {}
    for slug in ordered:
        gap = Gap(skill=slug, name=taxonomy.skill_name(slug))
        gaps.append(gap)

        if slug in held:
            # A course scheduled earlier in this loop already teaches it. Still a real
            # gap — it belongs in "skills to learn" — but it does not need a second
            # course, and the plan should say which one covers it.
            gap.course = covered_by.get(slug)
            continue

        tier = _target_tier(slug, held, seniority)
        teaching = [
            c for c in courses_for.get(slug, []) if c.format not in NON_TEACHING_FORMATS
        ]
        options = sorted(teaching, key=lambda c: _rank(c, slug, held, tier))
        if not options:
            continue
        gap.course, gap.alternatives = options[0], options[1:3]
        # Everything the chosen course teaches counts as held from here on, so each
        # later step is ranked against a learner who has already taken it. Without this
        # the plan keeps re-recommending the prerequisite it just scheduled.
        for taught in taxonomy.expand_skills(gap.course.teaches):
            held.add(taught)
            covered_by.setdefault(taught, gap.course)

    analysis = Analysis(
        role=role,
        path=path,
        required=required,
        have=have,
        gaps=gaps,
        transferable=[s for s in have_all if s not in required][:12],
        unknown=list(profile.extra_skills or []),
    )
    analysis.stages = _stages(db, profile, analysis)
    return analysis


def _stages(db: Session, profile: CareerProfile, a: Analysis) -> list[dict]:
    """The eight roadmap stages, as plain JSON so a stored plan stays renderable.

    The list key is `entries`, not `items`: these dicts are rendered by Jinja, where
    `stage.items` resolves to the dict *method* and silently renders nothing.
    """
    role_skills = list(a.required)
    gap_slugs = a.gap_slugs

    def course_item(p: Product, note: str = "") -> dict:
        return {
            "kind": "course",
            "label": p.title,
            "detail": note or f"{p.tier} · {p.duration_hours}h · {p.brand}",
            "href": f"/products/{p.slug}",
        }

    def skill_items(slugs, note: str) -> list[dict]:
        return [
            {"kind": "skill", "label": taxonomy.skill_name(s), "detail": note,
             "href": f"/explore/skills/{s}", "skill": s}
            for s in slugs
        ]

    current = skill_items(a.have, "counts toward the role")
    current += skill_items(a.transferable[:6], "transfers")
    current += [
        {"kind": "note", "label": name, "detail": "not in our catalogue yet", "href": ""}
        for name in a.unknown[:4]
    ]

    projects = catalog.with_format(db, STAGE_FORMATS["projects"], gap_slugs or role_skills, 4)
    certs = catalog.with_format(db, STAGE_FORMATS["certification"], role_skills, 3)
    interview = catalog.with_format(db, STAGE_FORMATS["interview"], role_skills, 3)

    # Assessment is not a separate product type in this catalogue — it is a property of
    # the courses already on the plan. Reporting it from the real course rows keeps the
    # stage honest instead of inventing an exam nobody can sit.
    assessed = [c for c in a.courses if c.assessments or c.labs]

    filled = {
        "current": current,
        "skills": skill_items(gap_slugs, "to learn"),
        "courses": _course_sequence(a),
        "projects": [course_item(p) for p in projects],
        "assessment": [
            {"kind": "assessment", "label": c.title,
             "detail": f"{c.assessments} assessments · {c.labs} labs", "href": f"/products/{c.slug}"}
            for c in assessed[:4]
        ],
        "certification": [course_item(p) for p in certs],
        "interview": [course_item(p) for p in interview],
        "target": [
            {"kind": "role", "label": a.title, "href": "",
             "detail": f"{len(a.have)} of {len(role_skills)} required skills already held"}
        ],
    }

    return [
        {"key": key, "title": title, "subtitle": subtitle, "entries": filled.get(key, [])}
        for key, title, subtitle in taxonomy.ROADMAP_STAGES
    ]


def _course_sequence(a: Analysis) -> list[dict]:
    """The courses stage: one entry per course, in plan order, naming every gap it
    closes. A course that covers four of the gaps is one step of the plan, not four."""
    order: list[int] = []
    grouped: dict[int, list[str]] = {}
    products: dict[int, Product] = {}
    for gap in a.gaps:
        if gap.course is None:
            continue
        if gap.course.id not in grouped:
            grouped[gap.course.id] = []
            products[gap.course.id] = gap.course
            order.append(gap.course.id)
        grouped[gap.course.id].append(gap.name)

    items = []
    for step, pid in enumerate(order, start=1):
        p = products[pid]
        items.append(
            {
                "kind": "course",
                "step": step,
                "label": p.title,
                "detail": f"{', '.join(grouped[pid])} · {p.tier} · {p.duration_hours}h",
                "href": f"/products/{p.slug}",
            }
        )
    return items


# ---- narrative ------------------------------------------------------------------


def _fallback_copy(profile: CareerProfile, a: Analysis) -> tuple[str, str]:
    """What the plan says when the LLM is off. Same facts, plainer voice.

    Not an error path — the offline demo and the whole test suite run through here, so
    it has to read like something a person would ship.
    """
    role_name = a.title or "your target role"
    current = profile.current_role or "where you are"
    first = a.gaps[0] if a.gaps else None

    headline = f"From {current} to {role_name}"
    if not a.gaps:
        return headline, (
            f"You already hold every skill we track for {role_name}. The useful next move "
            f"is depth and evidence: pick a project, then the interview preparation."
        )

    missing = ", ".join(g.name for g in a.gaps[:4])
    if len(a.gaps) > 4:
        missing += f" and {len(a.gaps) - 4} more"

    # Three different openings, because the honest sentence is different in each case.
    # Reporting "0% of what the role asks for" to someone with ten years of experience
    # is technically true and reads as a dismissal — when none of their skills are on
    # the role's list but they clearly have a craft, say that instead.
    if a.have:
        held = ", ".join(taxonomy.skill_name(s) for s in a.have[:3])
        opener = (
            f"{held} already count toward {role_name} — that is "
            f"{a.readiness * 100:.0f}% of what the role asks for. "
        )
    elif a.transferable:
        carried = ", ".join(taxonomy.skill_name(s) for s in a.transferable[:3])
        opener = (
            f"None of {role_name}'s listed skills are ones you have yet, but {carried} "
            f"is not nothing: this is a move sideways from a craft you already have, "
            f"not a restart. "
        )
    else:
        opener = f"{role_name} is a fresh start rather than a sidestep. "

    step = f"Start with {first.course.title}. " if first and first.course else ""
    return headline, (
        f"{opener}The gap is {missing}. {step}"
        f"The plan below closes them in an order where each course only assumes what "
        f"the one before it taught."
    ).strip()


def _llm_copy(profile: CareerProfile, a: Analysis) -> tuple[str, str, str]:
    """(headline, narrative, model). Raises MeshError upward — the caller falls back."""
    data, result = mesh.chat_json(
        [
            {"role": "system", "content": prompts.ADVISOR_SYSTEM},
            {"role": "user", "content": prompts.advisor_user_prompt(profile, a)},
        ],
        max_tokens=500,
        temperature=0.6,
    )
    headline = str(data.get("headline", ""))[:200]
    narrative = str(data.get("narrative", ""))
    if not headline or not narrative:
        raise MeshError("advisor returned no copy")
    return headline, narrative, result.model


def generate(db: Session, user_id: int) -> CareerPlan | None:
    """Analyse, write the copy, store the plan as the learner's current one."""
    profile = db.get(CareerProfile, user_id)
    if profile is None or not profile.target_role:
        return None

    analysis = analyse(db, profile)
    if analysis.role is None:
        log.warning("no such target role", extra={"role": profile.target_role})
        return None

    strategy, model = "deterministic", ""
    try:
        headline, narrative, model = _llm_copy(profile, analysis)
        strategy = "agentic"
    except MeshError as exc:
        # Budget exhausted, breaker open, or LLM_ENABLED=false. The plan is already
        # complete without it — only the prose changes.
        log.info("advisor copy fell back to template", extra={"reason": str(exc)[:120]})
        headline, narrative = _fallback_copy(profile, analysis)

    db.execute(
        update(CareerPlan)
        .where(CareerPlan.user_id == user_id, CareerPlan.is_current.is_(True))
        .values(is_current=False)
    )
    plan = CareerPlan(
        user_id=user_id,
        target_role=profile.target_role,
        path_slug=analysis.path.slug if analysis.path else "",
        headline=headline,
        narrative=narrative,
        have=list(analysis.have),
        gaps=analysis.gap_slugs,
        stages=analysis.stages,
        strategy=strategy,
        model=model,
        readiness=round(analysis.readiness, 4),
        is_current=True,
        created_at=utcnow(),
    )
    db.add(plan)
    db.flush()

    log.info(
        "career plan generated",
        extra={
            "user_id": user_id, "role": profile.target_role, "path": plan.path_slug,
            "have": len(analysis.have), "gaps": len(analysis.gaps),
            "readiness": plan.readiness, "strategy": strategy,
        },
    )
    return plan


def course_fit(db: Session, product: Product, user_id: int | None) -> dict:
    """"Is this course right for me?" — {verdict, answer, missing}.

    The prerequisite check is computed and handed to the model as settled fact rather
    than asked as a question. A model guessing at whether someone is ready is exactly
    the sort of confident wrong answer that costs a learner a wasted month, and the
    skill graph already knows.
    """
    profile = db.get(CareerProfile, user_id) if user_id else None
    held = set(held_skills(db, profile)) if profile else set()
    missing = [s for s in product.requires if s not in held]

    verdict = "good fit"
    if missing and profile:
        verdict = "not yet"
    elif profile and product.tier == "beginner" and preferred_tier(profile) == "advanced":
        # Only when they are already working in this area: an advanced tester meeting
        # machine learning for the first time genuinely does want the beginner course.
        area = taxonomy.category(product.category)
        familiar = {taxonomy.skill(h).category for h in held if taxonomy.skill(h)}
        if area and area.slug in familiar:
            verdict = "too basic"

    try:
        data, _ = mesh.chat_json(
            [
                {"role": "system", "content": prompts.FIT_SYSTEM},
                {"role": "user", "content": prompts.fit_user_prompt(product, profile, held, missing)},
            ],
            max_tokens=300,
            temperature=0.4,
        )
        if data.get("answer"):
            return {
                "verdict": str(data.get("verdict", verdict))[:24],
                "answer": str(data["answer"]),
                "missing": missing,
            }
    except MeshError as exc:
        log.info("fit check fell back to template", extra={"reason": str(exc)[:120]})

    return {"verdict": verdict, "answer": _fit_fallback(product, profile, missing), "missing": missing}


def _fit_fallback(product: Product, profile: CareerProfile | None, missing: list[str]) -> str:
    if profile is None:
        return (
            f"Tell the AI Career Advisor what you already know and we can answer this "
            f"properly. On the facts alone: {product.title} is {product.tier} level, "
            f"{product.duration_hours} hours, and assumes "
            f"{', '.join(taxonomy.skill_name(s) for s in product.requires) or 'no prior knowledge'}."
        )
    if missing:
        names = ", ".join(taxonomy.skill_name(s) for s in missing)
        return (
            f"Not yet. This course assumes {names}, which is not in your profile. "
            f"Close that first and it becomes the right next step."
        )
    return (
        f"Yes. You hold everything {product.title} assumes, and at {product.tier} level "
        f"over {product.duration_hours} hours it sits where you are now."
    )


def current_plan(db: Session, user_id: int) -> CareerPlan | None:
    return db.scalar(
        select(CareerPlan)
        .where(CareerPlan.user_id == user_id, CareerPlan.is_current.is_(True))
        .order_by(CareerPlan.created_at.desc())
    )
