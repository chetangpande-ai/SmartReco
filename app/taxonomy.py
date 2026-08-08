"""What exists to be learned, and what jobs it adds up to.

Loaded once at import from `app/data/taxonomy.json` — a fixed vocabulary that ships
with the code, so it needs no table, no migration and no query.

    Category ─ Subcategory ─ Topic ─┐
                                    ├─ Skill ─┬─ Course        (course_skills, in SQL)
    CareerRole ─── requires ────────┘         ├─ CareerRole
    CareerPath ─── ordered steps ─────────────┘

One decision does most of the work here: **topics, role requirements and career-path
steps are all the same skill vocabulary.** The source JSON writes them as prose, so the
same skill turns up as "AI Evals" in one place and "AI Evaluation" in another, and
"AWS/Azure/GCP" arrives as a single string covering three. Every name is canonicalised
to a slug once, on load, and everything downstream compares slugs. That is what makes
skill-gap analysis a set difference instead of fuzzy string matching smeared across the
codebase — and what lets the advisor answer "what should this learner learn next?"
rather than "what is popular?".

Adding a category, a role or a career path means editing the JSON. No template changes,
no code changes, no migration.
"""

import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache

from app.config import ROOT

log = logging.getLogger(__name__)

SOURCE = ROOT / "app" / "data" / "taxonomy.json"

# Written as one string in the source, but genuinely three skills — a learner who knows
# AWS should match a Cloud Architect requirement. An explicit list rather than "split on
# slash", because CI/CD is one skill whose name happens to contain one.
_SPLIT = {
    "AWS/Azure/GCP": ("AWS", "Azure", "GCP"),
    "LangChain/LangGraph": ("LangChain", "LangGraph"),
    "Selenium/Playwright": ("Selenium", "Playwright"),
}

# The same skill under a second name. Without these, "AI Evals" (what the AI Engineer
# *role* requires) and "AI Evaluation" (what the AI Engineer *path* teaches) are two
# skills, and the gap analysis reports a hole the roadmap already fills. Keys are
# lowercased free text — user input is matched against them too, which is why the
# informal spellings people actually type are in here.
_ALIASES = {
    "ai evals": "ai-evaluation",
    "ai eval": "ai-evaluation",
    "evals": "ai-evaluation",
    "ml": "machine-learning",
    "ml fundamentals": "machine-learning",
    "machine learning fundamentals": "machine-learning",
    "dl": "deep-learning",
    "genai": "generative-ai",
    "gen ai": "generative-ai",
    "generative ai": "generative-ai",
    "llm": "llms",
    "large language models": "llms",
    "agentic ai": "ai-agents",
    "ai agent": "ai-agents",
    "agents": "ai-agents",
    "dsa": "data-structures-algorithms",
    "data structures": "data-structures-algorithms",
    "algorithms": "data-structures-algorithms",
    "js": "javascript",
    "ts": "typescript",
    "k8s": "kubernetes",
    "postgres": "sql",
    "postgresql": "sql",
    "mysql": "sql",
    "rest": "rest-apis",
    "rest api": "rest-apis",
    "api testing": "api-testing",
    "apis": "rest-apis",
    "automation testing": "automation",
    "test automation": "automation",
    "manual testing": "testing",
    "qa": "testing",
    "vector db": "vector-databases",
    "vector database": "vector-databases",
    "prompt engineering": "prompt-engineering",
    "cicd": "ci-cd",
    "ci cd": "ci-cd",
    "aws cloud": "aws",
    "gcp cloud": "gcp",
    "google cloud": "gcp",
    "product": "product-strategy",
    "programming": "programming-languages",
    # The Product Manager role lists "Discovery"; the catalogue tree calls the same
    # thing "Product Discovery". Two slugs would put a permanent phantom gap in every
    # PM roadmap.
    "discovery": "product-discovery",
}

# Having the skill on the left means you have the skills on the right.
#
# The taxonomy mixes granularities on purpose — roles ask for "Testing", learners say
# "Selenium" — and without this the gap analysis tells a tester with ten years behind
# them that they need to learn testing. That single output is the difference between the
# advisor looking intelligent and looking like a keyword matcher, so the implications are
# written out rather than inferred from the tree: being filed under Test Automation does
# not by itself mean a tool implies the discipline.
_IMPLIES = {
    "selenium": ("automation", "testing"),
    "playwright": ("automation", "testing"),
    "cypress": ("automation", "testing"),
    "api-testing": ("testing", "rest-apis"),
    "test-automation": ("automation", "testing"),
    "performance-testing": ("testing",),
    "load-testing": ("testing",),
    "jmeter": ("load-testing", "testing"),
    "agentic-qe": ("ai-testing", "quality-engineering", "testing"),
    "ai-testing": ("testing",),
    "quality-engineering": ("testing",),
    "langgraph": ("ai-agents",),
    "langchain": ("llms",),
    "rag": ("embeddings", "llms"),
    "llms": ("generative-ai", "llm-fundamentals"),
    "generative-ai": ("llm-fundamentals",),
    "fine-tuning": ("deep-learning",),
    "deep-learning": ("machine-learning",),
    "mlops": ("machine-learning", "model-deployment"),
    "kubernetes": ("containers", "docker"),
    "docker": ("containers",),
    "terraform": ("infrastructure-automation",),
    "aws": ("cloud", "cloud-fundamentals"),
    "azure": ("cloud", "cloud-fundamentals"),
    "gcp": ("cloud", "cloud-fundamentals"),
    "cloud-deployment": ("cloud",),
    "spark": ("big-data",),
    "databricks": ("spark",),
    "airflow": ("etl",),
    "react": ("javascript", "frontend"),
    "angular": ("javascript", "frontend"),
    "vue": ("javascript", "frontend"),
    "typescript": ("javascript",),
    "node-js": ("javascript", "backend"),
    "django": ("python", "backend"),
    "flask": ("python", "backend"),
    "fastapi": ("python", "backend", "rest-apis"),
    "spring-boot": ("java", "backend"),
    "power-bi": ("data-visualization", "visualization"),
    "tableau": ("data-visualization", "visualization"),
    "penetration-testing": ("security", "ethical-hacking"),
    "cloud-security": ("security", "cloud"),
    "secure-coding": ("security",),
    "product-discovery": ("product-fundamentals",),
    "product-strategy": ("product-fundamentals",),
    "ci-cd": ("git", "automation"),
}

# Path steps that are a stage of the journey rather than a skill to learn.
_STAGE_STEPS = {
    "projects": "project",
    "capstone project": "project",
    "case studies": "project",
    "product case studies": "project",
    "interview preparation": "interview",
    "certification": "certification",
    "assessment": "assessment",
}

# Transition paths name both ends in their display name only. Everywhere else a path's
# slug is its target role's slug.
_TRANSITIONS = {"qa-to-ai": ("qa-engineer", "ai-engineer")}

# The eight stages every career roadmap renders, in order. The source JSON supplies the
# skill sequence; these are the scaffolding the platform wraps around it, so they live
# here rather than in the template that draws them.
ROADMAP_STAGES = (
    ("current", "Where you are", "The skills you already bring"),
    ("skills", "Skills to learn", "The gap between here and the role"),
    ("courses", "Courses", "Real courses from the catalogue that teach them"),
    ("projects", "Projects", "Build something that proves it"),
    ("assessment", "Assessment", "Check the skill actually landed"),
    ("certification", "Certification", "Make it legible to a recruiter"),
    ("interview", "Interview preparation", "Rehearse the conversation"),
    ("target", "Target role", "Where you wanted to be"),
)


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")


@dataclass(frozen=True)
class Skill:
    slug: str
    name: str
    subcategory: str = ""  # subcategory *key*, see Subcategory.key

    @property
    def category(self) -> str:
        sub = subcategory(self.subcategory)
        return sub.category if sub else ""


@dataclass(frozen=True)
class Subcategory:
    slug: str
    name: str
    category: str
    topics: tuple[str, ...]  # skill slugs

    @property
    def key(self) -> str:
        """Subcategory ids are only unique within their category — `tools` is both a
        Project Management and a UI/UX subcategory. Everything that stores or looks one
        up uses this, so the two never collapse into each other."""
        return f"{self.category}/{self.slug}"


@dataclass(frozen=True)
class Category:
    slug: str
    name: str
    subcategories: tuple[Subcategory, ...]

    @property
    def topics(self) -> tuple[str, ...]:
        return tuple(t for s in self.subcategories for t in s.topics)


@dataclass(frozen=True)
class Role:
    slug: str
    name: str
    skills: tuple[str, ...]  # skill slugs, in the order the source lists them


@dataclass(frozen=True)
class Step:
    position: int
    kind: str  # skill | project | assessment | certification | interview
    label: str
    skill: str = ""  # slug, empty for stage steps


@dataclass(frozen=True)
class Path:
    slug: str
    name: str
    steps: tuple[Step, ...]
    from_role: str = ""  # slug, set only on transition paths
    to_role: str = ""  # slug, empty when the path has no matching role

    @property
    def skills(self) -> tuple[str, ...]:
        return tuple(s.skill for s in self.steps if s.kind == "skill")


class _Taxonomy:
    """Everything from the JSON, indexed. Built once; see `taxonomy()` below."""

    def __init__(self, raw: dict):
        self.raw = raw
        self.skills: dict[str, Skill] = {}
        self._alias_index: dict[str, str] = {}

        self.categories = tuple(self._category(c) for c in raw["categories"])
        self.subcategories = {s.key: s for c in self.categories for s in c.subcategories}
        self.category_by_slug = {c.slug: c for c in self.categories}
        self.roles = {r.slug: r for r in (self._role(r) for r in raw["career_roles"])}
        self.paths = {p.slug: p for p in (self._path(p) for p in raw["career_paths"])}

        self.program_types = tuple(raw["program_types"])
        self.levels = tuple(raw["course_levels"])
        self.delivery_modes = tuple(raw["delivery_modes"])
        self.learning_features = tuple(raw["learning_features"])
        self.filters = raw["filters"]
        self.navigation = raw["top_navigation"]

        log.info(
            "taxonomy loaded",
            extra={
                "categories": len(self.categories),
                "skills": len(self.skills),
                "roles": len(self.roles),
                "paths": len(self.paths),
            },
        )

    # ---- loading -----------------------------------------------------------

    def _category(self, node: dict) -> Category:
        subs = []
        for sub in node["subcategories"]:
            key = f"{node['id']}/{sub['id']}"
            topics = tuple(
                slug for name in sub["topics"] for slug in self._register(name, subcategory=key)
            )
            subs.append(
                Subcategory(
                    slug=sub["id"], name=sub["name"], category=node["id"], topics=topics
                )
            )
        return Category(slug=node["id"], name=node["name"], subcategories=tuple(subs))

    def _role(self, node: dict) -> Role:
        skills = tuple(slug for name in node["skills"] for slug in self._register(name))
        return Role(slug=node["id"], name=node["name"], skills=skills)

    def _path(self, node: dict) -> Path:
        steps: list[Step] = []
        for raw_label in node["steps"]:
            stage = _STAGE_STEPS.get(raw_label.strip().lower())
            if stage:
                steps.append(Step(len(steps), stage, raw_label))
                continue
            for slug in self._register(raw_label):
                steps.append(Step(len(steps), "skill", self.skills[slug].name, slug))

        from_role, to_role = _TRANSITIONS.get(node["id"], ("", node["id"]))
        return Path(
            slug=node["id"],
            name=node["name"],
            steps=tuple(steps),
            from_role=from_role,
            to_role=to_role if to_role in {r["id"] for r in self.raw["career_roles"]} else "",
        )

    def _register(self, name: str, subcategory: str = "") -> tuple[str, ...]:
        """Canonicalise one source name into one or more skill slugs, recording it.

        Returns a tuple because a few source names cover several skills — see `_SPLIT`.
        """
        parts = _SPLIT.get(name.strip(), (name,))
        slugs = []
        for part in parts:
            slug = _ALIASES.get(part.strip().lower()) or slugify(part)
            existing = self.skills.get(slug)
            # First registration wins the display name; a later one only fills in the
            # subcategory, so a skill named by a role before any topic mentions it still
            # ends up filed under the right part of the catalogue.
            if existing is None:
                self.skills[slug] = Skill(slug, part.strip(), subcategory)
            elif subcategory and not existing.subcategory:
                self.skills[slug] = Skill(slug, existing.name, subcategory)
            self._alias_index[part.strip().lower()] = slug
            slugs.append(slug)
        return tuple(slugs)

    # ---- lookup ------------------------------------------------------------

    def resolve(self, text: str) -> str:
        """Free text a learner typed -> a canonical skill slug, or "" if we don't know it.

        Deliberately conservative. A wrong match tells someone they already have a skill
        they do not, which is worse for them than an honest "not recognised" — the
        advisor surfaces unmatched entries rather than dropping them.
        """
        key = text.strip().lower()
        if not key:
            return ""
        if key in _ALIASES:
            return _ALIASES[key]
        if key in self._alias_index:
            return self._alias_index[key]
        slug = slugify(key)
        return slug if slug in self.skills else ""

    def name_of(self, slug: str) -> str:
        skill = self.skills.get(slug)
        return skill.name if skill else slug.replace("-", " ").title()


@lru_cache(maxsize=1)
def taxonomy() -> _Taxonomy:
    with SOURCE.open(encoding="utf-8") as fh:
        return _Taxonomy(json.load(fh))


# Thin module-level accessors. Every caller goes through these rather than reaching into
# the object, so the loading strategy stays swappable without a codebase-wide edit.


def categories() -> tuple[Category, ...]:
    return taxonomy().categories


def category(slug: str) -> Category | None:
    return taxonomy().category_by_slug.get(slug)


def subcategory(slug: str) -> Subcategory | None:
    return taxonomy().subcategories.get(slug)


def roles() -> dict[str, Role]:
    return taxonomy().roles


def role(slug: str) -> Role | None:
    return taxonomy().roles.get(slug)


def paths() -> dict[str, Path]:
    return taxonomy().paths


def path(slug: str) -> Path | None:
    return taxonomy().paths.get(slug)


def paths_for_role(role_slug: str) -> list[Path]:
    """Every path that ends at this role, transitions first — someone already working in
    an adjacent job wants the bridge, not the from-scratch route."""
    matches = [p for p in taxonomy().paths.values() if p.to_role == role_slug]
    return sorted(matches, key=lambda p: (not p.from_role, p.name))


def skill(slug: str) -> Skill | None:
    return taxonomy().skills.get(slug)


def skill_name(slug: str) -> str:
    return taxonomy().name_of(slug)


def resolve_skill(text: str) -> str:
    return taxonomy().resolve(text)


def expand_skills(slugs) -> list[str]:
    """Add everything the given skills imply, transitively, keeping input order first.

    Closed over `_IMPLIES` until it stops growing rather than one level deep: Selenium
    implies Automation, Automation implies nothing, but LangGraph implies AI Agents
    implies nothing while RAG implies LLMs implies Generative AI implies LLM
    Fundamentals — a one-level pass would stop three skills short of the truth.
    """
    out = list(dict.fromkeys(slugs))
    frontier = list(out)
    while frontier:
        implied = [
            s for slug in frontier for s in _IMPLIES.get(slug, ()) if s not in out
        ]
        out.extend(dict.fromkeys(implied))
        frontier = implied
    return out


def implies(slug: str) -> tuple[str, ...]:
    return _IMPLIES.get(slug, ())


def resolve_skills(items) -> tuple[list[str], list[str]]:
    """Split free text — a comma-separated list, or a list of strings — into recognised
    skill slugs and the leftovers, in input order and deduplicated.

    The leftovers are returned rather than discarded: "I know Cypress" is real
    information even when Cypress is not in the taxonomy, and silently dropping it makes
    the advisor look like it ignored what the learner said.
    """
    if isinstance(items, str):
        items = re.split(r"[,;\n]", items)

    known: list[str] = []
    unknown: list[str] = []
    for item in items:
        text = item.strip()
        if not text:
            continue
        slug = resolve_skill(text)
        if slug and slug not in known:
            known.append(slug)
        elif not slug and text not in unknown:
            unknown.append(text)
    return known, unknown
