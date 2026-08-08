"""Fail the build when the catalogue and the taxonomy have drifted apart.

The career layer is a graph: a skill gap is a set of taxonomy slugs, and the only thing
turning that set back into something a learner can enrol in is `Course.teaches`. A typo
there does not raise — it silently produces a roadmap step with no course behind it, and
the first person to notice is whoever is demoing. So it gets checked, offline and free,
on every CI run.

    uv run python -m scripts.check_catalogue
"""

import sys
from collections import Counter

from app import taxonomy as T
from app.data.courses import COURSES
from app.services.advisor import NON_TEACHING_FORMATS


def main() -> int:
    tax = T.taxonomy()
    # Mirrors what the advisor will actually consider — a skill "taught" only by an
    # interview-prep course is a skill no roadmap can route a learner to.
    taught = {
        s for c in COURSES if c.format not in NON_TEACHING_FORMATS for s in c.teaches
    }
    problems: list[str] = []

    for course in COURSES:
        where = f"{course.title!r}"
        if not T.category(course.category):
            problems.append(f"{where}: unknown category {course.category!r}")
        if not T.subcategory(course.subcategory):
            problems.append(f"{where}: unknown subcategory {course.subcategory!r}")
        elif T.subcategory(course.subcategory).category != course.category:
            problems.append(f"{where}: subcategory is not under {course.category!r}")
        if course.format not in tax.program_types:
            problems.append(f"{where}: unknown format {course.format!r}")
        if course.level not in ("beginner", "intermediate", "advanced"):
            problems.append(f"{where}: unknown level {course.level!r}")
        for slug in course.teaches + course.requires:
            if slug not in tax.skills:
                problems.append(f"{where}: {slug!r} is not a taxonomy skill")
        # A course that requires what it teaches sends the advisor in a circle.
        for slug in set(course.teaches) & set(course.requires):
            problems.append(f"{where}: both teaches and requires {slug!r}")

    duplicates = [t for t, n in Counter(c.title for c in COURSES).items() if n > 1]
    problems += [f"duplicate course title {t!r}" for t in duplicates]

    # Every step of every career path has to resolve to a real course, or the roadmap
    # renders a stage the learner cannot act on.
    for path in tax.paths.values():
        for slug in path.skills:
            if slug not in taught:
                problems.append(f"path {path.slug!r}: no course teaches {slug!r}")

    for problem in problems:
        print(f"  {problem}", file=sys.stderr)

    covered = len(taught & set(tax.skills))
    print(
        f"{len(COURSES)} courses · {covered} taxonomy skills taught · "
        f"{len(tax.paths)} career paths · {len(problems)} problems"
    )
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
