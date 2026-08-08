"""Load the demo catalogue and its users into the database.

The courses themselves live in `app/data/courses.py` and the taxonomy they hang off
lives in `app/data/taxonomy.json`; this module is only the loader. That split is what
lets someone add a course or a career path without opening any code that runs.

`tier` carries the learner's level — beginner, intermediate, advanced. It is the
progression ladder the agent reasons about: someone who keeps opening advanced material
is not shopping the introductory shelf.

    uv run python -m app.seed          # idempotent: safe to re-run
    uv run python -m app.seed --reset  # wipe first
"""

import argparse
import logging
import random
import secrets
from collections import defaultdict
from datetime import timedelta

from sqlalchemy import select

from app.data.courses import COURSES, social_proof
from app.db import init_db, session_scope
from app.logging_conf import configure_logging
from app.models import Base, Event, Product, User, UserProfile, utcnow
from app.schemas import ProductIn
from app.security import hash_password
from app.services import outbox
from app.services.catalog import create_product, update_product

log = logging.getLogger(__name__)

DEMO_USERS = [
    ("admin@smartreco.dev", "admin12345", "Admin", "admin"),
    ("learner@smartreco.dev", "learner12345", "Alex Rivera", "user"),
    ("demo@smartreco.dev", "demo12345", "Demo User", "user"),
]

# Purely synthetic accounts, never meant to be logged into — their only purpose is to
# give item-to-item collaborative filtering (app/services/retrieval.py's `_cf_hits`)
# something real to find. With just the 3 DEMO_USERS above and no seeded interaction
# history, CF has nothing to compute from: it needs many users' *overlapping* enrolments,
# not one person's behaviour. Clearly namespaced (`synthetic-N@smartreco.demo`) and gated
# on a single existence check for idempotency, same spirit as DEMO_USERS' per-row skip.
SYNTHETIC_USER_COUNT = 24
SYNTHETIC_EMAIL = "synthetic-{n}@smartreco.demo"


def _product_in(course) -> ProductIn:
    """One catalogue entry as the validated input the catalogue service takes.

    `ProductIn` canonicalises `teaches`/`requires` through the taxonomy resolver and
    drops anything it does not recognise, so a typo here becomes a missing edge rather
    than a dead row in `course_skills`. `scripts/check_catalogue.py` fails the build on
    exactly that, which is how the typo gets noticed.
    """
    learners, reviews = social_proof(course.rating, course.price, course.hours)
    return ProductIn(
        title=course.title,
        description=course.description,
        category=course.category,
        subcategory=course.subcategory,
        tier=course.level,
        tags=list(course.keywords),
        price_cents=course.price * 100,
        brand=course.provider,
        instructor=course.instructor or course.provider,
        spec=course.spec,
        rating=course.rating,
        reviews=reviews,
        learners=learners,
        format=course.format,
        duration_hours=course.hours,
        delivery_mode=course.delivery,
        certificate=course.certificate,
        objectives=list(course.objectives),
        curriculum=list(course.curriculum),
        projects=list(course.projects),
        labs=course.labs,
        assessments=course.assessments,
        teaches=list(course.teaches),
        requires=list(course.requires),
        is_published=True,
    )


def _seed_synthetic_interactions(db) -> int:
    """Give collaborative filtering real co-occurrence to find: N synthetic learners,
    each assigned 1-2 categories as a persona, enrolling/wishlisting a handful of
    courses drawn from those categories (plus occasional cross-category noise so the
    signal isn't a perfect grid). Courses within the same persona cluster end up
    co-enrolled across many synthetic users, which is exactly the pattern
    `_cf_hits` looks for."""
    if db.scalar(select(User.id).where(User.email == SYNTHETIC_EMAIL.format(n=0))):
        return 0  # already seeded — regenerating would just duplicate interactions

    by_category: dict[str, list[int]] = defaultdict(list)
    for p in db.scalars(select(Product)):
        by_category[p.category].append(p.id)
    categories = [c for c, ids in by_category.items() if ids]
    if not categories:
        return 0  # no catalogue yet — nothing to build interactions over

    rng = random.Random(42)  # reproducible across runs, not security-sensitive
    shared_hash = hash_password("not-a-real-account")  # hashed once: bcrypt is slow by design
    committed_types = ("enroll", "wishlist")

    created = 0
    for n in range(SYNTHETIC_USER_COUNT):
        user = User(
            email=SYNTHETIC_EMAIL.format(n=n), password_hash=shared_hash,
            name=f"Synthetic Learner {n}", role="user",
        )
        db.add(user)
        db.flush()
        db.add(UserProfile(user_id=user.id))
        created += 1

        persona = rng.sample(categories, k=min(2, len(categories)))
        pool = [pid for cat in persona for pid in by_category[cat]]
        if len(categories) > len(persona) and rng.random() < 0.2:
            # A little cross-category noise — real interest clusters aren't this clean.
            other = rng.choice([c for c in categories if c not in persona])
            pool += by_category[other]

        picks = rng.sample(pool, k=min(rng.randint(2, 5), len(pool)))
        for pid in picks:
            db.add(
                Event(
                    user_id=user.id,
                    type=rng.choice(committed_types),
                    product_id=pid,
                    dedupe_key=secrets.token_hex(12),
                    server_ts=utcnow() - timedelta(days=rng.uniform(0, 30)),
                )
            )

    return created


def seed(reset: bool = False) -> dict:
    configure_logging()
    init_db()

    if reset:
        from app.db import engine
        from app.services.vectorstore import get_vector_store

        log.warning("resetting all data")
        get_vector_store().reset()
        Base.metadata.drop_all(bind=engine)
        init_db()

    created_users, created_courses, updated_courses = 0, 0, 0

    with session_scope() as db:
        for email, password, name, role in DEMO_USERS:
            if db.scalar(select(User.id).where(User.email == email)):
                continue
            user = User(email=email, password_hash=hash_password(password), name=name, role=role)
            db.add(user)
            db.flush()
            db.add(UserProfile(user_id=user.id))
            created_users += 1

    with session_scope() as db:
        existing = {p.title: p for p in db.scalars(select(Product))}
        for course in COURSES:
            row = existing.get(course.title)
            if row is None:
                create_product(db, _product_in(course))
                created_courses += 1
            else:
                # Converge rather than skip. A course seeded before the career layer
                # existed has no skills, no duration and no subcategory, and skipping it
                # would leave the catalogue permanently half-migrated. `update_product`
                # re-embeds only when the content hash actually moved, so re-running this
                # on an unchanged catalogue costs one hash per row and no tokens.
                update_product(db, row, _product_in(course))
                updated_courses += 1

    with session_scope() as db:
        synthetic_users = _seed_synthetic_interactions(db)

    # One drain embeds every new course in a single batched call rather than one per row.
    synced = outbox.drain_all()
    health = outbox.health()

    log.info(
        "seed complete",
        extra={
            "users": created_users, "courses": created_courses,
            "updated": updated_courses, "synced": synced,
            "synthetic_users": synthetic_users,
        },
    )
    return {
        "users_created": created_users,
        "products_created": created_courses,
        "products_updated": updated_courses,
        "synthetic_users_created": synthetic_users,
        "sync": synced,
        "health": health,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed the SmartReco demo catalogue")
    parser.add_argument("--reset", action="store_true", help="drop all data first")
    args = parser.parse_args()

    result = seed(reset=args.reset)
    print(f"\nusers created:      {result['users_created']}")
    print(f"synthetic learners: {result['synthetic_users_created']} (for collaborative filtering)")
    print(f"courses created:    {result['products_created']}")
    print(f"courses updated:    {result['products_updated']}")
    print(f"vector sync:        {result['sync']}")
    print(f"in sync:            {result['health']['in_sync']} "
          f"(sql={result['health']['sql_published']} vectors={result['health']['vector_count']})")
    print("\nsign in as:")
    for email, password, _, role in DEMO_USERS:
        print(f"  {role:<5}  {email}  /  {password}")
