"""Demo catalogue: 35 online courses across 8 tracks, plus 3 users.

Recognisable courses and providers, because a reviewer can judge instantly whether
recommending the Deep Learning Specialization to someone who lingered on the Machine
Learning Specialization is sensible — that judgement is much harder to make about an
invented catalogue.

Descriptions are written rather than templated: retrieval quality depends on them, and a
catalogue of "A good course" embeds into mush. Every fact here is also the *only* thing
the agent is allowed to claim about a course, so the syllabus line matters.

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

from app.db import init_db, session_scope
from app.logging_conf import configure_logging
from app.models import Base, Event, Product, User, UserProfile, utcnow
from app.schemas import ProductIn
from app.security import hash_password
from app.services import outbox
from app.services.catalog import create_product

log = logging.getLogger(__name__)

# title, provider, track, level, price$, rating, tags, syllabus, description
COURSES = [
    # ---- ai-ml ---------------------------------------------------------------
    ("Machine Learning Specialization", "DeepLearning.AI", "ai-ml", "beginner", 49, 4.9,
     ["machine-learning", "python", "supervised-learning", "regression"],
     "3 courses · 2 months at 10h/week · Python · certificate",
     "Andrew Ng's rebuilt introduction to machine learning, taught in Python rather than Octave. Linear and logistic regression, neural networks, decision trees and clustering, with the maths kept to what you actually need to reason about a model."),
    ("Deep Learning Specialization", "DeepLearning.AI", "ai-ml", "intermediate", 59, 4.8,
     ["deep-learning", "neural-networks", "tensorflow", "cnn"],
     "5 courses · 3 months at 10h/week · TensorFlow · certificate",
     "Builds neural networks from the ground up before touching a framework, then covers convolutional and sequence models. The hyperparameter tuning and structuring-ML-projects courses are the part practitioners come back to."),
    ("Neural Networks: Zero to Hero", "Andrej Karpathy", "ai-ml", "intermediate", 0, 4.9,
     ["neural-networks", "backpropagation", "pytorch", "transformers"],
     "10 video lectures · ~20h · PyTorch · code-along · free",
     "Backpropagation implemented by hand, then micrograd, then makemore, then a GPT built line by line. Nothing is hidden behind a framework call, which is exactly why it sticks."),
    ("LangChain for LLM Application Development", "DeepLearning.AI", "ai-ml", "intermediate", 0, 4.5,
     ["llm", "langchain", "rag", "agents"],
     "6 lessons · ~2h · Python notebooks · free",
     "A short practical tour of chains, memory, retrieval over your own documents, and agents. Assumes you can already write Python and want the vocabulary rather than the theory."),
    ("Practical Deep Learning for Coders", "fast.ai", "ai-ml", "beginner", 0, 4.8,
     ["deep-learning", "pytorch", "computer-vision", "nlp"],
     "8 lessons · ~40h · PyTorch and fastai · free",
     "Top-down teaching: you train a working image classifier in lesson one, then peel back the layers over the course. Suits people who learn by getting something running and asking why afterwards."),
    ("Natural Language Processing with Transformers", "O'Reilly", "ai-ml", "advanced", 89, 4.6,
     ["nlp", "transformers", "huggingface", "fine-tuning"],
     "11 chapters · ~30h · Hugging Face · text and notebooks",
     "Attention, tokenisation, fine-tuning and distillation, worked through the Hugging Face stack. Goes into production concerns — latency, quantisation, serving — that most NLP material skips."),

    # ---- web-dev -------------------------------------------------------------
    ("The Complete Web Developer Bootcamp", "Udemy", "web-dev", "beginner", 89, 4.7,
     ["html", "css", "javascript", "fullstack"],
     "65h video · 5 projects · lifetime access · certificate",
     "The long-form path from no code at all to a deployed full-stack application. HTML and CSS, then JavaScript properly, then Node and databases. Its length is the point — nothing is skipped."),
    ("Complete Intro to React", "Frontend Masters", "web-dev", "intermediate", 39, 4.8,
     ["react", "javascript", "hooks", "frontend"],
     "8h video · builds one app throughout · exercises",
     "React taught by building a single application from an empty folder, with no scaffolding tool to hide the wiring. Hooks, context, routing and testing, in the order you would actually meet them."),
    ("Total TypeScript", "Matt Pocock", "web-dev", "advanced", 249, 4.9,
     ["typescript", "types", "generics", "javascript"],
     "20h+ · interactive exercises · 200+ challenges",
     "Type-level programming taught as puzzles you solve in your editor. Generics, conditional types, and the patterns behind well-typed libraries — for people who already write TypeScript and keep hitting its ceiling."),
    ("Testing JavaScript", "Kent C. Dodds", "web-dev", "intermediate", 179, 4.7,
     ["testing", "javascript", "react", "end-to-end"],
     "20h video · unit, integration and e2e · exercises",
     "Builds a testing framework from scratch so the tools stop being magic, then covers static analysis, unit, integration and end-to-end testing. Opinionated about testing behaviour rather than implementation."),
    ("Responsive Web Design Certification", "freeCodeCamp", "web-dev", "beginner", 0, 4.6,
     ["html", "css", "flexbox", "accessibility"],
     "~300h · 5 build projects · certification · free",
     "Hands-on HTML and CSS with no video at all — you write code in the browser and it checks your work. Covers flexbox, grid and accessibility, ending in five projects you build unaided."),

    # ---- data ----------------------------------------------------------------
    ("Google Data Analytics Certificate", "Google", "data", "beginner", 49, 4.7,
     ["data-analysis", "sql", "tableau", "career-change"],
     "8 courses · 6 months at 10h/week · SQL, R, Tableau · certificate",
     "Built for people entering analytics with no background: spreadsheets, then SQL, then visualisation, then R. Ends with a portfolio case study, and is explicitly aimed at landing a first analyst job."),
    ("SQL for Data Analysis", "Udacity", "data", "beginner", 79, 4.5,
     ["sql", "postgres", "joins", "window-functions"],
     "4 weeks · PostgreSQL · query workspace · projects",
     "Joins, aggregations, subqueries and window functions, practised against a real transactional database rather than toy tables. Stops short of database design, which is not what analysts need first."),
    ("Data Engineering Zoomcamp", "DataTalks.Club", "data", "intermediate", 0, 4.7,
     ["data-engineering", "dbt", "airflow", "spark"],
     "9 weeks · Docker, dbt, Airflow, Spark, BigQuery · free",
     "A full pipeline built week by week: ingestion, warehousing, transformation, orchestration and streaming. Cohort-based and project-heavy, ending with an end-to-end pipeline you deploy yourself."),
    ("Statistics with Python Specialization", "University of Michigan", "data", "intermediate", 49, 4.6,
     ["statistics", "python", "inference", "regression"],
     "3 courses · 2 months · Python · certificate",
     "Inference, fitting and interpreting models, taught with real datasets and an emphasis on when a method is and is not appropriate. The counterweight to learning modelling purely as library calls."),
    ("Analytics Engineering with dbt", "dbt Labs", "data", "advanced", 0, 4.5,
     ["dbt", "sql", "data-modelling", "warehouse"],
     "Self-paced · dbt Cloud · modelling and testing · free",
     "Turning raw warehouse tables into tested, documented, version-controlled models. Assumes strong SQL and is about the engineering discipline around it rather than the query language."),

    # ---- cloud ---------------------------------------------------------------
    ("AWS Certified Solutions Architect Associate", "A Cloud Guru", "cloud", "intermediate", 129, 4.6,
     ["aws", "certification", "architecture", "networking"],
     "30h video · hands-on labs · practice exams · SAA-C03",
     "Exam-focused but genuinely architectural: virtual networks, identity, storage classes, high availability and cost. The labs matter more than the videos, and the practice exams are close to the real thing."),
    ("Docker and Kubernetes: The Complete Guide", "Udemy", "cloud", "intermediate", 94, 4.7,
     ["docker", "kubernetes", "devops", "ci-cd"],
     "22h video · multi-container apps · CI/CD · certificate",
     "Containers from first principles, then multi-container applications, then Kubernetes with a real deployment pipeline. Ends with a working CI/CD setup rather than a cluster you never ship to."),
    ("Terraform Deep Dive", "Pluralsight", "cloud", "advanced", 99, 4.4,
     ["terraform", "infrastructure-as-code", "modules", "devops"],
     "12h video · modules, state, workspaces · labs",
     "State management, module design and the failure modes that only show up on a team — drift, locking, and refactoring live infrastructure without downtime. Assumes you have already written some Terraform."),
    ("Google Cloud Professional Data Engineer", "Coursera", "cloud", "advanced", 59, 4.5,
     ["gcp", "bigquery", "streaming", "certification"],
     "6 courses · 3 months · BigQuery, Dataflow, Pub/Sub · certificate",
     "Designing data systems on Google Cloud: batch and streaming pipelines, warehouse modelling, and the trade-offs between managed services. Certification-aligned but heavier on design than on trivia."),

    # ---- security ------------------------------------------------------------
    ("Practical Ethical Hacking", "TCM Security", "security", "beginner", 30, 4.8,
     ["pentesting", "linux", "networking", "active-directory"],
     "25h video · lab environment · Active Directory · certificate",
     "Networking and Linux fundamentals first, then reconnaissance, exploitation and Active Directory attacks in a lab you build yourself. Widely recommended as the first step before a practical certification."),
    ("Web Security Academy", "PortSwigger", "security", "intermediate", 0, 4.9,
     ["web-security", "owasp", "xss", "sql-injection"],
     "200+ labs · browser-based · Burp Suite · free",
     "Every major class of web vulnerability with a working lab to exploit, written by the people who build Burp Suite. Reading is minimal; you learn by breaking something and being told exactly why it broke."),
    ("Offensive Security Certified Professional", "OffSec", "security", "advanced", 1649, 4.7,
     ["pentesting", "certification", "exploitation", "reporting"],
     "PEN-200 course · 90 days lab access · 24h practical exam",
     "The practical penetration testing certification: a 24-hour hands-on exam against machines you must actually compromise. Demanding, expensive, and the credential hiring managers recognise."),
    ("Security Engineering Fundamentals", "Educative", "security", "intermediate", 59, 4.3,
     ["appsec", "threat-modelling", "cryptography", "secure-design"],
     "Text-based · interactive · threat modelling · ~15h",
     "Designing systems that fail safely: threat modelling, applied cryptography, secrets handling and authentication design. Aimed at engineers who build the systems rather than the people testing them."),

    # ---- design --------------------------------------------------------------
    ("UI/UX Design Specialization", "CalArts", "design", "beginner", 49, 4.6,
     ["ui", "ux", "figma", "portfolio"],
     "4 courses · 3 months · Figma · portfolio project · certificate",
     "Design fundamentals taught by an art school: typography, colour and layout before tooling. Ends with a portfolio piece, which is the part that matters when nobody will read your certificate."),
    ("Design Systems with Figma", "Interaction Design Foundation", "design", "intermediate", 45, 4.4,
     ["figma", "design-systems", "components", "documentation"],
     "Self-paced · components, variants, tokens · ~20h",
     "Building a component library other people can actually use: naming, variants, tokens and the documentation that stops a system rotting six months after launch."),
    ("Refactoring UI", "Adam Wathan and Steve Schoger", "design", "beginner", 149, 4.9,
     ["visual-design", "css", "typography", "spacing"],
     "Book and 200 tips · before and after examples · for developers",
     "Visual design as a set of concrete rules for people who write code and cannot draw. Spacing, hierarchy, colour and depth, each shown as a before and after rather than explained in the abstract."),
    ("Motion Design for Interfaces", "School of Motion", "design", "advanced", 399, 4.5,
     ["motion", "animation", "prototyping", "interaction"],
     "8 weeks · After Effects · critiqued assignments",
     "Easing, choreography and the timing that makes an interface feel responsive rather than decorative. Assignments are critiqued by working motion designers, which is most of what you pay for."),

    # ---- product -------------------------------------------------------------
    ("Digital Product Management", "University of Virginia", "product", "beginner", 49, 4.6,
     ["product-management", "discovery", "roadmap", "agile"],
     "5 courses · 2 months · case studies · certificate",
     "The product manager's job described honestly: discovery, prioritisation, working with engineering, and measuring whether anything improved. Case-study driven rather than framework worship."),
    ("Product Analytics", "Reforge", "product", "advanced", 1995, 4.7,
     ["analytics", "experimentation", "retention", "metrics"],
     "6 weeks · cohort-based · live sessions · peer projects",
     "Choosing metrics that survive contact with reality, designing experiments, and reading retention curves. Cohort-based and expensive; the peer group is a large part of the value."),
    ("Continuous Discovery Habits", "Teresa Torres", "product", "intermediate", 299, 4.8,
     ["discovery", "user-research", "interviewing", "assumptions"],
     "Self-paced · opportunity solution trees · templates",
     "A weekly research habit rather than a research phase: interviewing continuously, mapping opportunities, and testing assumptions before building. Practical and unusually specific about the mechanics."),
    ("Technical Writing for Engineers", "Google", "product", "beginner", 0, 4.5,
     ["writing", "documentation", "communication", "editing"],
     "2 courses · ~8h · exercises · free",
     "Editing your own writing down to something readable: active voice, short sentences, and structuring a document so people can skim it. Short, free, and applies to every design doc you will ever write."),

    # ---- career --------------------------------------------------------------
    ("Grokking the Coding Interview", "Educative", "career", "intermediate", 79, 4.6,
     ["interview-prep", "algorithms", "patterns", "problem-solving"],
     "Text-based · 16 patterns · 180+ problems · interactive",
     "Coding interview problems grouped into sixteen recurring patterns rather than memorised individually, so an unseen question maps onto something you have already solved."),
    ("System Design Interview Course", "ByteByteGo", "career", "advanced", 149, 4.7,
     ["system-design", "scalability", "interview-prep", "architecture"],
     "Illustrated · 30+ case studies · deep dives",
     "Real systems worked through end to end — a URL shortener, a news feed, a chat service — with the trade-offs stated rather than assumed. Heavily illustrated, which is why the concepts stick."),
    ("Speaking to Technical Audiences", "LinkedIn Learning", "career", "beginner", 35, 4.2,
     ["communication", "presenting", "storytelling", "public-speaking"],
     "4h video · exercises · certificate",
     "Structuring a talk, handling questions you cannot answer, and presenting technical work to people who do not share your context. Short and practical rather than inspirational."),
]

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

    created_users, created_courses = 0, 0

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
        existing = {t for (t,) in db.execute(select(Product.title)).all()}
        for title, provider, track, level, price, rating, tags, syllabus, description in COURSES:
            if title in existing:
                continue
            create_product(
                db,
                ProductIn(
                    title=title,
                    description=description,
                    category=track,
                    tier=level,
                    tags=tags,
                    price_cents=price * 100,
                    brand=provider,
                    spec=syllabus,
                    rating=rating,
                    is_published=True,
                ),
            )
            created_courses += 1

    with session_scope() as db:
        synthetic_users = _seed_synthetic_interactions(db)

    # One drain embeds every new course in a single batched call rather than one per row.
    synced = outbox.drain_all()
    health = outbox.health()

    log.info(
        "seed complete",
        extra={
            "users": created_users, "courses": created_courses, "synced": synced,
            "synthetic_users": synthetic_users,
        },
    )
    return {
        "users_created": created_users,
        "products_created": created_courses,
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
    print(f"vector sync:        {result['sync']}")
    print(f"in sync:            {result['health']['in_sync']} "
          f"(sql={result['health']['sql_published']} vectors={result['health']['vector_count']})")
    print("\nsign in as:")
    for email, password, _, role in DEMO_USERS:
        print(f"  {role:<5}  {email}  /  {password}")
