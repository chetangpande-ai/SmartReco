"""Deterministic demo catalogue: 32 courses, 8 categories, 3 users.

Descriptions are written rather than generated because retrieval quality depends on
them — a catalogue of "Course about X" embeds into mush and makes the agent look worse
than it is.

    uv run python -m app.seed          # idempotent: safe to re-run
    uv run python -m app.seed --reset  # wipe first
"""

import argparse
import logging

from sqlalchemy import select

from app.db import init_db, session_scope
from app.logging_conf import configure_logging
from app.models import Base, Product, User, UserProfile
from app.schemas import ProductIn
from app.security import hash_password
from app.services import outbox
from app.services.catalog import create_product

log = logging.getLogger(__name__)

# title, category, level, price$, hours, rating, tags, instructor, description
COURSES = [
    ("Building Agentic AI Systems with LangGraph", "ai", "advanced", 89, 9, 4.8,
     ["agents", "langgraph", "orchestration", "llm"], "Dr. Maya Chen",
     "Design multi-step AI agents that plan, call tools, and recover from failure. Covers graph state machines, conditional routing, checkpointing, and human-in-the-loop review for production agent deployments."),
    ("Retrieval-Augmented Generation in Production", "ai", "advanced", 79, 8, 4.7,
     ["rag", "vector-search", "embeddings", "llm"], "Dr. Maya Chen",
     "Move beyond toy RAG demos. Chunking strategies, hybrid retrieval, re-ranking, groundedness evaluation, and the operational work of keeping a vector index consistent with a source of truth."),
    ("Prompt Engineering for Engineers", "ai", "intermediate", 49, 5, 4.5,
     ["prompting", "llm", "evaluation"], "Ravi Menon",
     "A systematic approach to prompting: structured outputs, few-shot selection, decomposition, self-consistency, and building evaluation harnesses so you can tell whether a prompt change actually helped."),
    ("Fine-Tuning Open Models on Your Own Data", "ai", "advanced", 99, 11, 4.6,
     ["fine-tuning", "lora", "transformers"], "Dr. Elena Volkov",
     "Adapt open-weight language models with LoRA and QLoRA. Dataset curation, hyperparameter selection, catastrophic forgetting, evaluation, and deciding when fine-tuning beats prompting or retrieval."),
    ("LLM Evaluation and Guardrails", "ai", "advanced", 69, 6, 4.7,
     ["evaluation", "guardrails", "safety", "llm"], "Ravi Menon",
     "Measure what your model actually does. Offline eval sets, LLM-as-judge with its failure modes, groundedness and hallucination detection, plus deterministic and model-based guardrails for user-facing output."),
    ("Introduction to Machine Learning", "ai", "beginner", 39, 12, 4.4,
     ["ml", "scikit-learn", "regression"], "Sofia Almeida",
     "Start from linear regression and finish able to train, validate, and interpret a real model. Bias-variance, cross-validation, feature engineering, and the honest limits of what a model can tell you."),
    ("Deep Learning with PyTorch", "ai", "intermediate", 79, 14, 4.6,
     ["pytorch", "neural-networks", "computer-vision"], "Dr. Elena Volkov",
     "Build neural networks from tensors up. Autograd, training loops you write yourself, convolutional and attention architectures, and the debugging habits that separate working models from silent failures."),
    ("MLOps: Shipping Models That Survive Contact With Users", "ai", "advanced", 89, 10, 4.5,
     ["mlops", "deployment", "monitoring"], "Tom Bergström",
     "Model registries, reproducible training, shadow deploys, drift detection, and rollback. The unglamorous engineering that decides whether a good model ever delivers value."),

    ("SQL for Data Analysis", "data", "beginner", 29, 8, 4.5,
     ["sql", "analytics", "postgres"], "Priya Raghavan",
     "Query like an analyst. Joins, window functions, CTEs, and aggregation patterns, taught against a messy realistic dataset rather than a clean textbook one."),
    ("Data Engineering with Python", "data", "intermediate", 69, 12, 4.4,
     ["etl", "airflow", "pipelines"], "Tom Bergström",
     "Build pipelines that do not wake you at 3am. Idempotent transforms, incremental loads, backfills, orchestration with Airflow, and designing for the day an upstream source changes shape."),
    ("Analytics Engineering with dbt", "data", "intermediate", 59, 7, 4.6,
     ["dbt", "modeling", "warehouse"], "Priya Raghavan",
     "Turn raw warehouse tables into trustworthy models. Staging and mart layers, tests as contracts, incremental materialisation, and documentation that stays true because it is generated."),
    ("Vector Databases and Semantic Search", "data", "intermediate", 59, 6, 4.7,
     ["vector-search", "embeddings", "chroma", "pinecone"], "Dr. Maya Chen",
     "How similarity search actually works. Embedding spaces, HNSW and IVF indexes, metadata filtering, hybrid lexical-plus-vector retrieval, and keeping an index synchronised with a relational source."),
    ("Statistics for Decision Making", "data", "beginner", 39, 9, 4.3,
     ["statistics", "ab-testing", "inference"], "Sofia Almeida",
     "Hypothesis testing, confidence intervals, and experiment design, aimed at people who have to make a call on Friday with imperfect data rather than publish a paper."),
    ("Data Visualisation That Persuades", "data", "beginner", 35, 5, 4.5,
     ["visualization", "charts", "storytelling"], "Hannah Weiss",
     "Choose the right chart, encode data honestly, and build a narrative that survives a sceptical room. Colour, annotation, and the specific ways charts mislead."),

    ("Python From Scratch", "programming", "beginner", 25, 15, 4.6,
     ["python", "fundamentals"], "Sofia Almeida",
     "A first programming course that does not hide the machine. Data structures, control flow, functions, files, and enough of the standard library to be independently useful."),
    ("Modern Python: Typing, Async and Packaging", "programming", "intermediate", 55, 8, 4.7,
     ["python", "async", "typing"], "Tom Bergström",
     "The Python that shipping teams write. Static typing and its escape hatches, asyncio and structured concurrency, dependency and packaging workflows with modern tooling."),
    ("FastAPI in Production", "programming", "intermediate", 59, 7, 4.8,
     ["fastapi", "api", "backend", "python"], "Marcus Hale",
     "Build an API you can operate. Dependency injection, background work, streaming, auth, observability, and load-shedding patterns for endpoints that must never block."),
    ("Test-Driven Development in Practice", "programming", "intermediate", 45, 6, 4.4,
     ["testing", "tdd", "pytest"], "Marcus Hale",
     "Tests as a design tool. What to test and what not to, fixtures without magic, property-based testing, and keeping a suite fast enough that people actually run it."),
    ("System Design Interview Preparation", "programming", "advanced", 75, 10, 4.6,
     ["system-design", "architecture", "scalability"], "Kwame Osei",
     "Reason about scale out loud. Partitioning, replication, caching layers, queues, consistency trade-offs, and how to structure an answer under time pressure."),
    ("JavaScript for Backend Developers", "programming", "beginner", 35, 9, 4.2,
     ["javascript", "node", "frontend"], "Hannah Weiss",
     "The parts of JavaScript that surprise people arriving from another language: the event loop, closures, prototypes, modules, and the DOM APIs worth knowing."),

    ("AWS Solutions Architect Foundations", "cloud", "intermediate", 79, 16, 4.5,
     ["aws", "cloud", "architecture"], "Kwame Osei",
     "Design on AWS with intent. VPC layout, IAM that is neither wide open nor unusable, storage tiers, managed databases, and the cost consequences of each choice."),
    ("Kubernetes for Application Developers", "cloud", "intermediate", 69, 11, 4.4,
     ["kubernetes", "containers", "devops"], "Tom Bergström",
     "Deploy and debug on Kubernetes without becoming a full-time operator. Pods, services, ingress, config and secrets, probes, resource limits, and reading a failing rollout."),
    ("Docker and Container Fundamentals", "cloud", "beginner", 35, 6, 4.6,
     ["docker", "containers", "devops"], "Marcus Hale",
     "Images, layers, volumes and networks explained from first principles, plus multi-stage builds and the practices that keep production images small and non-root."),
    ("Infrastructure as Code with Terraform", "cloud", "intermediate", 65, 8, 4.3,
     ["terraform", "iac", "devops"], "Kwame Osei",
     "Describe infrastructure so it can be reviewed, versioned and rebuilt. State management, modules, drift, and safe change workflows for shared environments."),
    ("Observability: Logs, Metrics and Traces", "cloud", "advanced", 69, 7, 4.7,
     ["observability", "prometheus", "tracing"], "Marcus Hale",
     "Instrument a system so incidents are diagnosable. Structured logging, useful metric cardinality, distributed tracing, SLOs, and alerts that correlate with real user pain."),

    ("Web Application Security Essentials", "security", "intermediate", 59, 8, 4.6,
     ["security", "owasp", "web"], "Dr. Amara Nwosu",
     "The OWASP Top 10 as engineering practice: injection, broken auth, CSRF, SSRF, and access control, each with the defect, the exploit, and the fix in code."),
    ("Applied Cryptography for Developers", "security", "advanced", 79, 9, 4.5,
     ["cryptography", "tls", "security"], "Dr. Amara Nwosu",
     "Use cryptography correctly without inventing it. Symmetric and public-key primitives, password hashing, TLS, key management, and the classic misuse patterns."),
    ("Threat Modelling for Product Teams", "security", "intermediate", 49, 5, 4.4,
     ["security", "threat-modeling", "architecture"], "Dr. Amara Nwosu",
     "Find design flaws before they ship. Trust boundaries, STRIDE, abuse cases, and running a threat modelling session that produces decisions rather than a document."),

    ("Product Design Fundamentals", "design", "beginner", 39, 7, 4.4,
     ["ux", "design", "prototyping"], "Hannah Weiss",
     "Research, information architecture, wireframing and usability testing — the loop that turns a vague product idea into something people can actually operate."),
    ("Design Systems at Scale", "design", "intermediate", 59, 6, 4.5,
     ["design-systems", "ui", "accessibility"], "Hannah Weiss",
     "Build a component library teams adopt willingly. Tokens, composition, accessibility as a default, documentation, and versioning shared UI without breaking consumers."),

    ("Product Management for Technical Teams", "business", "intermediate", 55, 8, 4.3,
     ["product", "roadmap", "strategy"], "Julia Marsh",
     "Prioritise credibly. Opportunity sizing, discovery, writing specs engineers respect, and communicating trade-offs to stakeholders who want everything at once."),
    ("Growth Analytics and Experimentation", "marketing", "intermediate", 49, 6, 4.4,
     ["growth", "ab-testing", "funnels"], "Julia Marsh",
     "Instrument a funnel, run experiments that are not fooling you, and read retention curves honestly. Covers cohort analysis, guardrail metrics and common statistical traps."),
]

DEMO_USERS = [
    ("admin@smartreco.dev", "admin12345", "Admin", "admin"),
    ("learner@smartreco.dev", "learner12345", "Alex Rivera", "user"),
    ("demo@smartreco.dev", "demo12345", "Demo User", "user"),
]


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

    created_users, created_products = 0, 0

    with session_scope() as db:
        for email, password, name, role in DEMO_USERS:
            if db.scalar(select(User.id).where(User.email == email)):
                continue
            user = User(
                email=email, password_hash=hash_password(password), name=name, role=role
            )
            db.add(user)
            db.flush()
            db.add(UserProfile(user_id=user.id))
            created_users += 1

    with session_scope() as db:
        existing = {t for (t,) in db.execute(select(Product.title)).all()}
        for title, cat, level, price, hours, rating, tags, instructor, desc in COURSES:
            if title in existing:
                continue
            create_product(
                db,
                ProductIn(
                    title=title,
                    description=desc,
                    category=cat,
                    level=level,
                    tags=tags,
                    price_cents=price * 100,
                    instructor=instructor,
                    duration_minutes=hours * 60,
                    rating=rating,
                    is_published=True,
                ),
            )
            created_products += 1

    # One drain embeds every new course in a single batched call rather than one per row.
    synced = outbox.drain_all()
    health = outbox.health()

    log.info(
        "seed complete",
        extra={"users": created_users, "products": created_products, "synced": synced},
    )
    return {
        "users_created": created_users,
        "products_created": created_products,
        "sync": synced,
        "health": health,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed the SmartReco demo catalogue")
    parser.add_argument("--reset", action="store_true", help="drop all data first")
    args = parser.parse_args()

    result = seed(reset=args.reset)
    print(f"\nusers created:    {result['users_created']}")
    print(f"products created: {result['products_created']}")
    print(f"vector sync:      {result['sync']}")
    print(f"in sync:          {result['health']['in_sync']} "
          f"(sql={result['health']['sql_published']} vectors={result['health']['vector_count']})")
    print("\nsign in as:")
    for email, password, _, role in DEMO_USERS:
        print(f"  {role:<5}  {email}  /  {password}")
