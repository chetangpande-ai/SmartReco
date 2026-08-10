# SmartReco — working notes

Read [AGENTS.md](AGENTS.md) for the coding style and project conventions, and
[`.claude/skills/karpathy-coding-style/SKILL.md`](.claude/skills/karpathy-coding-style/SKILL.md)
for the full style guide. Those two say *how* to write here. This file is the map of
*what is already here* — written after a full audit on 2026-08-03 so the next session
does not have to re-derive it.

## What this is

A **career learning marketplace** for an online learning platform, with two engines
under it:

1. **Behavioural recommendation** — a LangGraph agent that watches what you browse and
   recommends from it. This is the original system.
2. **The career layer** — a skill graph that answers "what should I learn next to become
   an X?" from a stated profile rather than from behaviour. Added 2026-08-08.

The catalogue is 66 real courses. Ignore the commit message on `7f4849b` ("changed
domain from learning courses to electronics") — it is wrong; the domain is courses
everywhere, and the schema just uses generic names for it (`brand` = provider, `spec` =
syllabus line, `tier` = beginner/intermediate/advanced).

FastAPI + SQLAlchemy 2 → LangGraph agent → hybrid retrieval over Chroma or Pinecone.
Every model call leaves through Mesh.

## The career layer

Three files hold it, and the split matters:

- **`app/taxonomy.py`** — the vocabulary, loaded once at import from
  `app/data/taxonomy.json`. Categories, subcategories, topics, 22 career roles with
  their required skills, 10 career paths with ordered steps. No table, no migration:
  it is versioned with the code because it changes with releases, not with users.
- **`app/data/courses.py`** — the 66 courses, each declaring what it `teaches` and what
  it `requires` as canonical taxonomy slugs. `app/seed.py` is only the loader.
- **`app/services/advisor.py`** — the engine. `analyse()` is deterministic: set
  difference against the role's requirements, then a walk down the career path resolving
  each gap to a course whose prerequisites the previous step already satisfied.

**The model does not choose the courses.** It writes the narrative around an analysis
that is already complete, and there is a template fallback under it — `LLM_ENABLED=false`
still produces a full, correct plan. That division is the whole design: "which course
next" is answerable from the skill graph, and a career plan is the most consequential
thing this platform says to anyone.

Three rules in the ranking do most of the quality work. Each exists because its absence
produced a specifically bad plan:

- **Skills imply skills** (`taxonomy._IMPLIES`). Someone who says "Selenium, API testing"
  has been testing for a living. A literal set difference against a role asking for
  "Testing" told a ten-year tester to learn testing.
- **Seniority is per-skill, not global** (`advisor._target_tier`). Ten years of testing
  says nothing about meeting machine learning for the first time. Applying seniority
  everywhere opened a plan with a 60-hour advanced course on an unfamiliar subject.
- **Interview-prep courses never teach a gap** (`advisor.NON_TEACHING_FORMATS`). They
  list Python because their problems are written in it. Without the exclusion, the plan
  for a Java developer opened with "Grokking the Coding Interview" to learn Python.

## Commands

```bash
make install    # uv sync --all-extras
make migrate    # alembic upgrade head — run before `seed` on an existing database
make seed       # 66 courses + 3 users, converges rather than skips
make run        # :8000
make test       # 437 tests, offline, ~50s
make cov        # coverage, currently 92%
make lint       # ruff, currently clean
make catalogue  # courses vs taxonomy, currently 0 problems
make eval       # retrieval quality, needs a live index
```

**The first `uv run` in a fresh session syncs the environment and can take minutes.** A
test run that looks hung on a cold checkout is almost always that, not a deadlock — the
suite itself finishes in about 40 seconds. Don't go bug-hunting before you've re-run it
on a warm environment.

## Verified counts

These were checked against the code, not against the README. The README repeats all of
them in prose, so if you change one, grep for it there.

| Fact | Value | Re-derive with |
|---|---|---|
| Graph nodes | **9** | `app.agent.graph.describe()["nodes"]` |
| Conditional edges | **6**, from 3 branch points | `describe()["edges"]` |
| Tables | **14** — 13 related + `embedding_cache` | `models.Base.metadata.tables` |
| Deterministic guardrail checks | **35** = 31 regex + 4 cross-checks | the three rail lists in `services/guardrails.py` |
| Trigger gates | **11** | `services/triggers.evaluate` |
| Seeded courses | **66** | `COURSES` in `app/data/courses.py` |
| Taxonomy categories / skills | **21 / 628** | `taxonomy.taxonomy()` |
| Career roles / paths | **22 / 10** | `taxonomy.roles()`, `taxonomy.paths()` |
| Roadmap stages | **8** | `taxonomy.ROADMAP_STAGES` |
| Event types | **12** current + 2 retired aliases | `schemas.EVENT_TYPES` |
| Tests / coverage | **437 / 92%** | `make cov` |
| Catalogue ↔ taxonomy | **0 problems** | `make catalogue` |

`node_path` counts `coldstart` as a real node; the numbered comments in `nodes.py` label
it `3b`, which is where the old "eight nodes" claim came from.

## The path a recommendation takes

```
tracker.js → POST /api/events → 202 in ~3ms
   → bounded asyncio queue (sheds when full, drops are counted)
   → batched bulk insert, ON CONFLICT DO NOTHING on dedupe_key
   → profile.refresh: 48h half-life decay, intent weights, interest centroid
   → triggers.evaluate: 11 gates, cheapest first, every skip records a reason
   → recommender.generate_for_user
   → LangGraph: analyze → plan → (coldstart | retrieve → grade →⟲refine) →
                generate → verify →⟲generate(×1) → finalize
   → Recommendation + RecommendationItem + AgentRun rows
```

Only `grade` and `generate` spend tokens. `analyze` is deterministic on purpose.

## Invariants that will bite you (career layer)

- **`main.py` runs `create_all` at startup, and alembic owns the schema too.** Anyone
  who starts the app before migrating gets new tables without new columns. The
  `9c41ab7e5d02` migration guards every create for exactly that reason — do not
  "clean it up" into plain `op.create_table` calls.
- **Subcategory ids are only unique within a category.** `tools` is both a Project
  Management and a UI/UX subcategory. Everything keys on `Subcategory.key`
  (`category/sub`); keying on the bare id silently merges them.
- **Stage dicts use `entries`, not `items`.** Jinja resolves `stage.items` to the dict
  *method* and renders nothing, with no error.
- **A course must never both teach and require the same skill.** It sends the advisor in
  a circle. `make catalogue` fails on it.
- **`ProductIn` canonicalises and drops unknown skills.** An unrecognised slug in
  `course_skills` can never be reached by a gap query, so it would be a link that only
  looks like it exists.

## Invariants that will bite you

- **Every model call goes through `services/mesh.py`.** That choke point is what makes
  the budget cap, circuit breaker and token accounting real. Never call the OpenAI SDK
  directly elsewhere.
- **The vector index holds exactly the published products.** Unpublishing enqueues a
  *delete*. Never add an `is_published` filter at query time.
- **`content_hash` includes `is_published`.** Leave it out and unpublishing hashes
  identical to publishing, so the sync is skipped and the course stays recommendable.
- **Datetimes are UTC.** `models.utcnow()` to write, `models.ensure_utc()` to read —
  SQLite returns naive where Postgres returns aware, and mixing them raises on SQLite
  only, which means it passes CI and dies in the demo.
- **Tests are offline.** `LLM_ENABLED=false` selects the hashing embedder and template
  copy writer. A test needing the network is a test that won't run in CI.

## Landmines

- **Observability ordering.** Logfire must be configured before the first
  `langsmith.Client`, which decides once at construction where its OTel spans go. Get it
  wrong and nothing errors — the spans just silently go somewhere else. That is why
  `observability.py` is one module and why `configure()` runs at import in `main.py`.
- **`embedder.embed_documents(..., db=session)`.** Pass the caller's session when you are
  inside a write transaction, or SQLite deadlocks against itself: the inner write waits
  on the outer lock, and the outer can't commit until the inner returns.
- **`main._load_user` returns a detached `User`.** Eagerly-loaded scalars are fine;
  touching a relationship on it raises.
- **Reasoning models can return 200 OK with `""`** when the token ceiling is eaten by
  hidden reasoning. `mesh.EmptyCompletion` exists for exactly that.
- **Chroma returns cosine *distance*.** Converted to similarity at the store boundary;
  nothing above `vectorstore.py` should know which backend is live.
- **Pinecone's `list()` paginates response objects, not id strings.** Getting that wrong
  once made `reconcile()` a no-op that looked like a repair.

## Fixed in the 2026-08-03 audit

Don't reintroduce these:

- `notify.send_once` returns `"sent" | "duplicate" | "failed"`, **not a bool**. The bool
  collapsed "already sent" and "send failed" into `False`, so `digest`'s `failed` counter
  was permanently 0 and the job reported clean runs while delivering nothing. Pinned by
  `test_a_send_failure_is_counted_as_failed_not_skipped`.
- `mesh.chat()` no longer charges a dropped parameter against the retry budget — a quirky
  model used to get one real attempt out of three, and the final error read
  `failed after 3 attempts: None`.
- The admin product form defaulted `tier` to `"entry"`, which `ProductIn` rejects. Valid
  tiers are `beginner | intermediate | advanced`.
- `mesh._unsupported` now records a *replacement* per parameter, not a bare drop. It
  remembered "max_tokens was rejected" and re-dropped it on every later call without
  re-adding `max_completion_tokens`, so every request after the first went out with no
  output ceiling — the empty-completion trap that setting exists to prevent. Pinned by
  `test_the_rename_is_carried_into_later_calls`.
- Whitespace-only search queries were embedded into the interest centroid as `""`.
- `mmr_select`'s redundancy `max()` now takes `default=0.0` instead of raising on an
  empty selection.
- `test_cooldown` was flaky under full-suite load: `_give_recommendation` stamped
  `last_rec_at = utcnow()` and the events after it got the same timestamp, which
  `_count_since`'s strict `>` reads as "not new". It now backdates 30s — still inside the
  90s cooldown. If you write a trigger test, make the recommendation and its events
  unambiguously ordered; timestamp granularity on Windows is coarse enough to bite.

## Doc drift

README.md is unusually specific — node counts, rail counts, table counts, coverage,
latency percentiles. That precision is the point, and it is also what rots. When you
change the graph shape, the guardrail lists, the schema or the test count, grep README.md
before you commit. The same numbers appear in module docstrings (`nodes.py`, `models.py`,
`retrieval.py`), which have drifted before.
