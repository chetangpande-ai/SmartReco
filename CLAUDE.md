# SmartReco — working notes

Read [AGENTS.md](AGENTS.md) for the coding style and project conventions, and
[`.claude/skills/karpathy-coding-style/SKILL.md`](.claude/skills/karpathy-coding-style/SKILL.md)
for the full style guide. Those two say *how* to write here. This file is the map of
*what is already here* — written after a full audit on 2026-08-03 so the next session
does not have to re-derive it.

## What this is

A behavioural recommendation agent for an online **learning platform**. The catalogue is
35 real courses. Ignore the commit message on `7f4849b` ("changed domain from learning
courses to electronics") — it is wrong; the domain is courses everywhere, and the schema
just uses generic names for it (`brand` = provider, `spec` = syllabus line, `tier` =
beginner/intermediate/advanced).

FastAPI + SQLAlchemy 2 → LangGraph agent → hybrid retrieval over Chroma or Pinecone.
Every model call leaves through Mesh.

## Commands

```bash
make install   # uv sync --all-extras
make seed      # 35 courses + 3 users, idempotent
make run       # :8000
make test      # 314 tests, offline, ~40s
make cov       # coverage, currently 90%
make lint      # ruff, currently clean
make eval      # retrieval quality, needs a live index
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
| Tables | **10** — 9 related + `embedding_cache` | `models.Base.metadata.tables` |
| Deterministic guardrail checks | **35** = 31 regex + 4 cross-checks | the three rail lists in `services/guardrails.py` |
| Trigger gates | **11** | `services/triggers.evaluate` |
| Seeded courses | **35** | `COURSES` in `app/seed.py` |
| Event types | **12** current + 2 retired aliases | `schemas.EVENT_TYPES` |
| Tests / coverage | **314 / 90%** | `make cov` |

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
