# Architecture

The system-level view: components, request path, deployment topology, and how the agent
manages context across a single run versus across a user's whole history. For exact
counts (nodes, tables, tests, coverage) see [README.md](../README.md) and
[CLAUDE.md](../CLAUDE.md) — this doc explains shape, not numbers, so it doesn't drift
when a number changes.

## Components

```
                     ┌──────────────┐
   browser ────────▶ │   FastAPI    │
   (Jinja pages,     │   app.main   │
   tracker.js)       └──────┬───────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                     ▼
  ┌───────────┐       ┌─────────────┐       ┌─────────────┐
  │  routers/ │       │  services/  │       │  agent/     │
  │  pages    │       │  events     │       │  graph      │
  │  events   │──────▶│  profile    │──────▶│  nodes      │
  │  auth     │       │  triggers   │       │  state      │
  │  admin    │       │  retrieval  │       │  prompts    │
  │  recs     │       │  guardrails │       └──────┬──────┘
  └───────────┘       │  mesh       │              │
                       └──────┬──────┘              │
                              │                      │
                    ┌─────────┴──────────┐          │
                    ▼                    ▼          ▼
              ┌───────────┐       ┌────────────┐  ┌────────┐
              │ Postgres/ │       │  Chroma /   │  │  Mesh  │
              │  SQLite   │       │  Pinecone   │  │ (LLM)  │
              └───────────┘       └────────────┘  └────────┘
```

Every arrow into "Mesh" is the same choke point — `app/services/mesh.py`. No other
module talks to an LLM or embedding API directly. That's what makes the budget cap,
circuit breaker and token accounting real system properties instead of per-call
discipline someone has to remember.

## Request path

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

This is the same pipeline documented in `CLAUDE.md`; it's repeated here because
everything below refers back to a stage in it.

## Context management: short-term vs. long-term

The two are structurally different, and conflating them is the fastest way to
misunderstand this codebase.

### Short-term — `AgentState` (`app/agent/state.py`)

Scoped to exactly one `graph.invoke()` call. Built by `new_state()`, discarded when the
call returns. There is no LangGraph checkpointer (`graph.compile()` is called bare,
`app/agent/graph.py:75`) — this is deliberate: SmartReco has no multi-turn conversation
concept. Every recommendation is a single-shot pipeline run, not a chat turn.

The interesting part is the accumulator fields — `node_path`, `warnings`, `llm_calls`,
`prompt_tokens`, `completion_tokens`, `cost_usd`, all `Annotated[list | int | float, add]`
in the `TypedDict`. LangGraph replaces a node's return value into state by default; a
field without a reducer gets overwritten on every node visit. That's wrong for these
fields specifically because `retrieve` and `grade` can each run twice in one request (the
refine loop), and `generate` can run twice (the one-shot repair loop) — without
`operator.add`, a second pass would erase the first pass's token count and clobber
`node_path` down to one entry instead of a real trace. `prompt_versions` (added for
prompt versioning, see [evals.md](evals.md)) is a deliberate exception: it's a dict, dicts
don't support `+`, so `grade`/`generate` merge it by hand
(`{**state.get("prompt_versions", {}), "grade": ...}`) rather than using a reducer.

### Long-term — `UserProfile` (`app/models.py`, `app/services/profile.py`)

One row per user, entirely rebuilt from the `events` table on every `refresh()` call — a
recompute over a lookback window, not an append-only log and not a vector memory store
(the vector index holds the *catalogue*, not per-user memory). Three mechanisms do the
work: recency decay (each event's weight halves every `profile_halflife_hours`, 48h by
default), intent weighting (`EVENT_WEIGHTS` — an enrollment is worth 5x a page view), and
an interest centroid (a single vector in the same embedding space as the catalogue, which
is what turns "has this person's interest actually changed" into a number —
`profile.drift()` — instead of a guess).

`profile.evidence()` is a second read path off the same event data, turned into
checkable natural-language facts. This is the *only* behavior the generated copy is
allowed to cite — handing the model a fact list instead of raw events is what stops it
inventing a plausible-sounding history, and it doubles as the "why am I seeing this?"
panel shown to the user.

### What's explicitly not memory

`AgentRun` (`app/models.py`) is a full record of one graph execution — node path,
retrieval stats, grade score, refine count, tokens, cost, prompt versions, a LangSmith
link. It looks like it could be memory. It isn't: nothing in `graph.py`/`state.py` ever
reads a prior `AgentRun` back into a new run's `AgentState`. It's an audit trail the
agent never consults about itself — a deliberate boundary, not an oversight. If a future
version wants the agent to reason about its own past runs, that's a new mechanism, not
something `AgentRun` already secretly provides.

## Data flow through the safety layers

Every generated recommendation passes through, in order: **Mesh** (budget check before
the call is even attempted) → **verify()** (candidate-set membership check, then a
second check against live SQL, since the vector index can lag) → **guardrails.validate()**
(31 regex rails + 4 cross-checks, always on; optional NeMo LLM layer, off by default) →
**finalize()** (scrub the offending sentences, or fall back to deterministic
catalogue-only copy if nothing survives). See [evals.md](evals.md) for how this is now
also measured (DeepEval metrics) and adversarially tested (`make red-team`).

## Deployment topology

Single-process FastAPI + Uvicorn, APScheduler running in-process (outbox drain,
reconcile, digest, rate-limiter pruning — see the "APScheduler assumes one process" note
in README's Honest Limitations). SQLite locally, Postgres in the AWS deployment path.
Chroma locally (in-process, zero setup), Pinecone in the managed deployment path — both
behind the same `VectorStore` protocol so nothing above `vectorstore.py` knows which
backend is live. Rate limiting is in-process token buckets, explicitly not
multi-worker-safe today (`app/ratelimit.py`).
