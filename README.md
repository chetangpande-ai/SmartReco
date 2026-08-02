# SmartReco

**A behavioural AI recommendation agent for a consumer electronics store.**

It watches how each shopper actually behaves, reasons over that behaviour with a
LangGraph agent, retrieves matching products from a vector database, and writes a
persuasive recommendation that is *grounded in real stock* and *checked for honesty*
before anyone sees it.

```
┌──────────────────────────────── BROWSER ─────────────────────────────────────┐
│ Jinja2 pages · tracker.js → throttle → batch(20 | 5s) → sendBeacon ──┐       │
└──────────────────────────────────────────────────────────────────────┼───────┘
                                                          202 in ~10ms │
┌──────────────────────────── FastAPI ─────────────────────────────────▼───────┐
│                                                                              │
│  ┌── INGEST ─────────┐   ┌── TRIGGER POLICY ────┐   ┌── AGENT (LangGraph) ─┐ │
│  │ bounded queue     │──▶│ enough events?       │──▶│ analyze → plan →     │ │
│  │ dedupe by idem key│   │ cooldown elapsed?    │   │ retrieve → grade →   │ │
│  │ chunked bulk write│   │ interests drifted?   │   │  (refine ⟲ ≤2) →     │ │
│  └────────┬──────────┘   │ budget left?         │   │ generate → verify →  │ │
│           │              │ signature cached? ─skip│  │ finalize             │ │
│           ▼              └──────────────────────┘   └──────┬───────────────┘ │
│  ┌── PROFILE ────────┐                                     │                 │
│  │ 48h half-life     │      ┌── RETRIEVAL ──────────────┐  │                 │
│  │ intent weighting  │─────▶│ vector kNN ⊕ BM25         │◀─┘                 │
│  │ interest centroid │      │ → RRF → filter → MMR      │                    │
│  └───────────────────┘      └───────────────────────────┘                    │
│  ┌── OUTBOX WORKER ──┐   ┌── APScheduler ────────────────────────────────┐   │
│  │ drain · backoff   │   │ 16:00 digest · 60s outbox · 1h reconcile      │   │
│  └────────┬──────────┘   └───────────────────────────────────────────────┘   │
└───────────┼──────────────────────┬────────────────┬──────────────┬───────────┘
            ▼                      ▼                ▼              ▼
     ┌─────────────┐      ┌──────────────┐   ┌────────────┐  ┌───────────┐
     │ SQLite / PG │      │   Mesh API   │   │  ChromaDB  │  │   SMTP    │
     │  10 tables  │      │ chat + embed │   │ / Pinecone │  │ / file    │
     └─────────────┘      └──────────────┘   └────────────┘  └───────────┘
```

---

## Quickstart

```bash
git clone <your-fork> && cd SmartReco
cp .env.example .env          # add your MESHAPI_API_KEY (starts with rsk_)
make install
make seed
make run                      # http://localhost:8000
```

| Sign in as | Email | Password |
|---|---|---|
| admin | `admin@smartreco.dev` | `admin12345` |
| shopper | `shopper@smartreco.dev` | `shopper12345` |

**It also runs with no API key at all.** Set `LLM_ENABLED=false` and the app uses a
deterministic embedder and template-based copy — every feature works, nothing is
mocked out, and it spends nothing. That's the mode the whole test suite runs in.

### See it work

1. Browse a few pairs of headphones, search for something, scroll, sit on a product
   page for a minute.
2. Open **`/me`** — the agent reads that behaviour and writes a recommendation.
   The *"Why these?"* panel lists the only facts it was allowed to cite.
3. Open **`/admin`** — SQL↔vector sync health, LLM calls avoided and why, the compiled
   agent graph, and every run with its node path, tokens, cost and latency.
4. Press **"Run digest now"** — the daily email renders to `data/outbox/*.html`.

The catalogue is 35 real products across audio, phones, laptops, cameras, TVs, smart
home, gaming and wearables — deliberately recognisable, so you can judge for yourself
whether a recommendation makes sense instead of taking it on trust.

---

## How each requirement is met

### 1. Platform

Email/password auth (bcrypt cost 12, JWT in an `HttpOnly` cookie, CSRF double-submit),
two roles, and **10 related tables**:

```
users ─┬─1:1─ user_profiles
       ├─1:N─ events ──N:1─ products ─1:N─ vector_outbox
       ├─1:N─ recommendations ─1:N─ recommendation_items ─N:1─ products
       ├─1:N─ agent_runs        └── embedding_cache
       └─1:N─ notifications
```

### 2. Product management with dual-write

Admin CRUD at `/admin/products`. The write is **not** a best-effort double write — it
is a **transactional outbox**:

1. Saving a product writes the `products` row **and** a `vector_outbox` row in **one
   commit**. Atomic.
2. A worker drains the outbox: embed → upsert → stamp `vector_synced_at`.
3. Failures retry with exponential backoff + jitter, then dead-letter after 5 attempts.
4. An hourly `reconcile()` diffs SQL against the index **by content hash** and repairs
   drift in both directions.

A naive `db.commit(); chroma.upsert()` has a window where SQL committed and the vector
write failed, and **nothing remembers it needs fixing**. Here the *intent to sync* is
committed with the data, so an outage is a delay rather than permanent divergence.

Three drift classes are tested by injecting each fault:

| Injected | Detected | Repaired |
|---|---|---|
| Vector deleted, SQL still has it | `missing` | ✅ |
| Ghost vector with no SQL row | `orphaned` | ✅ |
| Vector's `content_hash` tampered | `stale` | ✅ |

**The invariant:** the index contains *exactly* the published products. Unpublishing
enqueues a delete rather than setting a flag retrieval must remember to filter on.

### 3. Behavioural event tracking

`tracker.js` — ~230 lines, no dependencies. Captures `page_view`, `product_view`,
`product_click`, `search`, `filter`, `scroll_depth` (25/50/75/100), `dwell`,
`add_to_cart`, `purchase`, `rec_impression`, `rec_click`.

**Non-blocking by construction:**

- Batches at **20 events or 5 s**, whichever comes first
- Scroll throttled to 1/500 ms; search debounced 400 ms and ignored below 3 characters
- `navigator.sendBeacon` on `pagehide`/`visibilitychange` — the browser completes it
  after the page is gone, which is exactly when dwell time is finally known
- `requestIdleCallback(fn, {timeout: 500})` — the timeout is mandatory, a backgrounded
  tab defers an untimed callback indefinitely
- `localStorage` spill-over retries a failed batch on the next page load
- Every event carries a client-generated idempotency key, so retries are free

**Measured in a browser:** **50 events accepted in 10 ms**; unload flush lands in
0.32 s; three identical replays → `persisted: 6, duplicates: 2` from a batch of 8.

The server validates **per event**, not per batch — a retired event type from a
`tracker.js` still in someone's browser cache rejects that one event, not the 99 good
ones alongside it.

### 4. The agentic recommendation engine

Eight LangGraph nodes. **Only two spend tokens.**

```
START → analyze → plan ─┬─▶ coldstart ────────────┐
                        └─▶ retrieve → grade ─┬───┴─▶ generate → verify ─┬─▶ finalize → END
                                ▲             │                          │
                                └── refine ◀──┘ (score < 0.6, max 2)     └─▶ generate (repair ×1)
```

| Node | Model? | Job |
|---|---|---|
| `analyze` | — | profile → search query, metadata filters, citable evidence |
| `plan` | — | enough signal to retrieve against, or cold start? |
| `retrieve` | — | vector ⊕ BM25 → RRF → filter → MMR |
| `grade` | **fast** | LLM-as-judge on retrieval quality **and** re-rank, in one call |
| `refine` | — | reword the query, widen filters, retry |
| `generate` | **main** | persuasive JSON: headline, narrative, per-product reasons |
| `verify` | — | groundedness + honesty gate |
| `finalize` | — | fall back to honest copy rather than ship something wrong |

**`analyze` is deliberately deterministic.** Turning a behaviour profile into a search
query is arithmetic over scores already computed — paying a model to restate its own
inputs adds latency, cost and a failure mode for no accuracy gain. The nodes that need
*judgement* get a model.

**`verify` is the anti-hallucination gate.** Two independent failure modes, both tested:

- A recommended `product_id` that was never offered → **dropped**, repair triggered
- Copy that is grounded but *dishonest* → **caught by guardrails**

Repair is bounded to **one** attempt. An unbounded loop against a model repeating the
same mistake burns budget without converging.

**Tier as an upgrade signal.** Products carry `entry | mid | flagship`. A shopper
consistently opening flagship models is not shopping the entry shelf, so retrieval
filters accordingly — the same progression logic that makes the copy able to say
*"you've been looking at the top of this category"* rather than just *"here's something
similar"*.

### 5. Efficiency and production thinking

A recommendation costs a model call only if **every** gate passes:

```
events_since_last_rec ≥ 5
  AND now − last_rec ≥ 90s
  AND ( cosine drift ≥ 0.10  OR  ≥15 new events  OR  expired  OR  user pressed refresh )
  AND daily budget not exhausted  AND  per-user cap not hit  AND  circuit closed
  AND behaviour signature not already cached
```

The **behaviour signature** is the cache key: `sha256` of the *ranked order* of the top
categories, tags, brands, tier and price band. Ranked order, not raw scores — scores
move on every single event, so a score-based key would never hit.

Every skip records a reason, and `/admin` shows them:

```
cache_hit 41 · interests_unchanged 22 · too_few_new_events 18 · cooldown 9 · warming_up 4
```

Other production concerns: bounded ingest queue that **sheds** rather than growing
under flood; embedding cache keyed by content hash (re-seeding costs $0); circuit
breaker; per-model runtime parameter negotiation; structured JSON logs with a request
id threaded through to the agent; Prometheus `/metrics`; `/healthz` + `/readyz`;
token-bucket rate limiting; CSP and security headers.

---

## Bonus features

| Bonus | Status |
|---|---|
| ⭐ **Structured agent framework** | LangGraph 1.2, 8 nodes, conditional edges, bounded refine + repair loops. `/admin` renders the graph **read from the compiled object**, so it cannot drift from what runs |
| ⭐ **Scheduled proactive delivery** | APScheduler: digest at 16:00 UTC (with jitter), outbox drain 60 s, reconcile hourly. Idempotent via a unique `digest:<user>:<date>` key |
| ⭐ **Observability** | **LangSmith** traces the graph (with the Mesh calls nested inside it as real `llm` runs), **Logfire** traces the request around it, and one run reaches both through LangSmith's OTel bridge. Plus a durable `agent_runs` table: node path, grade, refines, tokens, cost, latency, trace URL. [Details ↓](#observability-two-views-one-run) |
| ⭐ **Retrieval polish** | Hybrid vector+BM25 → RRF → metadata filters → MMR → LLM re-rank. Relevance floor **tuned by sweep, not guessed** |
| ➕ **Guardrails** | Deterministic rails always on (free, offline); NeMo Guardrails as an opt-in second layer, routed through Mesh |
| ➕ **Two vector backends** | Chroma (default) and Pinecone behind one protocol |
| ➕ **Offline eval harness** | 20 paraphrase probes, recall@k + MRR, `make eval` |

---

## Design decisions with evidence

### Model selection — measured, not assumed

I probed the live gateway before writing a line of agent code:

| Model | Latency | Completion tokens | Valid JSON? |
|---|---|---|---|
| `moonshotai/kimi-k3` | 12.7 s | 396 (**220 reasoning**) | ✅ best copy |
| `openai/gpt-4o-mini` | 1.8 s | 142 (0 reasoning) | ✅ |
| `google/gemini-2.5-flash` | 5.5 s | 886 (757 reasoning) | ❌ truncated |
| `anthropic/claude-haiku-4.5` | 3.3 s | 181 | ❌ ```json fenced |

Three behaviours in `mesh.py` exist purely because of that probe:

1. **`EmptyCompletion`** — a reasoning model can return **200 OK with `""`** if the
   token ceiling is consumed by hidden reasoning. Default ceiling is 4000 with an
   explicit error instead of a baffling downstream parse failure.
2. **Runtime parameter negotiation** — `kimi-k3` returns 400 on any `temperature != 1`.
   A 400 naming a parameter drops it and *remembers* that per model, rather than a
   hardcoded quirk list that rots.
3. **Fence-tolerant JSON parsing** — models fence JSON *even in `json_object` mode*.

Default is `gpt-4o-mini`; switch `MESHAPI_MODEL` to `kimi-k3` for a demo recording.

### Retrieval floor — swept, not guessed

A kNN index returns *k* results whether or not any are near. With `k=40` over a
35-product catalogue that is nearly the whole catalogue, so those ranks are noise that
dilutes rank fusion. Sweeping the floor over 20 paraphrase probes (`make sweep`):

| ratio | recall@3 | recall@5 | MRR | avg pool |
|---|---|---|---|---|
| 0.00 | 0.85 | 0.90 | 0.863 | 30.1 |
| 0.35 | 0.85 | 0.90 | 0.863 | 24.2 |
| 0.50 | 0.85 | 1.00 | 0.882 | 12.4 |
| **0.55** | **0.95** | **1.00** | **0.896** | **9.2** |
| 0.65 | 1.00 | 1.00 | 0.917 | 5.0 |
| 0.80 | 1.00 | 1.00 | 0.917 | 2.5 |

Recall *improves* as the floor tightens — the opposite of the usual precision/recall
trade, and only explicable once you notice the pool was the whole catalogue. `0.55`
takes most of the gain while leaving MMR a pool worth diversifying over; past 0.65 the
vector side nearly vanishes and BM25 is effectively deciding alone. Tuned on one
catalogue of 35 items — re-run `make sweep` if yours differs.

### RRF fuses by rank, never by score

Cosine lives in `[-1,1]`; BM25 is unbounded and corpus-dependent. Normalising them into
a weighted sum means inventing an exchange rate that shifts with every catalogue edit.
Tested: a vector hit scoring `999999` and a lexical hit scoring `0.0001` produce
**identical** fused scores — both are rank 1.

### Guardrails, because the model is being asked to persuade

Invented discounts and manufactured scarcity are not hypothetical failure modes here —
they are the *most likely* ones, because persuasive training data is full of them. The
deterministic rails block outcome promises, fabricated urgency, invented discounts,
**any price not in the catalogue**, unsupported superlatives, and PII. They run always,
free, offline and in CI.

**Real products raise a second risk.** A model recognises the Sony WH-1000XM5 and will
happily recite specifications from training data that may be wrong or years stale. The
generation prompt therefore states that catalogue facts are the *only* facts it may
use, and the `spec` field exists so there is always something accurate to quote.

### Observability: two views, one run

Two backends, because they answer different questions and neither is complete alone.
**LangSmith** shows the graph — which node ran, what the model was actually asked, why
`verify` rejected a draft. **Logfire** shows the request that graph was serving — the
HTTP span, the SQL underneath it, the Mesh call and its token count, on one timeline.

They are joined rather than duplicated. Setting `LANGSMITH_TRACING_MODE=hybrid` makes
the LangSmith SDK write each run to LangSmith *and* emit it as an OpenTelemetry span,
which Logfire receives because Logfire owns the global tracer provider. One run, two
places. Verified against the live project — the whole graph arrives, refine loops,
guardrail repair and all:

```
smartreco.recommend                                          $0.000601
  └ LangGraph
     └ analyze → plan → retrieve → grade → generate → verify → finalize
                          ChatOpenAI 673 tok ─┘        └─ 1447 tok
```

Mesh traffic leaves through the raw OpenAI SDK, not a LangChain model, so LangSmith
would otherwise show chain nodes with nothing underneath them — no prompt, no tokens.
`wrap_openai` on the Mesh client fixes that: the three calls in the run above appear as
three `llm` runs totalling 2,747 tokens, which is exactly what `agent_runs` recorded
independently. Logfire gets the same calls through `instrument_openai`.

Two things here fail *silently*, which is why `app/observability.py` is one module and
not two, and why they have tests:

**Ordering.** `langsmith.Client` decides once, at construction, where its OTel spans go:
adopt an existing global provider, or build its own pointed at LangSmith's OTLP endpoint.
Configure Logfire after that and LangGraph spans never reach it — nothing errors, they
just go elsewhere. `configure()` runs Logfire first and only then sets hybrid mode; set
hybrid with no provider installed and every run lands in LangSmith *twice*.

**Sinks.** `LOGFIRE_ENABLED=true` with no token and no console output builds a span per
request and drops all of them — instrumentation overhead buying nothing. That is refused
with a warning naming the fix rather than started. And `send_to_logfire` is pinned to
`"if-token-present"`: the plain `True` makes Logfire open an interactive project setup,
which in a container is a boot that hangs forever.

The same wiring fixed a bug it exists to catch. `AgentRun.langsmith_url` had always
stored `""`: it was read next to the other fields, after the graph returned, where
`get_current_run_tree()` is already `None`. The link is now captured inside the traced
call, and a regression test models exactly that distinction.

`/readyz` reports which backends are live, so "are traces being recorded" is answerable
without reading logs — the question you ask once the demo is already running.

---

## Testing

```bash
make test     # 269 tests, offline, no API key, spends nothing
make cov      # coverage report (83%)
make eval     # retrieval quality against 20 paraphrase probes
```

| Module | Covers |
|---|---|
| `test_unit.py` | passwords, JWT, CSRF, JSON extraction, guardrails, BM25, RRF, MMR |
| `test_mesh.py` | Mesh gateway against a stub: retries, breaker, budget, parameter negotiation, the empty-completion trap |
| `test_storage.py` | embedding cache, vector stores, dual-write, outbox retry + dead-letter, all three drift classes |
| `test_pipeline.py` | ingest, dedupe, decay, evidence, drift, signature, **all 11 trigger gates** |
| `test_agent.py` | graph shape, groundedness verifier, repair bounding, end-to-end |
| `test_digest.py` | audience, rendering, once-only delivery, SMTP send path, scheduler |
| `test_api.py` | HTTP: auth, CSRF, tracking ingest, admin access control, dual-write over HTTP |
| `test_observability.py` | sink selection, the OTel bridge, and the two silent failure modes below |

Several tests are explicitly labelled regressions for bugs found during the build — an
unpublished product staying recommendable, a search-only shopper getting no interest
signal, duplicate texts being embedded twice in one batch, error pages returning 200.

---

## Configuration

Everything is in `.env` (see `.env.example`). The knobs that matter most:

| Variable | Default | Effect |
|---|---|---|
| `LLM_ENABLED` | `true` | `false` runs the whole app offline for $0 |
| `LLM_DAILY_BUDGET_USD` | `1.00` | Hard spend cap across all callers |
| `REC_MIN_EVENTS` | `5` | Events before a recommendation is considered |
| `REC_COOLDOWN_SECONDS` | `90` | Floor between runs for one shopper |
| `REC_DRIFT_THRESHOLD` | `0.10` | Interest change needed to justify a call |
| `RETRIEVAL_SCORE_RATIO` | `0.55` | Relevance floor, relative to the best hit |
| `VECTOR_BACKEND` | `chroma` | or `pinecone` |
| `LANGSMITH_TRACING` | `false` | `true` + an API key traces the graph |
| `LOGFIRE_ENABLED` | `false` | Needs `LOGFIRE_TOKEN` **or** `LOGFIRE_CONSOLE=true` to do anything |
| `MAIL_BACKEND` | `auto` | `auto` picks SMTP if configured, else the file sink |
| `DIGEST_HOUR` | `16` | Daily digest, UTC |

---

## Deployment

```bash
docker compose up --build                      # SQLite on a volume
docker compose --profile postgres up --build   # Postgres instead
make migrate                                   # alembic upgrade head
```

Multi-stage image, non-root user, healthcheck, `data/` on a volume. CI runs lint,
the suite on Python 3.11 and 3.12 with an 80% coverage floor, a from-scratch migration
check, a seed-and-verify-sync check, and a container build that must report healthy.

Set `MESH_API_KEY` as a GitHub Actions secret if you want live runs in CI — the test
suite does not need it.

---

## Honest limitations

- **Pinecone is implemented but untested.** No API key was available. The filter
  translation and call shapes are written against the 9.1 signatures and unit-tested,
  but I have not run it against a live index. Chroma is fully exercised.
- **SMTP has never talked to a real mail server.** No credentials were available, so
  every run has used the file sink. The send path *is* covered against a stubbed
  `smtplib.SMTP` — connection parameters, STARTTLS, login, and the multipart/alternative
  structure with both text and HTML bodies — which catches the failures that actually
  happen (missing plain-text part, wrong headers). What remains unproven is whether a
  given provider accepts the mail. `SesNotifier` is likewise written but unexercised.
- **Logfire is wired but has never shipped to the cloud.** No token was available. The
  configuration, instrumentation and the LangSmith→OTel bridge are proven — the bridge
  by inspecting which tracer LangSmith ends up holding, and by watching a real span come
  out of the console exporter — but `logfire.pydantic.dev` has not received a trace from
  this app. LangSmith, by contrast, is verified live: the full ten-node graph arrives in
  the `smartreco` project.
- **APScheduler assumes one process.** The jobs are individually safe to run
  concurrently (the outbox coalesces, reconcile is a diff, the digest has a unique
  key), so scaling out means changing the scheduler, not the jobs.
- **Skip counters are in-process** and reset on restart. The dashboard labels them
  "since this process started" and shows durable database totals separately, rather
  than mixing the two into one authoritative-looking wrong number.
- **The budget gate is checked before a call**, so a single call can overshoot the cap;
  cost is only knowable afterwards. The guarantee is that spending *stops* once the
  limit is crossed, not that it never crosses it.
- **Retrieval is tuned on 20 probes against 35 products.** recall@1 0.85, recall@5 1.00,
  MRR 0.896 — good, but a 35-item catalogue is small enough that these numbers would
  need re-establishing at real scale. The two hardest probes both confuse two Sonos
  speakers with each other, which is a genuinely fine distinction.
- **Product names and specs are real, prices and ratings are illustrative.** This is a
  demo storefront, not a price comparison service.
- **Product imagery is generated, not photographed.** Real product photography belongs
  to the manufacturers, so each card renders a category glyph on a hue derived from the
  product slug — deterministic, zero third-party requests, and compatible with the
  strict CSP. It reads as a design choice rather than a missing asset.

---

## Stack

FastAPI · SQLAlchemy 2 · SQLite/Postgres · Alembic · **Mesh API** (all model traffic) ·
LangChain 1.3 · LangGraph 1.2 · LangSmith · Logfire/OpenTelemetry · Chroma · Pinecone ·
NeMo Guardrails · APScheduler · Jinja2 · vanilla JS · uv · pytest · ruff · Docker

Code style follows [Andrej Karpathy's engineering philosophy](.claude/skills/karpathy-coding-style/SKILL.md);
see [AGENTS.md](AGENTS.md).
