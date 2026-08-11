# Capabilities, in depth

The mechanics behind the summary in [README.md](../README.md): the schema, the dual-write,
the event pipeline, the agent's nodes and the efficiency gates. Moved out of the README so
it stays scannable. Exact counts live in the README and [CLAUDE.md](../CLAUDE.md).


## Platform

Email/password auth (bcrypt cost 12, JWT in an `HttpOnly` cookie, CSRF double-submit),
two roles, and **14 tables** — thirteen related, plus a standalone embedding cache:

```
users ─┬─1:1─ user_profiles
       ├─1:N─ events ──N:1─ products ─1:N─ vector_outbox
       ├─1:N─ recommendations ─1:N─ recommendation_items ─N:1─ products
       ├─1:N─ agent_runs
       └─1:N─ notifications

embedding_cache          keyed by (embedder, text), joined to nothing on purpose
```

`embedding_cache` deliberately has no foreign key: vectors are a pure function of the
model and the text, so a re-seed reuses what was already paid for even when the row that
first needed them is gone.

## Catalogue management with dual-write

Admin CRUD at `/admin/products`. The write is **not** a best-effort double write — it
is a **transactional outbox**:

1. Saving a course writes the `products` row **and** a `vector_outbox` row in **one
   commit**. Atomic.
2. A worker drains the outbox: embed → upsert → stamp `vector_synced_at`.
3. Failures retry with exponential backoff + jitter, then dead-letter after 5 attempts.
4. An hourly `reconcile()` diffs SQL against the index **by content hash** and repairs
   drift in both directions.

A naive `db.commit(); chroma.upsert()` has a window where SQL committed and the vector
write failed, and **nothing remembers it needs fixing**. Here the *intent to sync* is
committed with the data, so an outage is a delay rather than permanent divergence.

Three drift classes are tested by injecting each fault — **on both backends, live**:

| Injected | Detected | Repaired |
|---|---|---|
| Vector deleted, SQL still has it | `missing` | ✅ |
| Ghost vector with no SQL row | `orphaned` | ✅ |
| Vector's `content_hash` tampered | `stale` | ✅ |

**The invariant:** the index contains *exactly* the published courses. Unpublishing
enqueues a delete rather than setting a flag retrieval must remember to filter on.

## Behavioural event tracking

`tracker.js` — ~290 lines, no dependencies. Captures `page_view`, `product_view`,
`product_click`, `search`, `search_result_click`, `filter`, `scroll_depth`
(25/50/75/100), `dwell`, `wishlist`, `enroll`, `rec_impression`, `rec_click`.

**Non-blocking by construction:**

- Batches at **20 events or 5 s**, whichever comes first
- Scroll throttled to 1/500 ms; search debounced 400 ms and ignored below 3 characters
- `navigator.sendBeacon` on `pagehide`/`visibilitychange` — the browser completes it
  after the page is gone, which is exactly when dwell time is finally known
- `requestIdleCallback(fn, {timeout: 500})` — the timeout is mandatory, a backgrounded
  tab defers an untimed callback indefinitely
- `localStorage` spill-over retries a failed batch on the next page load, **including
  when the server answered 429 or 500** — `fetch` resolves on those, so a `.catch()`-only
  handler drops exactly the batches worth keeping
- Flushes are chunked to the server's per-batch cap, so a stash drain after an outage
  cannot arrive as one oversized 422
- Every event carries a client-generated idempotency key, so retries are free

**Measured against the running server** (25 requests per row, end to end):

| batch | p50 | p95 | max |
|---|---|---|---|
| 1 event | 2.7 ms | 4.1 ms | 6.4 ms |
| 10 events (one page's worth) | 3.0 ms | 4.4 ms | 21.2 ms |
| a course page render, for scale | 5.5 ms | 13.6 ms | — |

The beacon costs less than rendering the page it is reporting on. Three identical
replays → `persisted: 6, duplicates: 2` from a batch of 8.

The server validates **per event**, not per batch — a retired event type from a
`tracker.js` still in someone's browser cache rejects that one event, not the 99 good
ones alongside it. The rate limiter charges per event and is sized from that: a
25-page session at ~10 events a page never trips it, an 800-event burst does.

## The agentic recommendation engine

Nine LangGraph nodes. **Only two spend tokens.**

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
| `coldstart` | — | no usable behaviour yet: the catalogue's own top-rated, rather than an invented interest |
| `retrieve` | — | vector ⊕ BM25 → RRF → filter → MMR |
| `grade` | **fast** | LLM-as-judge on retrieval quality **and** re-rank, in one call |
| `refine` | — | reword the query, widen filters, retry |
| `generate` | **main** | persuasive JSON: headline, narrative, per-course reasons |
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

**Level as a progression signal.** Courses carry `beginner | intermediate | advanced`.
A learner consistently opening advanced material has outgrown the introductory shelf, so
retrieval spans their level and the next rung up — which is exactly what someone learning
wants recommended, and what lets the copy say *"this picks up where that left off"*
rather than just *"here's something similar"*.

**Is it actually personalised, or a popular list with better prose?** Three learners,
same catalogue, different histories, run against the live gateway:

| | shared with the popularity baseline | with each other |
|---|---|---|
| beginner, browsed AI/ML, searched *"learn machine learning from scratch"* | 1 / 4 | — |
| advanced, browsed security, searched *"penetration testing certification"* | 1 / 4 | 0 / 4 |
| career-switcher, browsed data, searched *"sql for analysts"* | 0 / 4 | 0 / 4 |

No two learners were shown the same course, and each got material at **their** level —
the beginner got Andrew Ng, the advanced learner got OSCP first. The copy tracks the
behaviour too:

> **You've been exploring data courses, particularly focusing on SQL and analytics. The
> Google Data Analytics Certificate directly aligns with your goal of building a
> portfolio for a career change in data.**

The security learner's run exercised the refine loop **twice** —
`analyze → plan → retrieve → grade → refine → retrieve → grade → refine → retrieve →
grade → generate → verify → finalize`, $0.000691. The career-switcher was shown **two**
courses, not four: only two genuinely fit, so the agent stopped.

## Efficiency and production thinking

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

