# Low-level design

Module-by-module internals — the "how it actually works" detail that
[README.md](../README.md) documents in prose-with-evidence form but doesn't fully spell
out mechanically. Read [architecture.md](architecture.md) first for the system-level
view; this is one level down.

## `app/agent/state.py` — the accumulator-reducer mechanic

`AgentState` is a `TypedDict(total=False)`. Most fields overwrite on each node's return
(LangGraph's default merge behavior); six are `Annotated[T, operator.add]`:
`node_path`, `warnings`, `llm_calls`, `prompt_tokens`, `completion_tokens`, `cost_usd`.
LangGraph applies the reducer function to `(old_value, new_value)` on every node return
that touches that key. For a list field that means `old + new` (concatenation); for
`int`/`float` it means arithmetic addition. This is why `node_path` is a real execution
trace — including the refine loop revisiting `retrieve`/`grade` and the repair loop
revisiting `generate` — rather than just "the last node that ran."

`prompt_versions: dict[str, str]` has no reducer (dicts don't support `+`). `grade` and
`generate` each merge it manually before returning:
`{**state.get("prompt_versions", {}), "grade": prompts.GRADE_PROMPT_VERSION}`. Skipping
the manual merge would mean a second pass through the repair loop silently drops
whichever node's entry isn't being re-set that pass.

## `app/agent/graph.py` — wiring

`build_graph()` registers 9 nodes and wires them with `add_edge` (unconditional) and
`add_conditional_edges` (branch point + a router function returning the next node's
name). Compiled once at import (`graph = build_graph()`), no checkpointer. Three branch
points:

| Router | Decides between | On what |
|---|---|---|
| `route_after_plan` | `retrieve` / `coldstart` | is there a usable query at all |
| `route_after_grade` | `refine` / `generate` | `grade_score < threshold` AND `refine_count < max` AND grader proposed a `better_query` |
| `route_after_verify` | `generate` / `finalize` | `repair_notes` is set AND `node_path.count("generate") < 2` |

The repair cap is deliberately counted from `node_path` rather than a dedicated state
field — simple, but it means anyone adding a second node literally named `"generate"`
would silently change the cap's meaning. Worth a dedicated counter if that ever becomes
a real risk.

## `app/agent/nodes.py` — per-node logic

Only `grade` and `generate` spend tokens; `analyze` is arithmetic over profile scores by
design. Each node returns a **partial** state dict — only the keys it wants to change.

- **`analyze`** — no profile row → `strategy="coldstart"` immediately. Otherwise builds a
  weighted search query from `recent_queries[:3] + interests[:3] + brand_scores[:2] +
  tag_scores[:5] + top_terms[:5]`, deduped, capped at 300 chars. Tier filter always spans
  two rungs (`beginner→[beginner,intermediate]`) so "the next step up" stays reachable.
  Price ceiling is `price_affinity * 2.5` — generous on purpose; a tight cap silently
  hides the best match over a few dollars.
- **`grade`** — one call does judging *and* re-ranking, because they need identical
  context and splitting them doubles the spend to answer half a question each. Re-orders
  candidates by the model's `ranked_ids`, but **only among ids it was actually given**
  (`by_id[i] for i in ranked if i in by_id`) — an id it invented here would otherwise
  sail past the verifier looking like a legitimate candidate. `mesh.available` false, or
  a `MeshError`, both degrade to `grade_score=0.7` with a warning rather than blocking.
- **`refine`** — widens filters (drops the tier constraint, doubles the price ceiling)
  and rewords using the grader's `better_query`. Filter widening matters as much as the
  reword: a weak result set is often an over-tight filter, not a bad query.
- **`generate`** — no model available → `_deterministic_copy()` (a template writer built
  from `profile.evidence()`, not an LLM). A `repair_notes` from a prior `verify()` failure
  gets appended via `prompts.REPAIR_SUFFIX.format(violations=...)`.
- **`verify`** — two independent checks, both real failure modes: (1) picked ids not in
  the candidate set get dropped (classic RAG hallucination); (2) survivors get
  re-confirmed against live SQL, since the vector index can lag behind an unpublish.
  `guardrails.validate()` runs on the full copy blob with `allowed_prices_cents` scoped
  to the *actual* candidate prices, so an invented price fails even if the id is real.
- **`finalize`** — no-op if nothing needs repair. Picks survived but prose didn't →
  `guardrails.scrub()` (sentence-level filter, confidence drops to 0.4). Nothing
  survived → `_deterministic_copy()`, tagged `fell_back_to_deterministic`. This is the
  node that guarantees a user is never shown something known to be wrong.

## `app/services/profile.py` — decay math

`refresh()` windows events by `profile_lookback_days`, then per event computes
`decay = 0.5 ** (age_hours / half_life)` (half_life defaults to 48h), multiplies by an
intent weight (`EVENT_WEIGHTS`: enroll/purchase 5.0, wishlist/cart 3.0, search 2.0,
search-result-click 1.8, rec-click 1.6, product-click 1.2, product-view 1.0, filter 0.5,
scroll-depth 0.25, page-view 0.15, rec-impression 0.05 — `dwell` uses a separate
`min(dwell_ms/60000, 3.0)`, capped at 3 minutes so a stale open tab can't dominate).
Events past ~11 half-lives (weight ≤ 0.0005) are skipped outright. The centroid is the
weighted mean of embedded title/brand/category/tag/query text, L2-normalized, stored as
raw float32 bytes. `signature()` fingerprints *ranked order* of top interests/tags/
brand/tier/price-band rather than raw floats, because raw scores drift every event and
would defeat the whole point of using it as a cache key.

## `app/services/triggers.py` — the 11-gate cascade

`evaluate()` checks gates cheapest-first, in this order, so a cheap skip is never
miscounted as an expensive one: `no_activity` → `force`-bypass (still budget-gated) →
`cache_hit` (same `behavior_signature`, unexpired, not a fallback) → `warming_up`
(`events_total < rec_min_events`, default 5) → budget block → `first_recommendation` →
`too_few_new_events` → `cooldown` (`rec_cooldown_seconds`, default 90) →
`interests_unchanged` (none of drift ≥ `rec_drift_threshold` 0.10, staleness ≥
`rec_staleness_events` 15, or expiry) → budget block again → drift/staleness/expired
decision. `_budget_block()` itself checks three things: daily USD budget, per-user daily
call cap (`llm_max_calls_per_user_per_day`, default 25), circuit breaker state. Every
skip calls `metrics.inc(..., reason=reason)` — that's the data behind the admin
dashboard's "N calls avoided" panel.

## `app/services/mesh.py` — the state machines

Two small state machines guard every call:

- **Budget** (`_Budget`) — thread-locked, resets on date rollover, `check_and_reserve()`
  raises `BudgetExceeded` *before* the request is attempted, not after.
- **Circuit breaker** (`_Breaker`) — opens after 5 consecutive failures, blocks for 60s,
  then lets exactly one request through (a time-window check rather than an explicit
  half-open flag).

Retry (`chat()`, up to 3 attempts, exponential backoff) treats a rejected parameter
(`BadRequestError`) as free — it doesn't consume a retry attempt, because resending
without the offending parameter is a different request the gateway hasn't seen yet.
`_unsupported` remembers `{model: {param: replacement_or_None}}` learned at runtime;
critically the *replacement* is remembered, not just the drop — `max_tokens` gets
renamed to `max_completion_tokens` on some endpoints rather than simply unsupported, and
forgetting the replacement would silently strip the output ceiling from every later call.

## `app/services/guardrails.py` — two layers

Deterministic rails (31 regex rules across three categories — forbidden claims,
fabricated urgency, unsupported commerce — plus 4 cross-checks: invented discount,
price-not-in-catalogue, PII, length cap) always run, cost nothing, and are what actually
protects users. `NemoRails` is an optional second layer, LLM-judged, off by default,
loaded lazily so its absence never breaks the app. `validate()` is the combined entry
point; `scrub()` is the sentence-level last resort used by `finalize`.

## `app/services/retrieval.py` — hybrid pipeline

Vector kNN + hand-rolled BM25 (`LexicalIndex`, k1=1.5, b=0.75, rebuilt only when the
catalogue's `(count, max updated_at)` fingerprint changes) → reciprocal rank fusion
(`rrf_fuse`, K=60 — combines by *rank* because cosine similarity and BM25 scores live on
incomparable scales) → `_cut_weak_hits` (drops kNN results below
`max(retrieval_min_score, best*retrieval_score_ratio)`, tuned via `make sweep` against a
live index) → MMR (`mmr_select`, λ=0.65, relevance-minus-redundancy) → top N. Entirely
model-free except the embedding call itself, which is what keeps it unit-testable
offline.

**The collaborative-filtering leg composes with this rather than joining it.**
`rrf_fuse` stays a two-list primitive (it's the thing `tests/test_unit.py::TestRrf`
pins, and generalizing its signature for a leg that's only sometimes present wasn't
worth risking). Instead, `_cf_hits(db, committed_ids, exclude, top_n)` runs a plain
co-occurrence query — which other products the users who committed to `committed_ids`
also committed to — and `retrieve()` treats the vector⊕lexical result as one
already-ranked list, fusing *that* with the CF list in a second `rrf_fuse` pass. When
`committed_ids` is empty (no history yet), `retrieve()`'s `ranking` variable is set to
the exact same dict object as the first pass — not a copy, not a re-fusion — so a new
learner gets provably identical behaviour to before the leg existed. `Candidate` carries
the result as `cf_score`/`cf_rank`, and `retrieved_by` reports which legs actually
contributed (`"vector+lexical"`, `"vector+lexical+cf"`, `"popularity"` for the cold-start
path, etc.) rather than the old three-way ternary that only ever said `"both"`.

## `app/agent/ranking.py` — ranking beyond the LLM

Split out of `nodes.py` for the same reason `prompts.py` is: a distinct concern from
graph wiring. Two things live here, both deliberately *not* machine-learned — there's no
labelled click/conversion data at this catalogue's scale to train weights from, and the
module docstring says so rather than overclaiming.

- **`heuristic_score(candidate, profile_features)`** — a hand-weighted sum over category
  match (0.35), tag overlap (0.30), tier match (0.15), brand match (0.10), price
  closeness (0.05), rating (0.05), computed entirely from data already in memory at
  `grade()` time. `profile_features` is a new `AgentState` field — `analyze()` reads the
  structured `UserProfile` columns (`interests`, `tag_scores`, `brand_scores`,
  `price_affinity_cents`, `tier_affinity`) into it while its DB session is still open (as
  a `dict(...)` copy — `prof` itself is detached the moment that `with` block ends), since
  previously only the string-rendered `profile_summary`/`evidence` ever left `analyze()`.
  `grade()` reorders candidates by this score *unconditionally*, before anything else —
  which is what makes it "a tiebreaker on top of the LLM's ranking" even when the LLM
  does run: any id the LLM's `ranked_ids` omits falls back to heuristic order instead of
  arbitrary retrieval order.
- **`confidence_margin(scores)`** — gap between the best and second-best heuristic score.
  When it clears `settings.ranker_skip_margin` (0.25 default), `grade()` skips the LLM
  call entirely and trusts the heuristic ranking, recording
  `warnings: ["grade_skipped_heuristic_confident"]` — a new, real entry in the same
  skip-reason ledger the trigger policy already writes to.
- **`apply_exploration_slot(candidates, exposure, epsilon, top_k)`** — with probability
  `epsilon` (`settings.explore_epsilon`, 0.15 default), swaps the lowest-ranked kept slot
  for whichever candidate outside the top-k has the lowest `impression_counts()`
  (a new cross-user aggregate over `rec_impression` events, next to `popular()` in
  `retrieval.py`). Called at the top of `generate()` rather than as its own graph node —
  every upstream path (LLM-graded, heuristic-skipped, cold-start) already funnels through
  `generate()`, so this is the one place that reaches all of them without a 10th node.
  `epsilon<=0`/`epsilon>=1` are exact boundaries (`random.random()` never returns exactly
  1.0), which is what lets tests drive this deterministically without mocking `random` —
  the test suite itself pins `EXPLORE_EPSILON=0` process-wide (`tests/conftest.py`) so
  real randomness never leaks into an assertion that happens to have more than `top_k`
  candidates.

## `app/taxonomy.py` — one vocabulary, three sources

The source JSON writes skills as prose, and the same skill arrives spelled three ways:
`AI Evals` on the AI Engineer *role*, `AI Evaluation` on the AI Engineer *path*,
`AWS/Azure/GCP` as one string covering three skills. Everything is canonicalised to a
slug on load, so downstream code compares slugs and never strings.

- **`_SPLIT`** — the handful of source names that are genuinely several skills. An
  explicit list rather than "split on slash", because `CI/CD` is one skill whose name
  contains one.
- **`_ALIASES`** — second names folded into a canonical slug. Doubles as the matcher for
  free text a learner types, which is why informal spellings (`ml`, `genai`, `k8s`,
  `dsa`) are in there.
- **`_IMPLIES` + `expand_skills()`** — having a skill means having the ones it entails.
  Closed transitively rather than one level deep: RAG implies LLMs implies Generative AI
  implies LLM Fundamentals, and a single pass stops three skills short. This is what
  stops the advisor telling a ten-year tester to learn testing.
- **`Subcategory.key`** — `category/sub`, because subcategory ids are only unique inside
  their category. `tools` is both a Project Management and a UI/UX subcategory; keyed on
  the bare id, one silently replaces the other.

## `app/services/advisor.py` — the skill-gap engine

`analyse()` is deterministic end to end. The model is called afterwards, for prose only,
and there is a template fallback under it — `LLM_ENABLED=false` produces the same plan.

1. **Held skills** = stated skills + skills from *completed* enrollments, then
   `expand_skills()`. Completions count because they are the one claim on the page that
   can be verified.
2. **Required** = the target role's skills, or the path's own steps when the path
   describes a job the role list does not name (Generative AI Engineer).
3. **Order** comes from the career path, not the role. A role's skill list is an
   unordered job description; a path is a teaching sequence someone thought about.
   Requirements the path never mentions are appended so nothing vanishes.
4. **Course per gap**, ranked by `_rank()` — lower is better:
   `(unmet prerequisites, tier distance, position of the skill in the course's own list,
   -rating, -reviews)`.

Three rules in that ranking each exist because their absence produced a specific bad
plan, and each is pinned by a test:

- **`unmet` dominates.** A rating-first ranking hands someone the best-rated RAG course,
  which is also the one assuming they already know RAG.
- **`_target_tier()` is per-skill.** Seniority applies only where the learner already
  works — judged by whether they hold any skill from the same part of the catalogue.
  Applied globally, fifteen years of QA opened a plan with a 60-hour advanced course on
  a subject the learner had never touched.
- **`NON_TEACHING_FORMATS`.** Interview-prep courses list Python because their problems
  are written in it. Without the exclusion, the plan for a Java developer opened with
  *Grokking the Coding Interview* to learn Python.

After a course is chosen, everything it teaches is marked held, so the next gap is ranked
against a learner who has already taken it. That is what makes the output a sequence
rather than a list, and it is asserted directly:
`test_a_course_never_precedes_its_own_prerequisites` walks the plan and checks it.

`_stages()` composes the eight `taxonomy.ROADMAP_STAGES` into plain JSON stored on
`CareerPlan.stages`, so an old plan stays renderable after the catalogue moves. The list
key is `entries`, not `items` — Jinja resolves `stage.items` to the dict *method* and
renders nothing, silently.

## `app/services/learning.py` — enrollment state

Separate from the `enroll` event on purpose. The event is an immutable record that a
click happened at a moment in time and feeds the behaviour profile; the row is mutable
state that changes on every session. Deriving "62% through" by replaying an append-only
log on each dashboard render would be absurd, and deleting the row must not rewrite
history — so neither derives from the other.

`set_progress()` is the only place a certificate can be minted, which is what makes
"issued exactly once, and never revoked by re-opening a finished course" a property of
one function rather than a convention.
