# Design decisions

Why this project is built the way it is, with the evidence behind each choice. Moved out
of [README.md](../README.md) so the README stays scannable — nothing here is summarised
there, so this is the only place these arguments live. Exact counts live in the README
and [CLAUDE.md](../CLAUDE.md) rather than being restated here.


## Model selection — measured, not assumed

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

## Retrieval floor — swept, not guessed

A kNN index returns *k* results whether or not any are near. With `k=40` over a
35-course catalogue that is nearly everything, so those ranks are noise that dilutes rank
fusion. Sweeping the floor over 20 paraphrase probes (`uv run python scripts/eval_retrieval.py --sweep`):

| ratio | recall@1 | recall@5 | MRR | avg pool |
|---|---|---|---|---|
| 0.00 | 0.95 | 1.00 | 0.975 | 32.8 |
| 0.35 | 0.95 | 1.00 | 0.975 | 26.0 |
| 0.50 | 0.95 | 1.00 | 0.975 | 13.3 |
| **0.55** | **0.95** | **1.00** | **0.975** | **10.0** |
| 0.60 | 0.95 | 1.00 | 0.975 | 7.5 |
| 0.80 | 0.95 | 1.00 | 0.975 | 1.9 |

Quality is **flat** across the entire range: the fused ranking is already right, so the
floor is not a quality knob here at all — it decides how much noise MMR has to
diversify over. `retrieve` asks for 8 candidates, so `0.55` (pool 10) leaves a real
choice while `0.60` and tighter starves it below what was requested. Tuned on one
catalogue of 35 items — re-run `uv run python scripts/eval_retrieval.py --sweep` if yours differs.

**The probes needed fixing before the numbers meant anything.** A first draft described
teaching styles with no subject — *"get something working in week one and explain it
afterwards"* — and scored recall@1 **0.50**. Vector-only inspection showed the best hit
at similarity 0.29: a sentence about pedagogy matches every course equally. That was
measuring the probes, not the retrieval. Rewritten to name the subject while still
avoiding the course's title words, recall@1 went to **0.95**.

## RRF fuses by rank, never by score

Cosine lives in `[-1,1]`; BM25 is unbounded and corpus-dependent. Normalising them into
a weighted sum means inventing an exchange rate that shifts with every catalogue edit.
Tested: a vector hit scoring `999999` and a lexical hit scoring `0.0001` produce
**identical** fused scores — both are rank 1.

## Guardrails, because the model is being asked to persuade

Invented discounts and manufactured scarcity are not hypothetical failure modes here —
they are the *most likely* ones, because persuasive training data is full of them. The
deterministic rails block outcome promises, fabricated urgency, invented discounts,
**any price not in the catalogue**, unsupported superlatives, and PII. They run always,
free, offline and in CI.

**Real courses raise a second risk.** A model recognises the Deep Learning
Specialization and will happily recite a syllabus from training data that may be wrong or
years stale. The generation prompt therefore states that catalogue facts are the *only*
facts it may use, and the `spec` field exists so there is always something accurate to
quote.

**Education invites a specific set of inventions.** The catalogue has a title, provider,
track, level, price, rating, tags and a syllabus line — no accreditation, no placement
data, no cohort dates, no completion statistics. So "job guarantee", "93% of graduates
find work", "accredited", "recognised by employers", "seats are filling", "1-on-1
mentoring" are unsupported *by construction*, and every one is a phrase a model reaches
for unprompted, because its training data on course marketing is saturated with them.

The reverse error matters just as much. One syllabus line genuinely says **"lifetime
access"** — a rail on that phrase would reject the model for quoting the catalogue
correctly. The rails forbid only what the catalogue cannot support, never what it does,
and a test pins both directions.

## Evals, prompt versioning and red-teaming — measured, not assumed

Guardrails prove *known* bad patterns are blocked. They don't answer "is this prompt
actually behaving," which needs a judge, or "does this hold up against someone trying to
break it," which needs an attacker. Both were added, both run for real, not just wired.

**DeepEval, not DeepEval-and-RAGAs.** `GEval` — DeepEval's implementation of the
custom-rubric-as-LLM-judge pattern — is also where DeepEval's own RAG metrics
(`FaithfulnessMetric`, `AnswerRelevancyMetric`) live, so one dependency covers both
"measure faithfulness the way RAGAs would" and "grade against our own rubric" instead of
two. Routed through `MeshDeepEvalModel` (`app/services/eval_llm.py`), so an eval run
still hits the same budget cap and `/metrics` counters a real recommendation would.

The custom rubric's five criteria are copied from `GENERATE_SYSTEM`'s HARD RULES, on
purpose — when the prompt changes, the rubric is what says whether the change helped.
One real run, three scenarios, `gpt-4o-mini` on both sides:

```
ml_progression:    Faithfulness 1.00   AnswerRelevancy 1.00   grounded_persuasion 0.40
web_fundamentals:  Faithfulness 0.88   AnswerRelevancy 0.78   grounded_persuasion 0.30
data_engineering:  Faithfulness 1.00   AnswerRelevancy 1.00   grounded_persuasion 0.60
```

`web_fundamentals` scoring lower is the harness working: the judge's reason named a
specific unsupported claim in the model's narrative — the same failure class `verify()`
catches downstream, caught one stage earlier.

**Red-teaming targets the real attack surface, not a hypothetical one.** A search query
is quoted verbatim into `evidence()`, which is quoted verbatim into the prompt — so a
search box is a prompt-injection vector. `scripts/red_team.py` runs four probes (injection
via search query, a fake system-role marker, an injected catalogue description, a
roleplay jailbreak) through the real `generate → verify → finalize` path with a real
model. **4/4 blocked** on `gpt-4o-mini` — not because the model always refuses, but
because `verify()`/`guardrails` catch whatever gets through either way. Hermetic versions
of the same idea (`tests/test_agent.py::TestAdversarialGeneration`, expanded
`TestGuardrails` cases) stub the model response and run in every CI build, no key needed.

**PII had an input-side gap.** The output-side check (`guardrails.check_copy`) always
existed; nothing scrubbed a user's own search text before it reached a prompt. Typing an
email into search flowed unredacted through `profile.evidence()` into the model's
context — the output check would only have caught it if the model happened to echo it
back. `app/services/pii.py` closes that at the point raw text becomes prompt content, not
at ingestion, so a user's own event history stays intact.

Full writeup: [`evals.md`](evals.md).

## Recommending fewer than four

Retrieval always returns a full slate, so a narrow interest gets padded with whatever
fused in behind it. An A/B run caught it live on the previous catalogue: a fitness
shopper's fourth pick was a video doorbell, reasoned as *"in case you're looking for more
tech at home"*. Honest, grounded, and exactly what makes a recommender feel generic.

The generation prompt now asks for *up to* `REC_TOP_K` and says plainly that three
courses someone would actually take beats four with a filler. Verified on the current
catalogue: the career-switching learner was shown **two** courses, because only two
genuinely fit a beginner moving into data.

## Observability: two views, one run

Two backends, because they answer different questions and neither is complete alone.
**LangSmith** shows the graph — which node ran, what the model was actually asked, why
`verify` rejected a draft. **Logfire** shows the request that graph was serving — the
HTTP span, the SQL underneath it, the Mesh call and its token count, on one timeline.

They are joined rather than duplicated. Setting `LANGSMITH_TRACING_MODE=hybrid` makes
the LangSmith SDK write each run to LangSmith *and* emit it as an OpenTelemetry span,
which Logfire receives because Logfire owns the global tracer provider. One run, two
places — **both verified live against real projects**, with the LangSmith SDK confirmed
to be writing into `logfire._internal.tracer._ProxyTracer` rather than one of its own.
The whole graph arrives, refine loops, guardrail repair and all:

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

