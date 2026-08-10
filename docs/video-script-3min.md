# SmartReco — 3-minute product video

A **story** video: the product, the two engines, the architecture, and why it can be
trusted. Different job from [`demo-script.md`](demo-script.md), which is a 4-minute
click-through of the running app — that one is the *capture guide* for the screen footage
this script cuts together.

**The through-line, one sentence:**

> Any platform can show you a catalogue. This one tells you which course is next — and
> shows its work.

Every scene either sets that up or pays it off. If a shot does neither, cut it.

**Budget:** 180 seconds, ~450 words of narration at 150 wpm. That is the whole allowance
and it is smaller than it feels. Write to the word counts below, not to the seconds.

## Pre-flight — do this before you record a single frame

**Sign in first, then build the career profile.** Not the other way round, and not
skipped. Everything the video claims about the career engine lives in a `CareerProfile`
row; without one, `/careers/{slug}` renders `advisor.preview()` — the same roadmap with an
*anonymous* profile — and the page shows the opposite of the story:

| `/careers/qa-to-ai` | No profile | With the QA profile |
|---|---|---|
| Readiness | *(none)* | **33%** |
| Where you are | **0 — empty** | 9 chips, marked `transfers` |
| Skills to learn | **15, opening with `Testing · to learn`** | 8, and **no** Testing |

Read that bottom-left cell again. Signed out, the roadmap tells a ten-year QA engineer to
learn testing — which is the precise failure Scene 3 says this product avoids. Record in
that state and the screen contradicts the voiceover.

So, in order, before recording:

1. `make seed` if the catalogue is empty, `make migrate` if the career tables are new.
2. Open a fresh private window. Sign in as `learner@smartreco.dev`.
3. **Then** go to `/career?demo=1`. Signed out the form still pre-fills, but the submit
   button reads *"Sign in to build your path"* and cannot be submitted — this is the trap
   that looks like the demo link is broken.
4. Submit. Confirm `/careers/qa-to-ai` now reads 33% with a populated "Where you are".
5. Only now start recording.

Optionally set `MESHAPI_MODEL=kimi-k3` — slower, better copy, and Scene 4 holds on that
copy long enough to read it.

## Sequence

| # | Scene | Time | Words | Job |
|---|---|---|---|---|
| 1 | Cold open — the problem | 0:00–0:18 | 45 | Earn the next 20 seconds |
| 2 | What it is — two engines | 0:18–0:38 | 50 | Frame the product |
| 3 | **Flow A — the career advisor** | 0:38–1:18 | 105 | The differentiator. Longest scene on purpose |
| 4 | Flow B — behavioural recommendation | 1:18–1:48 | 75 | The other engine, and grounding made visible |
| 5 | Architecture | 1:48–2:22 | 85 | Show the engineering |
| 6 | Trust — guardrails and operations | 2:22–2:50 | 70 | Pay off "shows its work" |
| 7 | Close | 2:50–3:00 | 25 | Land the through-line |

---

### Scene 1 — Cold open (0:00–0:18)

**Visual.** No logo, no title yet. Open mid-scroll on `/courses` — the marketplace grid
moving fast, sixty-six cards blurring past. It slows. It stops. Beat of silence.
Title card: **SmartReco**.

**VO.**
> Every learning platform has the same problem. Here is the catalogue — sixty-six
> courses. Which one is next, for *you*? Most platforms answer that with "people also
> viewed." This one answers it with a plan.

**Why.** The problem before the product. A viewer who has not felt the problem does not
care about the solution, and you have eighteen seconds to make them feel it.

---

### Scene 2 — What it is (0:18–0:38)

**Visual.** Land on `/` and hold on the banner: *"Where do you want your career to go?"*
Then a fast three-up of `/me`, `/career`, `/admin` — one second each, just to establish
that there is a whole product here.

**VO.**
> SmartReco is a career learning marketplace with an AI agent underneath it. It answers
> "what should I learn next" twice. Once from what you *do* — tracked quietly as you
> browse. And once from what you *say* — the role you are in, and the role you want.
> Two engines, deliberately different.

**Why.** "Two engines, deliberately different" is the line the rest of the video hangs
off. Say it once, clearly, and never explain it again — Scenes 3 and 4 *are* the
explanation.

---

### Scene 3 — Flow A: the career advisor (0:38–1:18)

The differentiator. Give it the most time of any scene and do not rush the pause.

**The advisor and the roadmap are two different pages.** Signed in (see Pre-flight above —
this scene is worthless without a profile), in this order:

1. **✦ AI Advisor** in the nav → `/career?demo=1`, or the *"Load the QA → AI Engineer
   example"* link on the page. This pre-fills the form boxes. It is the only auto-fill.
2. Press **Build my career path with AI**. The page returns to `/career#plan` and the plan
   renders **below the form** — scroll down; it does not navigate anywhere new.
3. Click **"full roadmap for this path"** in that plan → `/careers/qa-to-ai`. This is the
   eight-stage spine in `screenshots/career-roadmap.png`, and only now does it have the
   green chips.

The **"Popular moves"** list at the bottom of the advisor page links to the same roadmaps
directly. Handy for jumping straight there on a retake — but only *after* step 2 has been
done once, or you land on the empty version.

**Visual.**
1. `/career?demo=1` — the form filling in. Let the viewer read it.
2. Submit, then follow the click path above to `/careers/qa-to-ai`.
3. **Hold on stages 1 and 2 for a full six seconds.**
4. Speed-ramp 2× down the remaining six stages, ending on **Target role · AI Engineer**.

**Add in the editor, not in the app.** Draw a highlight box around *Selenium · transfers*
and *API Testing · transfers*, and a second one around the **Skills to learn** row with an
arrow pointing at the *absence* of "Testing." The app has no hover states or tooltips here
— `transfers`, `counts toward the role` and `to learn` are printed permanently beside each
skill chip (`career/_roadmap.html`, the `skill-note` span) in small grey type. On video
that type is too small to carry the point on its own, which is exactly what the overlays
are for.

**VO.**
> Ten years in QA. Java, Selenium, API testing. Target role: AI Engineer.
>
> *(beat)*
>
> Watch the first two stages. Selenium and API testing are marked as transferring — and
> "Testing" never appears in the gap list at all, because someone who lists Selenium has
> been testing for a living. Then six courses, ordered so each one only assumes what the
> one before it taught. Python, then GenAI, then agents.
>
> And here is the part that matters: no language model chose any of this. It is computed
> from a skill graph — six hundred and twenty-eight skills, twenty-two roles. The model
> only wrote the paragraph around it.

**Why.** This is the strongest eight seconds in the product. A negative — a thing the
system correctly *did not* say — is far more persuasive than any output, because it is
the failure mode every viewer has already suffered from a recommender.

---

### Scene 4 — Flow B: behavioural recommendation (1:18–1:48)

**Two traps in this scene.** Both cost a take if you meet them live:

- **"Why these?" is collapsed by default.** It is a `<details>` element
  (`dashboard.html`) — on screen it is a closed one-line summary, not an open panel. You
  have to **click it**. Miss that and the scene's whole point never appears.
- **`/me` can open on a spinner.** Generation runs in the background, and the page shows
  *"The agent is reading your activity…"* while it works. Land, let it resolve, *then*
  start talking. If nothing ever generates you did not produce enough activity — the gates
  want **≥5 events** and **90 seconds** since the last run. The **Refresh** button on the
  page forces it.

**Visual.**
1. Fast montage, ~8s: search *"machine learning"*, click two results, land on a course
   page and sit. Overlay the event names as they fire — `search`, `product_click`,
   `dwell`.
2. Cut to `/me`. Wait out the spinner. Recommendation cards with the per-card reason.
3. **Click** "Why these?" to expand, and hold five seconds. The money shot of the scene.

**VO.**
> The other engine just watches. Search, clicks, dwell time — captured in three
> milliseconds, which is less than the page takes to render. Open your dashboard, and the
> agent has read that behaviour and written a recommendation grounded only in the real
> catalogue.
>
> And this panel — "Why these?" — is the complete list of facts it was allowed to cite.
> Nothing else. Three learners with different histories were shown zero courses in common.

**Why.** Grounding is abstract until it is a panel on screen. Hold on it.

---

### Scene 5 — Architecture (1:48–2:22)

**Visual.** [`smartreco-overview.svg`](smartreco-overview.svg), revealed in layers rather
than shown all at once — browser, then ingest, then the agent, then the stores. Land on
the nine-node graph and **colour the two model-spending nodes differently from the other
seven**; that contrast is the whole point of the scene.

**VO.**
> Underneath: FastAPI, a bounded event queue, hybrid retrieval over a vector index, and a
> nine-node LangGraph agent — analyze, plan, retrieve, grade, generate, verify.
>
> Only two of those nine nodes spend money on a model. Everything else is computed. And
> before the agent runs at all, eleven gates ask whether this request is even worth a
> model call.
>
> Every model call in the system leaves through one file. That single choke point is what
> makes the budget cap and the circuit breaker real, instead of a promise.

**Why.** Do not narrate the diagram box by box — the viewer reads faster than you talk.
Say the *decisions*: two nodes of nine, eleven gates, one choke point. Three numbers, one
idea each.

---

### Scene 6 — Trust and operations (2:22–2:50)

**Visual.** `/admin`. Pan across sync health → the **% avoided** call-efficiency stat →
the compiled agent graph. The admin tabs run **Operations · Catalogue · Agent runs** in
that order. Then the **Agent runs** tab; open a run whose node path reads
`analyze → plan → retrieve → grade → refine → retrieve → grade → …` and let the viewer see
the loop. Text-on-screen for the three numbers as they are spoken.

**Do not restart the server before recording this scene.** The call-efficiency counters
are in-process and reset to zero on restart (README, Honest Limitations). Restart and the
panel reads no data instead of a percentage — the one number this scene is built around.
Warm the app up with a few page loads, then record.

**The refine-loop rows are real and already in the database** — several show two refines
across a thirteen-node path at roughly $0.0007. Pick one with `refines = 2` and let the
viewer read the whole path; that is the shot.

**VO.**
> And because it is being asked to *persuade*, it gets checked. Thirty-five deterministic
> rules block invented discounts, fake scarcity, job guarantees. A verifier drops any
> course the agent was never actually offered. Four prompt-injection attacks were run
> against the live path — four blocked.
>
> This is the operations view: every run's node path, tokens and cost. That one is a real
> refine loop. Seven hundredths of a cent.

**Alternate closing line, if the all-time figures are healthy on the day:** *"Fifty-six
recommendations, three and a half cents."* A cumulative number is a stronger argument than
a single run — check the "Spend all time" tile and use whatever it actually says.

**Why.** Ending the technical stretch on a cost number is deliberate — it says "this was
built by someone who would have to pay the bill," which is the impression you want to
leave.

---

### Scene 7 — Close (2:50–3:00)

**Visual.** Cut back to the QA→AI roadmap, pull out to the full page, hold. Logo.

**VO.**
> Any platform can show you a catalogue. This one tells you which course is next — and
> shows you its work.

---

## Production notes

**Capture.** 1920×1080, browser maximised. `make seed` and `make migrate` first. Set
`MESHAPI_MODEL=kimi-k3` for the recording — slower, noticeably better copy, and Scene 4
holds on that copy long enough to read it.

**Incognito does not reset the learner's history.** Worth knowing before you plan Scene 4
around it: events are keyed to the *user row*, not the browser, so a private window signs
into the same accumulated profile. The seeded learner already carries a substantial one —
searches, dwell times, eleven AI/ML course views — and `/me` renders fully populated on
first load, no spinner. Two options, both fine:

- **Use the seeded learner.** `/me` is instantly full, nothing to wait for. The Scene 4
  narration still holds — it says the agent *has read* the behaviour, not that the
  behaviour happened sixty seconds ago. Simplest, and what the script assumes.
- **Register a fresh account** at `/register` for a genuinely live build. More honest to
  the "watch it learn" framing, but you must generate ≥5 events, wait out the spinner, and
  rebuild the career profile before Scene 3 works.

Do not claim on camera that the profile was built in the last minute unless you took the
second option.

**Scene 6 needs the admin account.** `/admin` returns 403 for the learner; sign out and
back in as `admin@smartreco.dev` before recording it. Record Scene 6 separately from 3
and 4 for that reason alone.

**Use the screenshots as cutaways, not as the video.** The eight PNGs in
`docs/screenshots/` are already framed and current; they cover any beat where live capture
is fiddly. Live-capture only Scenes 3, 4 and 6, where *motion* is the argument.

**Speed-ramp the long scrolls.** The eight-stage roadmap and the marketplace grid are both
too long at 1×. 2× with a hold at the top and bottom of each.

**The invisible-agent problem.** The agent thinking has no natural visual. The admin
node-path row in Scene 6 is the answer — it is the only place the reasoning is literally
on screen. Do not try to animate the graph as a substitute.

**If you run long,** in this order: compress Scene 2 into Scene 1's tail (−12s), trim
Scene 5 to the graph layer only (−10s), drop the montage in Scene 4 to three seconds
(−5s). **Never** shorten the six-second hold in Scene 3 — if that beat does not land,
nothing else in the video matters.

**If you run short,** the best addition is the `/products/{slug}/fit` button answering
*"not yet — this assumes Python and LLM Fundamentals"*. A recommender willing to say no is
worth ten seconds anywhere after Scene 3.

## Asset map

| Scene | Asset |
|---|---|
| 1 | Live capture, `/courses` |
| 2 | `screenshots/home.png`, `dashboard.png`, `career-advisor.png`, `admin.png` |
| 3 | Live capture, `/career?demo=1` → submit → `/careers/qa-to-ai`. Fallback: `career-roadmap.png` |
| 4 | Live capture, search → `/me`. Fallback: `dashboard.png` |
| 5 | [`smartreco-overview.svg`](smartreco-overview.svg) |
| 6 | Live capture, `/admin` + `/admin/agent-runs`. Fallback: `admin.png` |
| 7 | `screenshots/career-roadmap.png` |

Numbers quoted in the narration live in [README.md](../README.md) and
[CLAUDE.md](../CLAUDE.md). If one changes there, it changes here — re-record the line
rather than leaving a stale figure in a video you cannot edit later.
