# SmartReco — 3-minute demo recording script

Follow this top to bottom in one take, no editing required.

**Setup before hitting record:** run the app (`make run` or `uv run uvicorn app.main:app
--reload --port 8000`), then open `http://127.0.0.1:8000` in a fresh private/incognito
window — so the recommendation built in Scene 3 is visibly driven by what you do on
camera, not stale history. Maximize the browser. Start screen capture (Win+G, or OBS).

## Scene 1 — Browse the catalogue (0:00–0:25)

1. Land on `/` — narrate: *"SmartReco is a behavioural recommendation agent over a
   35-course catalogue."*
2. Click a category chip (e.g. "AI & ML"), then a tier chip ("beginner") — show the
   filters combine.
3. Point out the **Courses / My picks** nav on the top right.

## Scene 2 — Generate real behavioural signal (0:25–1:00)

This is the part that makes Scene 3 honest — you're building the profile live.

1. Search for something specific, e.g. "machine learning" — narrate: *"Every search,
   click, and dwell gets tracked."*
2. Click into 2–3 course detail pages from the results. On one, stay on the page ~10
   seconds (dwell time is a real signal — see `EVENT_WEIGHTS` in `app/services/profile.py`).
3. Add one to your wishlist / cart if that action exists on the product page.

## Scene 3 — Sign in and get the recommendation (1:00–1:40)

1. Go to `/login`, sign in as `learner@smartreco.dev` / `learner12345`.
2. Open `/me` — narrate: *"The agent reads that behaviour and writes a recommendation
   grounded only in the real catalogue."*
3. **Point at the "Why these?" evidence panel** — narrate: *"These are the only facts the
   model was allowed to cite — nothing invented."* This is the single most
   differentiating screen in the app; linger here 5–8 seconds.

## Scene 4 — Admin operations (1:40–2:30)

1. Sign out, log back in as `admin@smartreco.dev` / `admin12345`.
2. Point out the top-right **account chip** — avatar with the shield badge, name, Sign
   out grouped separately from **Courses / My picks / Operations** nav.
3. Click **Operations** → narrate over the dashboard:
   - *"SQL↔vector sync health"* — point at the "in sync" badge.
   - *"LLM call efficiency — the trigger policy decided most requests didn't need a
     model call at all"* — point at the "% avoided" stat.
   - *"The agent graph, read live from the compiled LangGraph — this can't drift from
     what actually runs."* — hover the node boxes (Analyze → Plan → Retrieve/Coldstart →
     Grade → Refine → Generate → Verify → Finalize).
4. Click **Agent runs** tab — narrate: *"Every run's node path, grade score, refine
   loops, tokens and cost — the full decision trace."* Click into one run.
5. Click **Catalogue** tab → **Add course** — narrate: *"Admins manage the catalogue
   directly; publishing triggers a dual-write to the vector index."*

## Scene 5 — Close (2:30–2:50)

1. Back to `/admin`, click **"Run digest now"** — narrate: *"The daily digest job,
   runnable on demand for the demo."*
2. End on the `/me` recommendation screen — narrate: *"Grounded retrieval, an
   LLM-as-judge grading loop, a groundedness verifier, and a deterministic fallback if
   anything fails — the agent never ships an ungrounded answer."*
