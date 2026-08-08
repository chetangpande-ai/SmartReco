# SmartReco — 4-minute demo recording script

Follow this top to bottom in one take, no editing required.

**Setup before hitting record:** run the app (`make run` or `uv run uvicorn app.main:app
--reload --port 8000`), then open `http://127.0.0.1:8000` in a fresh private/incognito
window — so the recommendation built in Scene 3 is visibly driven by what you do on
camera, not stale history. Maximize the browser. Start screen capture (Win+G, or OBS).

Run `make seed` first if the catalogue is empty, and `make migrate` if you have not
applied the career-layer schema.

## Scene 1 — The career pitch (0:00–0:50)

This is the differentiator. Lead with it.

1. Land on `/` — narrate: *"A career learning marketplace: 66 courses across 21 tracks,
   and an AI advisor that works out which of them you should take, in what order."*
   Point at the banner: **"Where do you want your career to go?"**
2. Click **Build my career path with AI**, then the
   **"Load the QA → AI Engineer example"** link (or go straight to `/career?demo=1`).
   Narrate the inputs as they fill: *"Ten years in QA. Java, Selenium, API testing.
   Wants to move into AI engineering."*
3. Hit **Build my career path with AI**. Scroll to the roadmap.
4. **Linger on the first two stages** — this is the whole argument:
   - *"It knows Selenium and API testing mean this person has been testing for a living,
     so it does not put 'learn testing' in their gap list."*
   - *"The eight steps are ordered so each course only assumes what the one before it
     taught — Python, then GenAI, then RAG, then agents. Nothing here was chosen by a
     language model; it is computed from the skill graph. The model wrote the paragraph."*
5. Scroll on through **Projects → Assessment → Certification → Interview preparation →
   Target role.** Narrate: *"From where you are, to where you want to be."*

## Scene 2 — Browse the catalogue (0:50–1:15)

1. Click **Explore** — narrate: *"The taxonomy underneath: 21 categories, 628 skills."*
   Click into a category, then a skill (e.g. RAG) — point at **"Roles that ask for this"**.
   *"Every skill page is also a career page."*
2. Click **Explore courses** and stack two or three filters — category, career role,
   free. Narrate: *"Filters are links, so any combination is a shareable URL."*
3. Open a course. Click **"Ask AI if this course is right for me"** — narrate:
   *"It is willing to say no. The prerequisite check is computed, not guessed."*

## Scene 3 — Generate real behavioural signal (1:15–1:50)

This is the part that makes the next scene honest — you're building the profile live.

1. Search for something specific, e.g. "machine learning" — narrate: *"Every search,
   click, and dwell gets tracked."*
2. Click into 2–3 course detail pages from the results. On one, stay on the page ~10
   seconds (dwell time is a real signal — see `EVENT_WEIGHTS` in `app/services/profile.py`).
3. Add one to your wishlist / cart if that action exists on the product page.

## Scene 4 — Sign in and get the recommendation (1:50–2:30)

1. Go to `/login`, sign in as `learner@smartreco.dev` / `learner12345`.
2. Open `/me` — narrate: *"The agent reads that behaviour and writes a recommendation
   grounded only in the real catalogue."*
3. Point at the **career tile** top-right of the greeting: *"Three of the nine skills
   AI Engineer asks for — and it moves as they finish courses, because a completed
   course is the one claim on this page we can actually verify."*
4. **Open the "Why these?" panel at the bottom** — narrate: *"These are the only facts
   the model was allowed to cite — nothing invented."* Linger here 5–8 seconds.

## Scene 5 — Admin operations (2:30–3:20)

1. Sign out, log back in as `admin@smartreco.dev` / `admin12345`.
2. Point out the top-right **account chip** — avatar with the shield badge, name, Sign
   out grouped separately from the **Explore / Careers / My learning** nav.
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

## Scene 6 — Close (3:20–3:50)

1. Back to `/admin`, click **"Run digest now"** — narrate: *"The daily digest job,
   runnable on demand for the demo."*
2. End on the `/career` roadmap screen — narrate: *"Grounded retrieval, an
   LLM-as-judge grading loop, a groundedness verifier, and a deterministic fallback if
   anything fails — the agent never ships an ungrounded answer."*
