---
name: karpathy-coding-style
description: Andrej Karpathy's engineering philosophy encoded as working rules — write the simplest code that fully solves the problem, keep it flat and readable top-to-bottom, delete before you add, avoid speculative abstraction and defensive clutter. Load this before writing or reviewing any code in this repo.
---

# Karpathy Coding Style

Rules distilled from Andrej Karpathy's publicly stated engineering philosophy (nanoGPT,
micrograd, llm.c, minGPT, and his writing on code quality). This is an interpretation
for day-to-day work, not a verbatim document.

The governing idea:

> **Code is a liability, not an asset. The best code is no code. The second best is
> code so simple you can hold all of it in your head at once.**

---

## 1. Simplicity is the whole job

- Write the simplest thing that *fully* solves the problem. Not the cleverest, not the
  most general, not the most "extensible."
- If you cannot explain what a function does in one sentence, it does too much.
- Prefer boring, obvious code. Cleverness is a tax the next reader pays with interest.
- **Line count is a real cost.** Fewer lines that do the same thing is almost always
  better — as long as density never becomes obscurity. Compressed ≠ cryptic.

## 2. Delete before you add

- The first move when touching code is to look for something to remove.
- Dead code, unused parameters, unreachable branches, "we might need this later" hooks —
  delete on sight. Version control remembers.
- A refactor that removes 200 lines and adds 50 is a better day's work than one that
  adds 200.
- Never leave commented-out code. Delete it.

## 3. No speculative generality

- Build for the case you have **right now**. Not the three hypothetical ones.
- Do not add a config option, a strategy interface, a plugin registry, or a base class
  until there are **two real, concrete** implementations demanding it. One
  implementation behind an interface is pure overhead.
- YAGNI is not a suggestion. Most "flexibility" is never used and always costs
  readability.
- **Exception worth naming:** an abstraction is justified when it is the *thing being
  tested* or the *thing being swapped in an outage*. A vector-store protocol with a real
  second implementation earns its keep. A `BaseServiceFactoryProvider` does not.

## 4. Flat over nested

- Minimize indirection. Every hop the reader has to follow is a chance to lose the plot.
- Prefer one readable 60-line function over six 10-line functions that only ever call
  each other in a fixed order. Over-decomposition hides control flow.
- Avoid deep inheritance. Composition, plain functions, and plain data beat class
  hierarchies almost every time.
- Early-return to keep the happy path at the leftmost indentation level.
- A file you can read top-to-bottom and understand completely is the goal.

## 5. Explicit over implicit

- No magic. No metaclass tricks, no decorator stacks that rewrite behavior, no
  action-at-a-distance via globals.
- Pass dependencies in as arguments. Implicit shared state is where bugs live.
- Name things for what they *are*. `events_since_last_rec` beats `n`. But `i` is fine
  for a loop index — don't pad short-lived locals with ceremony.

## 6. Don't be defensive for its own sake

- Do **not** wrap everything in `try/except`. Swallowing exceptions hides bugs and turns
  a loud, findable crash into a silent wrong answer.
- Catch an exception only when you have a **specific, useful** response to it: a retry, a
  fallback with real value, or a boundary where a user must see a clean error.
- Never `except Exception: pass`. If you truly must swallow, log it with the reason.
- Let programmer errors crash. A traceback in development is a gift.
- Validate at the boundaries of the system (HTTP input, external API responses), then
  trust your own types internally.

## 7. Comments explain *why*, never *what*

- The code already says what it does. If it doesn't, fix the code, don't add a comment.
- Good comments capture intent, a non-obvious constraint, a measured tradeoff, or a
  landmine: *"kimi-k3 rejects temperature != 1"*, *"decay half-life 48h — tuned so a
  single browsing session doesn't dominate a week of history."*
- Delete comments that have drifted out of sync. A wrong comment is worse than none.
- No banner art, no `# ---- END OF SECTION ----` noise, no docstrings that restate the
  signature.

## 8. Dependencies are a long-term debt

- Every dependency is someone else's bugs, breaking changes, and supply chain.
- Do not pull in a library for something the standard library does in ten lines.
- Do pull in a library for something genuinely hard and well-solved (an HTTP server, a
  DB driver, a crypto primitive). Never roll your own crypto.
- Pin versions. Reproducibility beats novelty.

## 9. Data structures first

- Get the data model right and the code mostly writes itself. Bad schema → every
  function downstream is fighting it.
- Prefer plain dicts, lists, tuples, and dataclasses over bespoke wrapper classes.
- Make illegal states unrepresentable where it's cheap to do so.

## 10. Know your hot paths, ignore the rest

- Do not micro-optimize code that runs once at startup.
- **Do** think hard about the code that runs on every request, every event, every token.
- Measure before optimizing. Then optimize the algorithm, not the constant factor.
- Batching is usually the biggest win available. Look for it first.

## 11. Working > perfect, but finished > working

- Get an end-to-end path working, then improve it. Never build three perfect layers that
  have never once run together.
- But "it runs on my machine once" is not done. Done means seeded, tested, documented,
  and reproducible by someone else.

## 12. Naming and layout

- Modules named for what they contain, lowercase, no `utils.py` dumping ground.
- Consistency inside a file beats consistency with an external style guide.
- Format mechanically (`ruff`), never argue about it, never hand-align.

---

## Checklist before calling code done

1. Can a competent stranger read this file top-to-bottom and understand it?
2. Is there anything I can **delete** without losing behavior?
3. Is any abstraction here serving exactly one caller? Inline it.
4. Does every `try/except` have a specific, useful response — or is it hiding a bug?
5. Does every comment say *why*, and is it still true?
6. Is the hot path batched, and is the cold path left alone?
7. Would I be happy to be handed this file at 2am during an outage?

## Anti-patterns — reject on sight

| Smell | Do instead |
|---|---|
| `AbstractBaseManagerFactory` | A function |
| `try: ... except Exception: pass` | Let it crash, or handle one specific error |
| `utils.py` with 40 unrelated helpers | Put each next to its only caller |
| Config option with one possible value | Hardcode it |
| Interface with one implementation | Delete the interface |
| Comment restating the code | Delete the comment |
| Six one-line functions called in sequence | One readable function |
| `**kwargs` passed through four layers | Named parameters |
| Deep `if/else` pyramid | Early returns |
