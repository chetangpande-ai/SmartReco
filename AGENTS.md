# Working in this repository

## Coding style

This project follows **Andrej Karpathy's engineering philosophy**, written out in full at
[`.claude/skills/karpathy-coding-style/SKILL.md`](.claude/skills/karpathy-coding-style/SKILL.md).
Read it before writing code here. The short version:

> Code is a liability. Write the simplest thing that fully solves the problem, keep it
> flat and readable top-to-bottom, delete before you add, and never wrap something in a
> `try/except` unless you have a specific useful response to the failure.

The rules that bite most often in this codebase:

- **No speculative abstraction.** An interface needs two real implementations. The
  `VectorStore` protocol exists because Chroma *and* Pinecone are both implemented; the
  notifier protocol exists because SMTP, file and SES are all real.
- **Comments say *why*.** `# kimi-k3 rejects temperature != 1` is worth keeping. A
  comment restating the code is not.
- **Defensive catches hide bugs.** This repo has already been bitten once: a broad
  `except` around both the database write and its metrics turned a metrics bug into a
  fake "event batch write failed" error. Scope `try` blocks to the thing that can fail.

## Project conventions

- **Every model call goes through `app/services/mesh.py`.** That single choke point is
  what makes the budget cap, circuit breaker and token accounting enforceable rather
  than decorative. Do not call the OpenAI SDK directly anywhere else.
- **Datetimes are UTC everywhere.** Use `models.utcnow()` to write and
  `models.ensure_utc()` when reading a column back — SQLite returns naive datetimes
  where Postgres returns aware ones, and mixing them raises only on SQLite.
- **The vector index contains exactly the published products.** Never add an
  `is_published` filter at query time; unpublishing enqueues a delete instead.
- **Tests are offline.** `LLM_ENABLED=false` selects the hashing embedder and template
  copy writer. A test that needs the network is a test that will not run in CI.

## Before opening a PR

```bash
make lint && make test
```

Both must be clean. `pytest` runs without an API key and spends nothing.
