"""The single door to Mesh API. Every LLM and embedding call in this app goes through
here, which is what makes the budget cap, the circuit breaker and the token accounting
actually enforceable rather than aspirational.

Three behaviours here exist because of things measured against the live gateway, not
because they seemed like good ideas:

  1. Reasoning models emit hidden reasoning tokens before any visible text. kimi-k3
     spent 220 of 396 completion tokens reasoning. Cap output too low and you get a
     perfectly successful 200 response containing "". Hence a high default ceiling and
     an explicit EmptyCompletion error instead of a confusing downstream parse failure.
  2. kimi-k3 rejects `temperature` unless it is exactly 1, with a 400. Rather than
     hardcoding a model quirk list that rots, we strip the offending parameter on 400
     and remember it per model for the process lifetime.
  3. Several models wrap JSON in ```json fences even when asked for json_object.
     Observed on claude-haiku-4.5 and on kimi-k3 without response_format.
"""

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import date

from openai import (
    APIConnectionError,
    APIStatusError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

from app import metrics
from app.config import settings

log = logging.getLogger(__name__)


class MeshError(RuntimeError):
    pass


class EmptyCompletion(MeshError):
    """200 OK with no visible content — almost always the reasoning-token trap."""


class BudgetExceeded(MeshError):
    pass


class CircuitOpen(MeshError):
    pass


# USD per 1M tokens (input, output). Used for the budget guard and the cost column in
# the admin dashboard — an estimate for control, not an invoice. Unknown models fall
# back to a deliberately pessimistic rate so a surprise model cannot quietly overspend.
PRICING: dict[str, tuple[float, float]] = {
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-4o": (2.50, 10.00),
    "openai/gpt-5-mini": (0.25, 2.00),
    "anthropic/claude-haiku-4.5": (1.00, 5.00),
    "google/gemini-2.5-flash": (0.30, 2.50),
    "moonshotai/kimi-k3": (0.60, 2.50),
    "openai/text-embedding-3-small": (0.02, 0.0),
    "openai/text-embedding-3-large": (0.13, 0.0),
}
_UNKNOWN_PRICE = (3.00, 15.00)


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    inp, out = PRICING.get(model, _UNKNOWN_PRICE)
    return (prompt_tokens * inp + completion_tokens * out) / 1_000_000


@dataclass
class LLMResult:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0


@dataclass
class _Budget:
    """Process-local daily spend guard. Resets on date rollover."""

    day: date = field(default_factory=date.today)
    spent_usd: float = 0.0
    calls: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def check_and_reserve(self, limit: float) -> None:
        with self.lock:
            today = date.today()
            if today != self.day:
                self.day, self.spent_usd, self.calls = today, 0.0, 0
            if self.spent_usd >= limit:
                raise BudgetExceeded(f"daily LLM budget ${limit:.2f} exhausted (spent ${self.spent_usd:.4f})")
            self.calls += 1

    def record(self, cost: float) -> None:
        with self.lock:
            self.spent_usd += cost

    def snapshot(self) -> dict:
        with self.lock:
            return {"day": str(self.day), "spent_usd": round(self.spent_usd, 6), "calls": self.calls}


@dataclass
class _Breaker:
    """Stop hammering a gateway that is already failing. Opens after `threshold`
    consecutive failures, then lets exactly one request through to probe recovery."""

    threshold: int = 5
    reset_after: float = 60.0
    failures: int = 0
    opened_at: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def before(self) -> None:
        with self.lock:
            if self.opened_at and time.time() - self.opened_at < self.reset_after:
                raise CircuitOpen("Mesh circuit breaker is open")

    def success(self) -> None:
        with self.lock:
            self.failures, self.opened_at = 0, 0.0

    def failure(self) -> None:
        with self.lock:
            self.failures += 1
            if self.failures >= self.threshold:
                self.opened_at = time.time()
                log.error("mesh circuit opened", extra={"failures": self.failures})

    @property
    def is_open(self) -> bool:
        return bool(self.opened_at) and time.time() - self.opened_at < self.reset_after


_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def extract_json(text: str) -> dict:
    """Parse a JSON object out of a model response that may be fenced or padded."""
    cleaned = _FENCE.sub("", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise MeshError(f"no JSON object in response: {text[:200]!r}")
    return json.loads(cleaned[start : end + 1])


def _traced_client(client: OpenAI) -> OpenAI:
    """Return the client wrapped for LangSmith, or unchanged when tracing is off.

    Imported here rather than at module scope: app.observability reaches back into
    app.db, and mesh must stay importable by anything.
    """
    from app.observability import configure

    if not configure()["langsmith"]:
        return client
    from langsmith.wrappers import wrap_openai

    return wrap_openai(client)


class MeshClient:
    def __init__(self) -> None:
        self._client: OpenAI | None = None
        self._budget = _Budget()
        self._breaker = _Breaker()
        # Parameters a given model has rejected, learned at runtime: {model: {param:
        # replacement or None}}. A replacement is carried forward rather than just
        # dropped, because `max_tokens` is renamed rather than unsupported and losing it
        # would silently discard the output ceiling on every later call.
        self._unsupported: dict[str, dict[str, str | None]] = {}

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            if not settings.meshapi_api_key:
                raise MeshError("MESHAPI_API_KEY is not set")
            client = OpenAI(
                base_url=settings.meshapi_base_url,
                api_key=settings.meshapi_api_key,
                timeout=settings.meshapi_timeout_seconds,
                max_retries=0,  # we retry ourselves so the breaker sees each failure
            )
            # Mesh traffic goes out through the raw OpenAI SDK, not a LangChain model, so
            # LangSmith is blind to it: without this the graph trace is chain nodes with
            # no prompts and no token counts under them. The wrapper nests each call
            # inside whichever node is running. (Logfire sees the same calls via
            # instrument_openai; this is the LangSmith half.)
            self._client = _traced_client(client)
        return self._client

    @property
    def available(self) -> bool:
        return settings.has_llm and not self._breaker.is_open

    @property
    def budget_remaining_usd(self) -> float:
        return max(0.0, settings.llm_daily_budget_usd - self._budget.snapshot()["spent_usd"])

    def status(self) -> dict:
        return {
            "enabled": settings.llm_enabled,
            "key_present": bool(settings.meshapi_api_key),
            "circuit_open": self._breaker.is_open,
            "model": settings.meshapi_model,
            "model_fast": settings.meshapi_model_fast,
            "budget": self._budget.snapshot(),
            "budget_limit_usd": settings.llm_daily_budget_usd,
        }

    def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = 0.7,
        attempts: int = 3,
    ) -> LLMResult:
        model = model or settings.meshapi_model
        if not settings.has_llm:
            raise MeshError("LLM disabled or key missing")

        self._breaker.before()
        self._budget.check_and_reserve(settings.llm_daily_budget_usd)

        params: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens or settings.llm_max_output_tokens,
        }
        if temperature is not None:
            params["temperature"] = temperature
        if json_mode:
            params["response_format"] = {"type": "json_object"}
        for rejected, replacement in self._unsupported.get(model, {}).items():
            value = params.pop(rejected, None)
            if replacement is not None and value is not None:
                params[replacement] = value

        # Only transient failures spend the attempt budget. Dropping a rejected parameter
        # re-sends a payload the gateway has not seen yet, so charging it as a retry used
        # to leave a quirky model one real attempt out of three. The loop still terminates:
        # _drop_rejected_param pops the offending key, so each parameter can fire once.
        last_error: Exception | None = None
        failures = 0
        while failures < attempts:
            started = time.perf_counter()
            try:
                resp = self.client.chat.completions.create(**params)
            except BadRequestError as e:
                # A rejected parameter is deterministic: retrying unchanged is pointless.
                dropped = self._drop_rejected_param(model, params, str(e))
                if not dropped:
                    self._breaker.failure()
                    raise MeshError(f"Mesh rejected request: {e}") from e
                log.warning("mesh param dropped", extra={"model": model, "param": dropped})
                continue
            except (RateLimitError, APIConnectionError, APIStatusError) as e:
                last_error = e
                failures += 1
                self._breaker.failure()
                if failures >= attempts:
                    break
                time.sleep(0.5 * (2 ** (failures - 1)))
                continue

            self._breaker.success()
            return self._to_result(resp, model, started)

        raise MeshError(f"Mesh call failed after {attempts} attempts: {last_error}") from last_error

    def _drop_rejected_param(self, model: str, params: dict, message: str) -> str | None:
        """Return the parameter we removed, or None if the 400 was something else."""
        low = message.lower()
        for param in ("temperature", "response_format", "max_tokens"):
            if param in low and param in params:
                # Newer reasoning endpoints renamed max_tokens rather than dropping it.
                replacement = "max_completion_tokens" if param == "max_tokens" else None
                value = params.pop(param)
                if replacement is not None:
                    params[replacement] = value
                self._unsupported.setdefault(model, {})[param] = replacement
                return param
        return None

    def _to_result(self, resp, model: str, started: float) -> LLMResult:
        usage = resp.usage
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        details = getattr(usage, "completion_tokens_details", None)
        reasoning = getattr(details, "reasoning_tokens", 0) or 0 if details else 0

        cost = estimate_cost(model, prompt_tokens, completion_tokens)
        self._budget.record(cost)
        metrics.inc("smartreco_llm_calls_total", model=model)
        metrics.inc("smartreco_llm_tokens_total", prompt_tokens + completion_tokens, model=model)
        metrics.inc("smartreco_llm_cost_usd_total", cost, model=model)

        text = (resp.choices[0].message.content or "").strip()
        latency_ms = int((time.perf_counter() - started) * 1000)
        if not text:
            raise EmptyCompletion(
                f"{model} returned no content ({reasoning} reasoning tokens of "
                f"{completion_tokens}); raise llm_max_output_tokens"
            )

        log.info(
            "mesh call",
            extra={
                "model": model,
                "tokens": prompt_tokens + completion_tokens,
                "reasoning": reasoning,
                "cost_usd": round(cost, 6),
                "ms": latency_ms,
            },
        )
        return LLMResult(
            text=text,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning,
            cost_usd=cost,
            latency_ms=latency_ms,
        )

    def chat_json(self, messages: list[dict], **kwargs) -> tuple[dict, LLMResult]:
        result = self.chat(messages, json_mode=True, **kwargs)
        return extract_json(result.text), result

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._breaker.before()
        self._budget.check_and_reserve(settings.llm_daily_budget_usd)
        model = settings.meshapi_embedding_model
        try:
            resp = self.client.embeddings.create(model=model, input=texts)
        except (RateLimitError, APIConnectionError, APIStatusError, BadRequestError) as e:
            self._breaker.failure()
            raise MeshError(f"embedding call failed: {e}") from e

        self._breaker.success()
        tokens = getattr(resp.usage, "prompt_tokens", 0) or 0
        cost = estimate_cost(model, tokens, 0)
        self._budget.record(cost)
        metrics.inc("smartreco_llm_calls_total", model=model)
        metrics.inc("smartreco_llm_tokens_total", tokens, model=model)
        metrics.inc("smartreco_llm_cost_usd_total", cost, model=model)
        return [d.embedding for d in resp.data]


mesh = MeshClient()
