"""Guardrails on generated persuasive copy.

We are asking a model to *persuade* people to spend money. That is exactly the setting
where an unconstrained model invents a discount, manufactures scarcity, or promises an
outcome nobody can promise. None of those are hypothetical failure modes — they are the
most likely ones, because persuasive training data is full of them.

Two layers:

  * **Deterministic rails, always on, zero cost.** Regex and catalogue cross-checks.
    They cannot be talked out of it by a clever completion, they add no latency, and
    they run identically offline and in CI.
  * **NeMo Guardrails, optional.** Loaded only when `guardrails_llm_enabled` is set and
    the package is installed, for policy that genuinely needs a model to judge. Off by
    default because it costs tokens per generation.

The deterministic layer is the one that actually protects users. The LLM layer is a
refinement, not the floor.
"""

import logging
import re
from dataclasses import dataclass, field

from app.config import settings

log = logging.getLogger(__name__)


@dataclass
class GuardrailReport:
    violations: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def add(self, rule: str, detail: str) -> None:
        self.violations.append(f"{rule}: {detail}")


# Claims nobody selling a course can honestly make.
_FORBIDDEN = [
    (r"\bguarantee(d|s)?\b", "outcome guarantee"),
    (r"\brisk[- ]free\b", "risk-free claim"),
    (r"\bno[- ]brainer\b", "pressure language"),
    (r"\b100%\s*(success|effective|guaranteed|certain)", "absolute success claim"),
    (r"\bovernight\b", "unrealistic timeframe"),
    (r"\binstantl?y\s+(master|learn|become|earn)", "unrealistic timeframe"),
    (r"\b(double|triple|10x|5x)\s+(your\s+)?(salary|income|pay)\b", "earnings promise"),
    (r"\bget\s+hired\s+(guaranteed|immediately|instantly)", "employment promise"),
    (r"\bsecret\s+(that|the\s+\w+\s+don'?t\s+want)", "conspiracy framing"),
    # Real products invite comparative claims the catalogue cannot support. "Best fit
    # for you" is fine and stays; "best on the market" is an assertion about products
    # we hold no data on.
    (r"\bbest\s+(on\s+the\s+market|in\s+(the\s+)?(world|class)|available\s+anywhere)\b",
     "unsupported superlative"),
    (r"\b(unmatched|unbeatable|second\s+to\s+none)\b", "unsupported superlative"),
    (r"\bbeats?\s+(every|all|any)\s+\w+", "unsupported comparison"),
    (r"\bworld'?s\s+(best|fastest|finest)\b", "unsupported superlative"),
]

# Manufactured urgency and scarcity. The catalogue has no stock and no sales.
_FABRICATED_URGENCY = [
    (r"\b(only|just)\s+\d+\s+(seats?|spots?|places?|left|remaining)", "invented scarcity"),
    (r"\benrol+ment\s+closes?\b", "invented deadline"),
    (r"\blimited[- ]time\b", "invented deadline"),
    (r"\bact\s+(now|fast|today)\b", "pressure language"),
    (r"\bhurry\b", "pressure language"),
    (r"\bexpires?\s+(today|tonight|in\s+\d+)", "invented deadline"),
    (r"\blast\s+chance\b", "pressure language"),
]

_DISCOUNT = re.compile(r"\b(\d{1,2})%\s*(off|discount)|\bsave\s+\$?\d+|\bhalf[- ]price\b", re.I)
_PRICE = re.compile(r"\$\s?(\d[\d,]*)(?:\.(\d{2}))?")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PHONE = re.compile(r"\+?\d[\d\s().-]{8,}\d")


def check_copy(
    text: str,
    *,
    allowed_prices_cents: set[int] | None = None,
    max_chars: int = 1200,
) -> GuardrailReport:
    """Validate one piece of generated copy against the always-on rails."""
    report = GuardrailReport()
    lowered = text.lower()

    for pattern, label in _FORBIDDEN:
        match = re.search(pattern, lowered)
        if match:
            report.add("forbidden_claim", f"{label} ({match.group(0)!r})")

    for pattern, label in _FABRICATED_URGENCY:
        match = re.search(pattern, lowered)
        if match:
            report.add("fabricated_urgency", f"{label} ({match.group(0)!r})")

    if _DISCOUNT.search(text):
        # There is no discount field in the catalogue, so any discount is invented.
        report.add("invented_discount", _DISCOUNT.search(text).group(0))

    if allowed_prices_cents is not None:
        for match in _PRICE.finditer(text):
            dollars = int(match.group(1).replace(",", ""))
            cents = dollars * 100 + int(match.group(2) or 0)
            if cents not in allowed_prices_cents:
                report.add("wrong_price", f"{match.group(0)} is not a catalogue price")

    if _EMAIL.search(text) or _PHONE.search(text):
        report.add("pii", "contact details in generated copy")

    if len(text) > max_chars:
        report.add("too_long", f"{len(text)} chars > {max_chars}")

    return report


def scrub(text: str) -> str:
    """Last-resort cleanup for copy that failed twice: strip the offending sentences
    rather than showing the user a claim we know to be false."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = [s for s in sentences if check_copy(s).ok]
    return " ".join(kept).strip() or ""


class NemoRails:
    """Thin adapter over NeMo Guardrails, loaded lazily and only if configured.

    Kept behind `available` rather than raising on import so the app runs identically
    whether or not the optional extra is installed.
    """

    def __init__(self) -> None:
        self._rails = None
        self._tried = False

    @property
    def available(self) -> bool:
        if not settings.guardrails_llm_enabled:
            return False
        self._load()
        return self._rails is not None

    def _load(self) -> None:
        if self._tried:
            return
        self._tried = True
        config_dir = settings.guardrails_config_dir
        try:
            import os

            # NeMo's `openai` engine reads these. Pointing them at Mesh keeps the
            # brief's rule intact: every model call in this project goes through Mesh.
            os.environ.setdefault("OPENAI_API_KEY", settings.meshapi_api_key)
            os.environ.setdefault("OPENAI_BASE_URL", settings.meshapi_base_url)

            from nemoguardrails import LLMRails, RailsConfig

            self._rails = LLMRails(RailsConfig.from_path(str(config_dir)))
            log.info("nemo guardrails loaded", extra={"config": str(config_dir)})
        except Exception as exc:
            # An optional safety layer failing to load must not take the app down; the
            # deterministic rails above still run on every generation.
            log.warning("nemo guardrails unavailable", extra={"error": str(exc)[:200]})
            self._rails = None

    def check(self, text: str) -> GuardrailReport:
        report = GuardrailReport()
        if not self.available:
            return report
        try:
            result = self._rails.generate(
                messages=[{"role": "user", "content": f"Review this marketing copy: {text}"}]
            )
            content = (result or {}).get("content", "") if isinstance(result, dict) else str(result)
            if "BLOCKED" in content.upper():
                report.add("nemo", content[:200])
        except Exception as exc:
            log.warning("nemo check failed", extra={"error": str(exc)[:200]})
        return report


nemo = NemoRails()


def validate(text: str, allowed_prices_cents: set[int] | None = None) -> GuardrailReport:
    """Full check: deterministic rails always, NeMo only when explicitly enabled."""
    report = check_copy(text, allowed_prices_cents=allowed_prices_cents)
    if nemo.available:
        report.violations.extend(nemo.check(text).violations)
    return report
