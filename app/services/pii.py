"""PII pattern detection, shared between the output-side guardrail check
(`guardrails.check_copy`) and input-side scrubbing of raw user text before it reaches a
prompt (`profile.evidence`/`summarise`).

One source of truth for the patterns: duplicating an email regex in two files means one
of them drifts the next time someone tightens it.
"""

import re

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PHONE = re.compile(r"\+?\d[\d\s().-]{8,}\d")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CARD = re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b")

_PATTERNS = [_EMAIL, _PHONE, _SSN, _CARD]


def contains_pii(text: str) -> bool:
    return any(p.search(text) for p in _PATTERNS)


def scrub_pii(text: str) -> str:
    """Redact PII-shaped substrings. Used at the point raw user text becomes something
    that reaches a prompt — not at ingestion, so the user's own event history stays
    intact."""
    for pattern in _PATTERNS:
        text = pattern.sub("[redacted]", text)
    return text
