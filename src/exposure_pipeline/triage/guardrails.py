"""Guardrails around the LLM call.

1. **Masking** - counterparty identifiers are pseudonymised with a keyed HMAC (same input ->
   same token, so the model can still see that two failing rows share a counterparty, but the
   token cannot be reversed without the secret salt); names are removed outright. A regex
   sweep catches ids embedded in free text (keys like "CP000123|2026-01-07"). A final leak
   check refuses to send a prompt that still contains any original sensitive value.
2. **Token budget** - the context is shrunk step by step (fewer sample rows, fewer keys,
   fewer related breaks) until it fits; if it still doesn't, no call is made.
3. **Schema validation** - model output is re-validated against ``TriageSuggestion``.
4. **Confidence gate** - below the threshold, the suggestion is downgraded to
   ``needs_manual_review`` (the model's guess is kept in the audit log, not applied).
"""
from __future__ import annotations

import copy
import hashlib
import hmac
import json
import re
from typing import Any, Callable, Iterable

from pydantic import ValidationError

from exposure_pipeline.triage.schemas import NEEDS_MANUAL_REVIEW, TriageSuggestion


class GuardrailError(RuntimeError):
    reason = "guardrail"


class PIILeakError(GuardrailError):
    reason = "pii_leak_blocked"


class TokenBudgetExceeded(GuardrailError):
    reason = "token_budget_exceeded"


class SchemaValidationError(GuardrailError):
    reason = "schema_validation_failed"


# ----------------------------------------------------------------------------- masking
ID_FIELDS: dict[str, str] = {
    "counterparty_id": "CP",
    "ultimate_parent_id": "CP",
    "parent_counterparty_id": "CP",
    "group_id": "GRP",
    "lei": "LEI",
}
# Field names are explicit (not just "name"): rule metadata also has a "name" key that must survive.
NAME_FIELDS = {"counterparty_name", "legal_name", "short_name", "contact_name"}
NAME_TOKEN = "<NAME_REDACTED>"
# Patterns for identifiers that may appear inside other strings (keys, messages).
TEXT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bCP\d{5,}\b"), "CP"),
    (re.compile(r"\bGRP\d{4,}\b"), "GRP"),
    (re.compile(r"\b[A-Z0-9]{18}\d{2}\b"), "LEI"),
]


class Masker:
    def __init__(self, salt: str):
        if not salt:
            raise ValueError("masking salt must not be empty")
        self._key = salt.encode()

    def token(self, value: str, prefix: str) -> str:
        digest = hmac.new(self._key, value.encode(), hashlib.sha256).hexdigest()[:10]
        return f"{prefix}#{digest}"

    def _mask_text(self, text: str, seen: set[str]) -> str:
        for pattern, prefix in TEXT_PATTERNS:
            def repl(m: re.Match, prefix: str = prefix) -> str:
                seen.add(m.group(0))
                return self.token(m.group(0), prefix)
            text = pattern.sub(repl, text)
        return text

    def _mask(self, obj: Any, seen: set[str], field: str | None = None) -> Any:
        if isinstance(obj, dict):
            return {k: self._mask(v, seen, k) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._mask(v, seen, field) for v in obj]
        if obj is None:
            return None
        if field in NAME_FIELDS:
            seen.add(str(obj))
            return NAME_TOKEN
        if field in ID_FIELDS:
            seen.add(str(obj))
            return self.token(str(obj), ID_FIELDS[field])
        if isinstance(obj, str):
            return self._mask_text(obj, seen)
        return obj

    def mask(self, obj: Any) -> tuple[Any, set[str]]:
        """Return (masked deep copy, set of original sensitive values that were replaced)."""
        seen: set[str] = set()
        masked = self._mask(copy.deepcopy(obj), seen)
        return masked, {s for s in seen if len(s) >= 4}

    @staticmethod
    def assert_no_leak(text: str, sensitive: Iterable[str]) -> None:
        leaked = [s for s in sensitive if s and s in text]
        if leaked:
            # Do not include the leaked values in the exception message (it may get logged).
            raise PIILeakError(f"{len(leaked)} sensitive value(s) still present in prompt")


# ----------------------------------------------------------------------------- tokens
def heuristic_token_count(text: str) -> int:
    """~4 characters per token for English/JSON; conservative enough for a budget check."""
    return max(1, len(text) // 4 + 1)


def tiktoken_counter(model: str) -> Callable[[str], int]:
    """Exact counter via tiktoken; falls back to the heuristic if the encoding is unavailable
    (tiktoken downloads encodings on first use, which may be blocked in locked-down envs)."""
    try:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("o200k_base")
        return lambda s: len(enc.encode(s))
    except Exception:  # noqa: BLE001
        return heuristic_token_count


# Ordered shrink steps: each returns a smaller copy of the context.
def _shrink_steps() -> list[Callable[[dict], dict]]:
    def records(n: int) -> Callable[[dict], dict]:
        def f(c: dict) -> dict:
            c = copy.deepcopy(c)
            c["sample_failing_records"] = c.get("sample_failing_records", [])[:n]
            return c
        return f

    def keys(n: int) -> Callable[[dict], dict]:
        def f(c: dict) -> dict:
            c = copy.deepcopy(c)
            c["failing_keys_sample"] = c.get("failing_keys_sample", [])[:n]
            return c
        return f

    def related(n: int) -> Callable[[dict], dict]:
        def f(c: dict) -> dict:
            c = copy.deepcopy(c)
            c["other_breaks_same_day"] = c.get("other_breaks_same_day", [])[:n]
            return c
        return f

    return [records(3), keys(5), related(5), records(1), keys(0), related(0), records(0)]


class TokenBudget:
    def __init__(self, max_input_tokens: int, counter: Callable[[str], int] = heuristic_token_count):
        self.max_input_tokens = max_input_tokens
        self.counter = counter

    def fit(self, context: dict, render: Callable[[dict], str]) -> tuple[dict, str, int]:
        """Return (context, rendered prompt, token count) within budget, shrinking as needed."""
        candidate = context
        text = render(candidate)
        n = self.counter(text)
        steps = iter(_shrink_steps())
        while n > self.max_input_tokens:
            step = next(steps, None)
            if step is None:
                raise TokenBudgetExceeded(f"prompt needs {n} tokens > budget {self.max_input_tokens}")
            candidate = step(candidate)
            text = render(candidate)
            n = self.counter(text)
        return candidate, text, n


# ----------------------------------------------------------------------------- output checks
def validate_suggestion(parsed: Any) -> TriageSuggestion:
    """Accept a TriageSuggestion, a dict, or a JSON string; raise SchemaValidationError otherwise."""
    try:
        if isinstance(parsed, TriageSuggestion):
            return TriageSuggestion.model_validate(parsed.model_dump())
        if isinstance(parsed, str):
            parsed = json.loads(parsed)
        if isinstance(parsed, dict):
            return TriageSuggestion.model_validate(parsed)
    except (ValidationError, json.JSONDecodeError, TypeError) as exc:
        raise SchemaValidationError(str(exc)[:500]) from exc
    raise SchemaValidationError(f"unexpected output type {type(parsed).__name__}")


def confidence_gate(suggestion: TriageSuggestion, threshold: float) -> tuple[str, str | None]:
    """Return (category to propose, fallback reason or None)."""
    if suggestion.confidence < threshold:
        return NEEDS_MANUAL_REVIEW, f"low_confidence ({suggestion.confidence:.2f} < {threshold:.2f})"
    return suggestion.category, None
