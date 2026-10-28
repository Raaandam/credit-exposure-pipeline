"""LangChain + OpenAI triage chain with guardrails.

Flow per break::

    context --mask--> masked context --budget--> prompt --leak check--> LLM (temperature 0,
    structured output) --schema validation--> confidence gate --> TriageOutcome

Any failure along the way yields ``needs_manual_review`` with a ``fallback_reason``; the
chain never raises for a single bad break and never guesses silently.

The structured LLM is injected (``structured_llm``) so tests can use a stub and so the
provider can be swapped (e.g. Azure OpenAI via ``AzureChatOpenAI``) without touching the
guardrail logic. It must return LangChain's ``include_raw=True`` shape:
``{"raw": AIMessage, "parsed": TriageSuggestion | None, "parsing_error": Exception | None}``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from exposure_pipeline.config import Settings, get_settings
from exposure_pipeline.triage.guardrails import (
    GuardrailError,
    Masker,
    TokenBudget,
    confidence_gate,
    heuristic_token_count,
    tiktoken_counter,
    validate_suggestion,
)
from exposure_pipeline.triage.schemas import CATEGORY_DEFINITIONS, NEEDS_MANUAL_REVIEW, TriageSuggestion

log = logging.getLogger(__name__)

PROMPT_VERSION = "triage-v1.0"
TEMPERATURE = 0.0

SYSTEM_PROMPT = """You are a data-quality triage assistant for a bank's credit exposure pipeline.
The pipeline loads counterparty, facility, collateral, rating-change and FX feeds and checks
them with completeness, validity, uniqueness, referential-integrity, timeliness and
reconciliation rules. You receive ONE failed rule (a "break") with its evidence and must
classify the most likely root cause into exactly one category:

{categories}

Instructions:
- Base your answer only on the evidence provided. Do not invent identifiers, values or systems.
- Identifiers are pseudonymised (e.g. CP#1a2b3c4d5e); refer to them only by these tokens.
- Use the other breaks on the same day and the feed metadata (manifest business date,
  record counts, last successful load) to distinguish e.g. a truncated master file from bad ids.
- confidence is your probability that the category is correct. If the evidence is thin or
  points to several categories, give a lower confidence rather than guessing.
- root_cause_note: 2-4 factual sentences. suggested_fix: one concrete next step.
- You only suggest. A human analyst approves or rejects your suggestion."""

HUMAN_PROMPT = """Data-quality break to triage (JSON):
```json
{context_json}
```
Return the structured triage result."""


def build_prompt() -> ChatPromptTemplate:
    categories = "\n".join(f"- {k}: {v}" for k, v in CATEGORY_DEFINITIONS.items())
    return ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT.replace("{categories}", categories)), ("human", HUMAN_PROMPT)]
    )


def render_messages(prompt: ChatPromptTemplate, context: dict) -> tuple[list[BaseMessage], str]:
    messages = prompt.format_messages(context_json=json.dumps(context, indent=1, default=str, sort_keys=True))
    text = "\n\n".join(f"[{m.type}]\n{m.content}" for m in messages)
    return messages, text


def prompt_hash(model: str, text: str) -> str:
    """Identifies an exact (model, prompt version, prompt) combination for reproducibility."""
    return hashlib.sha256(f"{model}|{PROMPT_VERSION}|{TEMPERATURE}|{text}".encode()).hexdigest()


class StructuredLLM(Protocol):
    def invoke(self, messages: list[BaseMessage]) -> Any: ...


@dataclass
class TriageOutcome:
    suggested_category: str
    llm_category: str | None = None
    confidence: float | None = None
    root_cause_note: str | None = None
    suggested_fix: str | None = None
    fallback_reason: str | None = None
    prompt_text: str = ""
    prompt_hash: str = ""
    output_text: str | None = None
    model_requested: str = ""
    model_version: str | None = None
    temperature: float = TEMPERATURE
    prompt_version: str = PROMPT_VERSION
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None
    cost_usd: float | None = None
    extra: dict = field(default_factory=dict)

    @property
    def is_fallback(self) -> bool:
        return self.suggested_category == NEEDS_MANUAL_REVIEW


def _raw_output_text(raw: Any) -> str | None:
    if raw is None:
        return None
    tool_calls = getattr(raw, "tool_calls", None)
    if tool_calls:
        return json.dumps([{"name": t.get("name"), "args": t.get("args")} for t in tool_calls], default=str)
    content = getattr(raw, "content", raw)
    return content if isinstance(content, str) else json.dumps(content, default=str)


class TriageChain:
    def __init__(
        self,
        structured_llm: StructuredLLM,
        model_name: str,
        masker: Masker,
        budget: TokenBudget,
        confidence_threshold: float = 0.7,
        price_input_per_1m: float | None = None,
        price_output_per_1m: float | None = None,
    ):
        self.llm = structured_llm
        self.model_name = model_name
        self.masker = masker
        self.budget = budget
        self.threshold = confidence_threshold
        self.prices = (price_input_per_1m, price_output_per_1m)
        self.prompt = build_prompt()

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "TriageChain":
        settings = settings or get_settings()
        from langchain_openai import ChatOpenAI  # imported lazily: optional at test time

        llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=TEMPERATURE,
            max_tokens=settings.triage_max_output_tokens,
            timeout=30,
            max_retries=2,
        )
        # function_calling: widest model support; the Pydantic schema is re-validated locally anyway.
        structured = llm.with_structured_output(TriageSuggestion, method="function_calling", include_raw=True)
        counter = tiktoken_counter(settings.openai_model) if settings.triage_tokenizer == "tiktoken" else heuristic_token_count
        return cls(
            structured, settings.openai_model, Masker(settings.masking_salt),
            TokenBudget(settings.triage_max_input_tokens, counter), settings.triage_confidence_threshold,
            settings.price_input_per_1m, settings.price_output_per_1m,
        )

    def _cost(self, tin: int | None, tout: int | None) -> float | None:
        pin, pout = self.prices
        if pin is None or pout is None or tin is None or tout is None:
            return None
        return round(tin / 1e6 * pin + tout / 1e6 * pout, 8)

    def triage(self, context: dict) -> TriageOutcome:
        out = TriageOutcome(suggested_category=NEEDS_MANUAL_REVIEW, model_requested=self.model_name)

        # 1-3. mask -> fit budget -> leak check (no API call if any of these fail)
        masked, sensitive = self.masker.mask(context)
        try:
            fitted, text, n_tokens = self.budget.fit(masked, lambda c: render_messages(self.prompt, c)[1])
            self.masker.assert_no_leak(text, sensitive)
        except GuardrailError as exc:
            out.fallback_reason = f"{exc.reason}: {exc}"
            out.prompt_text = "[not sent: blocked by guardrail]" if exc.reason == "pii_leak_blocked" else ""
            return out
        messages, out.prompt_text = render_messages(self.prompt, fitted)
        out.prompt_hash = prompt_hash(self.model_name, out.prompt_text)
        out.extra["estimated_input_tokens"] = n_tokens

        # 4. call the model
        t0 = time.perf_counter()
        try:
            response = self.llm.invoke(messages)
        except Exception as exc:  # noqa: BLE001 - timeouts, rate limits, auth, network
            out.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            out.fallback_reason = f"llm_error: {type(exc).__name__}"
            log.warning("triage LLM call failed: %s", exc)
            return out
        out.latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        raw, parsed, parsing_error = None, response, None
        if isinstance(response, dict) and "raw" in response:
            raw, parsed, parsing_error = response.get("raw"), response.get("parsed"), response.get("parsing_error")
        out.output_text = _raw_output_text(raw) if raw is not None else (
            parsed.model_dump_json() if isinstance(parsed, TriageSuggestion) else json.dumps(parsed, default=str))
        meta = getattr(raw, "response_metadata", None) or {}
        usage = getattr(raw, "usage_metadata", None) or {}
        out.model_version = meta.get("model_name") or meta.get("model")
        out.input_tokens, out.output_tokens = usage.get("input_tokens"), usage.get("output_tokens")
        out.cost_usd = self._cost(out.input_tokens, out.output_tokens)

        # 5. schema validation (fall back to the raw tool-call args if LangChain failed to parse)
        if parsed is None and raw is not None and getattr(raw, "tool_calls", None):
            parsed = raw.tool_calls[0].get("args")
        try:
            if parsed is None:
                raise GuardrailError(f"no parsed output ({type(parsing_error).__name__ if parsing_error else 'empty'})")
            suggestion = validate_suggestion(parsed)
        except GuardrailError as exc:
            out.fallback_reason = f"schema_validation_failed: {exc}"[:500]
            return out

        # 6. confidence gate
        out.llm_category = suggestion.category
        out.confidence = suggestion.confidence
        out.root_cause_note = suggestion.root_cause_note
        out.suggested_fix = suggestion.suggested_fix
        out.suggested_category, out.fallback_reason = confidence_gate(suggestion, self.threshold)
        return out
