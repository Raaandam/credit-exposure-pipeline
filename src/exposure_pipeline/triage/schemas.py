"""Output contract for the triage model.

The schema is passed to the model as a tool/function definition (structured output) AND
re-validated locally, so the model cannot return an unknown category, a confidence outside
[0, 1], or an essay. ``needs_manual_review`` is intentionally NOT a value the model may
return: it is assigned only by our guardrails, which keeps "the model was unsure" and "our
checks rejected the answer" distinguishable in the audit log.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from exposure_pipeline import reference as ref

LLMCategory = Literal[
    "missing_source_data",
    "mapping_error",
    "stale_feed",
    "duplicate",
    "unit_or_fx_mismatch",
    "genuine_business_change",
]
NEEDS_MANUAL_REVIEW = "needs_manual_review"
ALL_CATEGORIES: tuple[str, ...] = (*ref.BREAK_CATEGORIES, NEEDS_MANUAL_REVIEW)

CATEGORY_DEFINITIONS: dict[str, str] = {
    "missing_source_data": "Expected records or fields were not delivered (nulls, truncated file, missing FX row, parent record absent).",
    "mapping_error": "Values were delivered but do not map to our reference data (unknown codes, orphan ids, wrong code scheme).",
    "stale_feed": "The file delivered is not for this business date (re-delivered older extract, old as-of / effective dates).",
    "duplicate": "The same business key was delivered more than once.",
    "unit_or_fx_mismatch": "Amounts or rates use the wrong unit, scale or currency convention (thousands, pence, percent vs decimal).",
    "genuine_business_change": "The data is correct; the rule fired because of a real business event (e.g. credit-event downgrade).",
}


class TriageSuggestion(BaseModel):
    """Structured triage result for one data-quality break."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    category: LLMCategory = Field(description="Most likely root-cause category of the break.")
    confidence: float = Field(ge=0.0, le=1.0, description="Probability (0-1) that the category is correct.")
    root_cause_note: str = Field(
        min_length=10, max_length=1200,
        description="2-4 sentences explaining the most likely root cause, citing evidence from the context.",
    )
    suggested_fix: str = Field(
        min_length=3, max_length=800,
        description="Concrete next step for the data analyst or source system owner.",
    )
